//! Stateful Raft properties driven by state-dependent cluster commands.
//!
//! Commands are only drawn when their preconditions hold, and the five
//! Raft safety properties are checked after every effective transition.
//! The oracle retains leaders and committed entries across the complete
//! case so sequential violations cannot disappear with current state.

pub mod pbt_harness;

use noraft::{
    Action, ClusterConfig, Log, LogEntry, LogIndex, LogPosition, Message, Node, NodeId, Role, Term,
};
use pbt_harness::run_config;
use std::cell::Cell;
use std::collections::{BTreeMap, VecDeque};
use std::fmt;

const MIN_STEPS: usize = 20;
const MAX_STEPS: usize = 200;

#[derive(Debug, Clone, Copy)]
enum Cmd {
    TickElection(NodeId),
    DeliverNext(NodeId),
    DuplicateNext(NodeId),
    DropNext(NodeId),
    Crash(NodeId),
    Restart(NodeId),
    Propose(NodeId),
    TakeSnapshot(NodeId),
    DeliverSnapshot(NodeId),
    DuplicateSnapshot(NodeId),
    DropSnapshot(NodeId),
    PersistStorage(NodeId),
}

impl fmt::Display for Cmd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (name, id) = match self {
            Self::TickElection(id) => ("TickElection", id),
            Self::DeliverNext(id) => ("DeliverNext", id),
            Self::DuplicateNext(id) => ("DuplicateNext", id),
            Self::DropNext(id) => ("DropNext", id),
            Self::Crash(id) => ("Crash", id),
            Self::Restart(id) => ("Restart", id),
            Self::Propose(id) => ("Propose", id),
            Self::TakeSnapshot(id) => ("TakeSnapshot", id),
            Self::DeliverSnapshot(id) => ("DeliverSnapshot", id),
            Self::DuplicateSnapshot(id) => ("DuplicateSnapshot", id),
            Self::DropSnapshot(id) => ("DropSnapshot", id),
            Self::PersistStorage(id) => ("PersistStorage", id),
        };
        write!(f, "{name}({})", u64::from(*id))
    }
}

#[derive(Debug, Clone, Copy)]
enum CmdKind {
    TickElection,
    DeliverNext,
    DuplicateNext,
    DropNext,
    Crash,
    Restart,
    Propose,
    TakeSnapshot,
    DeliverSnapshot,
    DuplicateSnapshot,
    DropSnapshot,
    PersistStorage,
}

/// Test-only durable snapshot mirroring the three inputs `Node::restart`
/// requires (`current_term`, `voted_for`, `log`).
#[derive(Debug, Clone)]
struct DurableSnapshot {
    term: Term,
    voted_for: Option<NodeId>,
    log: Log,
}

impl DurableSnapshot {
    fn from_node(node: &Node) -> Self {
        Self {
            term: node.current_term(),
            voted_for: node.voted_for(),
            log: node.log().clone(),
        }
    }
}

/// In-flight storage transaction on the step-driven cluster harness.
/// `target` is the state that will replace `durable_states[id]` when
/// `Cmd::PersistStorage(id)` fires. Outbound produced by the same node
/// while `pending_storage` exists is buffered here and released at the
/// same time.
///
/// `outbound_snapshots` carries the `(LogPosition, ClusterConfig)`
/// live on the source node at hold time. Reading them at release time
/// instead would deliver a boundary the source never asked the user
/// to send, because `Cmd::TakeSnapshot` on the same node can advance
/// `snapshot_position` while the transfer is still held.
#[derive(Debug)]
struct PendingStorage {
    target: DurableSnapshot,
    outbound_messages: Vec<(NodeId, Message)>,
    outbound_snapshots: Vec<(NodeId, LogPosition, ClusterConfig)>,
}

impl PendingStorage {
    fn new(target: DurableSnapshot) -> Self {
        Self {
            target,
            outbound_messages: Vec::new(),
            outbound_snapshots: Vec::new(),
        }
    }
}

struct CrashedNode {
    durable: DurableSnapshot,
}

struct Cluster {
    nodes: BTreeMap<NodeId, Node>,
    queues: BTreeMap<NodeId, VecDeque<Message>>,
    snapshot_queues: BTreeMap<NodeId, VecDeque<(LogPosition, ClusterConfig)>>,
    crashed: BTreeMap<NodeId, CrashedNode>,
    // Last state that has been fully persisted for each node. Restart
    // restores the node from here, and `Cmd::PersistStorage` refreshes
    // it from `pending_storage[id].target`.
    durable_states: BTreeMap<NodeId, DurableSnapshot>,
    // In-flight storage transaction plus the outbound produced by the
    // same node while it is pending. Cleared on crash (with its held
    // outbound) or on `Cmd::PersistStorage` (with its held outbound
    // released into `queues` / `snapshot_queues`).
    pending_storage: BTreeMap<NodeId, PendingStorage>,
}

impl Cluster {
    fn bootstrap(node_ids: &[NodeId]) -> Result<Self, String> {
        let mut nodes: BTreeMap<NodeId, Node> = node_ids
            .iter()
            .copied()
            .map(|id| (id, Node::start(id)))
            .collect();
        let queues = node_ids
            .iter()
            .copied()
            .map(|id| (id, VecDeque::new()))
            .collect();
        let snapshot_queues = node_ids
            .iter()
            .copied()
            .map(|id| (id, VecDeque::new()))
            .collect();
        // Seed each node's initial durable state from `Node::start`. The
        // bootstrap node's `create_cluster` below extends its live log
        // and the follow-up `drain_actions` + explicit persist stashes
        // that extension into `durable_states[bootstrap_id]`.
        let durable_states: BTreeMap<NodeId, DurableSnapshot> = nodes
            .iter()
            .map(|(id, node)| (*id, DurableSnapshot::from_node(node)))
            .collect();
        let bootstrap_id = *node_ids
            .first()
            .ok_or("a cluster needs at least one node")?;
        let position = nodes
            .get_mut(&bootstrap_id)
            .expect("the bootstrap node was inserted")
            .create_cluster(node_ids);
        if position == LogPosition::INVALID {
            return Err("cluster bootstrap returned an invalid position".into());
        }

        let mut cluster = Self {
            nodes,
            queues,
            snapshot_queues,
            crashed: BTreeMap::new(),
            durable_states,
            pending_storage: BTreeMap::new(),
        };
        cluster.drain_actions(bootstrap_id)?;
        // Force-persist the bootstrap transaction so the cluster starts
        // with the initial cluster-config entry committed to durable
        // state. Without this, the very first `Cmd::Crash(bootstrap_id)`
        // would resurrect an empty log and stall the cluster.
        cluster.persist_pending(bootstrap_id);
        Ok(cluster)
    }

    // Fold a successful `Node::handle_snapshot_installed` into the
    // durable state so a subsequent crash preserves the boundary. The
    // API's precondition treats the snapshot as already persisted by
    // the user (`src/node.rs::Node::handle_snapshot_installed` /
    // `Node::restart` doc), so mirroring the tick harness's behavior:
    // `durable_states[id]` is updated atomically, and if a pending
    // transaction is in flight its target is refreshed to the same
    // post-install live state (the old target's log prefix has just
    // been superseded by the snapshot).
    fn fold_snapshot_install_into_durable(&mut self, id: NodeId) {
        let node = self
            .nodes
            .get(&id)
            .expect("caller runs on a live node right after the install");
        let snapshot = DurableSnapshot::from_node(node);
        self.durable_states.insert(id, snapshot.clone());
        if let Some(pending) = self.pending_storage.get_mut(&id) {
            pending.target = snapshot;
        }
    }

    // Commit `pending_storage[id]` into `durable_states[id]` and release
    // any outbound that was held for the transaction. No-op if no
    // pending exists. Used by bootstrap and by `Cmd::PersistStorage`.
    fn persist_pending(&mut self, id: NodeId) {
        let Some(pending) = self.pending_storage.remove(&id) else {
            return;
        };
        self.durable_states.insert(id, pending.target);
        for (destination, message) in pending.outbound_messages {
            if let Some(queue) = self.queues.get_mut(&destination) {
                queue.push_back(message);
            }
        }
        for (destination, position, config) in pending.outbound_snapshots {
            if let Some(queue) = self.snapshot_queues.get_mut(&destination) {
                queue.push_back((position, config));
            }
        }
    }

    fn drain_actions(&mut self, id: NodeId) -> Result<(), String> {
        let peers: Vec<NodeId> = self
            .nodes
            .get(&id)
            .ok_or_else(|| format!("node {} is not running", u64::from(id)))?
            .peers()
            .collect();
        let actions: Vec<Action> = self
            .nodes
            .get_mut(&id)
            .expect("the node was checked above")
            .actions_mut()
            .collect();

        // Update or open a pending storage transaction if any storage
        // action is present in this batch. The transaction target always
        // reflects the latest live Node state, so subsequent commands
        // that add more storage without a `Cmd::PersistStorage` in
        // between naturally accumulate into the same transaction.
        let has_storage_action = actions.iter().any(|a| {
            matches!(
                a,
                Action::SaveCurrentTerm | Action::SaveVotedFor | Action::AppendLogEntries(_)
            )
        });
        if has_storage_action {
            let target = {
                let source = self
                    .nodes
                    .get(&id)
                    .expect("drain_actions is called for a running node");
                DurableSnapshot::from_node(source)
            };
            self.pending_storage
                .entry(id)
                .and_modify(|p| p.target = target.clone())
                .or_insert_with(|| PendingStorage::new(target));
        }

        // Outbound is held for the entire time a pending storage
        // transaction exists for this node, not only for the same batch
        // that opened it: a later outbound whose payload reflects
        // uncommitted state must not be exposed to peers until that
        // state is durable.
        let holding = self.pending_storage.contains_key(&id);

        for action in actions {
            match action {
                Action::SetElectionTimeout
                | Action::SaveCurrentTerm
                | Action::SaveVotedFor
                | Action::AppendLogEntries(_) => {
                    // Storage side is captured in `pending_storage`
                    // above; `SetElectionTimeout` has no bearing on
                    // durable state.
                }
                Action::BroadcastMessage(message) => {
                    if holding {
                        let pending = self
                            .pending_storage
                            .get_mut(&id)
                            .expect("holding implies pending exists");
                        for peer in &peers {
                            pending.outbound_messages.push((*peer, message.clone()));
                        }
                    } else {
                        for peer in &peers {
                            if let Some(queue) = self.queues.get_mut(peer) {
                                queue.push_back(message.clone());
                            }
                        }
                    }
                }
                Action::SendMessage(to, message) => {
                    if holding {
                        let pending = self
                            .pending_storage
                            .get_mut(&id)
                            .expect("holding implies pending exists");
                        pending.outbound_messages.push((to, message));
                    } else if let Some(queue) = self.queues.get_mut(&to) {
                        queue.push_back(message);
                    }
                }
                Action::InstallSnapshot(to) => {
                    let (position, config) = {
                        let source = self
                            .nodes
                            .get(&id)
                            .expect("drain_actions is called for a running node");
                        (
                            source.log().snapshot_position(),
                            source.log().snapshot_config().clone(),
                        )
                    };
                    if holding {
                        let pending = self
                            .pending_storage
                            .get_mut(&id)
                            .expect("holding implies pending exists");
                        pending.outbound_snapshots.push((to, position, config));
                    } else if let Some(queue) = self.snapshot_queues.get_mut(&to) {
                        queue.push_back((position, config));
                    }
                }
            }
        }
        Ok(())
    }

    fn running_ids(&self) -> Vec<NodeId> {
        self.nodes.keys().copied().collect()
    }

    fn crashed_ids(&self) -> Vec<NodeId> {
        self.crashed.keys().copied().collect()
    }

    fn queued_ids(&self) -> Vec<NodeId> {
        self.queues
            .iter()
            .filter_map(|(id, queue)| (!queue.is_empty()).then_some(*id))
            .collect()
    }

    fn snapshot_queued_ids(&self) -> Vec<NodeId> {
        self.snapshot_queues
            .iter()
            .filter_map(|(id, queue)| (!queue.is_empty()).then_some(*id))
            .collect()
    }

    fn leader_ids(&self) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter_map(|(id, node)| (node.role() == Role::Leader).then_some(*id))
            .collect()
    }

    // Nodes whose `commit_index` has advanced past their current snapshot
    // boundary. `TakeSnapshot` is only offered for these so the install
    // strictly advances `snapshot_position`, and the bootstrap-time
    // `commit_index == snapshot_position == LogIndex::ZERO` case is excluded.
    fn snapshot_takers(&self) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter_map(|(id, node)| {
                (node.commit_index() > node.log().snapshot_position().index).then_some(*id)
            })
            .collect()
    }

    fn persist_pending_ids(&self) -> Vec<NodeId> {
        self.pending_storage.keys().copied().collect()
    }

    fn sample_command(&self, ctx: &mut noprop::TestCaseContext) -> Cmd {
        let running = self.running_ids();
        let crashed = self.crashed_ids();
        let queued = self.queued_ids();
        let snapshot_queued = self.snapshot_queued_ids();
        let leaders = self.leader_ids();
        let snapshot_takers = self.snapshot_takers();
        let persistable = self.persist_pending_ids();
        let mut kinds = Vec::new();
        let mut weights = Vec::new();

        kinds.push(CmdKind::TickElection);
        weights.push(3);
        if !queued.is_empty() {
            kinds.push(CmdKind::DeliverNext);
            weights.push(12);
            kinds.push(CmdKind::DuplicateNext);
            weights.push(1);
            kinds.push(CmdKind::DropNext);
            weights.push(1);
        }
        if running.len() > 1 {
            kinds.push(CmdKind::Crash);
            weights.push(1);
        }
        if !crashed.is_empty() {
            kinds.push(CmdKind::Restart);
            weights.push(2);
        }
        if !leaders.is_empty() {
            kinds.push(CmdKind::Propose);
            weights.push(5);
        }
        if !snapshot_takers.is_empty() {
            kinds.push(CmdKind::TakeSnapshot);
            weights.push(2);
        }
        if !snapshot_queued.is_empty() {
            kinds.push(CmdKind::DeliverSnapshot);
            weights.push(6);
            kinds.push(CmdKind::DuplicateSnapshot);
            weights.push(1);
            // Bumped from 1 to 3 to keep `cases_with_snapshot_dropped`
            // reliable after `Cmd::PersistStorage` diluted the
            // per-selection probability of the low-weight snapshot
            // commands. DropSnapshot only fires when
            // `snapshot_queued` is non-empty, which is a rare event
            // that does not scale linearly with the case budget.
            kinds.push(CmdKind::DropSnapshot);
            weights.push(3);
        }
        if !persistable.is_empty() {
            // Kept modest so the harness's overall weight distribution
            // (and the coverage assertions calibrated on it in
            // `snapshot_dropped` etc.) is not disturbed too much, while
            // still commonly persisting pending storage before the run
            // stalls under held outbound.
            kinds.push(CmdKind::PersistStorage);
            weights.push(5);
        }

        let kind = kinds[noprop::sample_weighted_index(ctx, &weights)];
        match kind {
            CmdKind::TickElection => Cmd::TickElection(noprop::sample_choice(ctx, &running)),
            CmdKind::DeliverNext => Cmd::DeliverNext(noprop::sample_choice(ctx, &queued)),
            CmdKind::DuplicateNext => Cmd::DuplicateNext(noprop::sample_choice(ctx, &queued)),
            CmdKind::DropNext => Cmd::DropNext(noprop::sample_choice(ctx, &queued)),
            CmdKind::Crash => Cmd::Crash(noprop::sample_choice(ctx, &running)),
            CmdKind::Restart => Cmd::Restart(noprop::sample_choice(ctx, &crashed)),
            CmdKind::Propose => Cmd::Propose(noprop::sample_choice(ctx, &leaders)),
            CmdKind::TakeSnapshot => {
                Cmd::TakeSnapshot(noprop::sample_choice(ctx, &snapshot_takers))
            }
            CmdKind::DeliverSnapshot => {
                Cmd::DeliverSnapshot(noprop::sample_choice(ctx, &snapshot_queued))
            }
            CmdKind::DuplicateSnapshot => {
                Cmd::DuplicateSnapshot(noprop::sample_choice(ctx, &snapshot_queued))
            }
            CmdKind::DropSnapshot => {
                Cmd::DropSnapshot(noprop::sample_choice(ctx, &snapshot_queued))
            }
            CmdKind::PersistStorage => {
                Cmd::PersistStorage(noprop::sample_choice(ctx, &persistable))
            }
        }
    }

    // Returns `Ok(true)` iff this command called
    // `Node::handle_snapshot_installed` on a remote-transferred snapshot
    // (`DeliverSnapshot` / `DuplicateSnapshot`) and got `true` back.
    // Every other command returns `Ok(false)` on success. Feeds
    // `cases_with_remote_snapshot_installed`.
    fn apply(&mut self, command: Cmd) -> Result<bool, String> {
        match command {
            Cmd::TickElection(id) => {
                self.nodes
                    .get_mut(&id)
                    .expect("commands only select running nodes")
                    .handle_election_timeout();
                self.drain_actions(id)?;
                Ok(false)
            }
            Cmd::DeliverNext(id) => {
                let message = self
                    .queues
                    .get_mut(&id)
                    .and_then(VecDeque::pop_front)
                    .expect("commands only select non-empty queues");
                self.nodes
                    .get_mut(&id)
                    .expect("queues only exist for running nodes")
                    .handle_message(&message)
                    .map_err(|error| {
                        format!(
                            "node {} rejected a harness-produced message {message:?}: {error:?}",
                            u64::from(id)
                        )
                    })?;
                self.drain_actions(id)?;
                Ok(false)
            }
            Cmd::DuplicateNext(id) => {
                let message = self
                    .queues
                    .get(&id)
                    .and_then(|queue| queue.front().cloned())
                    .expect("commands only select non-empty queues");
                self.nodes
                    .get_mut(&id)
                    .expect("queues only exist for running nodes")
                    .handle_message(&message)
                    .map_err(|error| {
                        format!(
                            "node {} rejected a duplicated harness message {message:?}: {error:?}",
                            u64::from(id)
                        )
                    })?;
                self.drain_actions(id)?;
                Ok(false)
            }
            Cmd::DropNext(id) => {
                self.queues
                    .get_mut(&id)
                    .and_then(VecDeque::pop_front)
                    .expect("commands only select non-empty queues");
                Ok(false)
            }
            Cmd::Crash(id) => {
                self.nodes
                    .remove(&id)
                    .expect("commands only select running nodes");
                self.queues.remove(&id);
                self.snapshot_queues.remove(&id);
                // Drop any in-flight storage transaction (and its held
                // outbound) so the restart cannot resurrect an
                // unpersisted change. `durable_states[id]` is what the
                // node will be resurrected from.
                self.pending_storage.remove(&id);
                let durable = self
                    .durable_states
                    .get(&id)
                    .cloned()
                    .expect("durable state is seeded for every node at bootstrap");
                self.crashed.insert(id, CrashedNode { durable });
                Ok(false)
            }
            Cmd::Restart(id) => {
                let state = self
                    .crashed
                    .remove(&id)
                    .expect("commands only select crashed nodes");
                let node = Node::restart(
                    id,
                    state.durable.term,
                    state.durable.voted_for,
                    state.durable.log,
                );
                self.nodes.insert(id, node);
                self.queues.insert(id, VecDeque::new());
                self.snapshot_queues.insert(id, VecDeque::new());
                self.drain_actions(id)?;
                Ok(false)
            }
            Cmd::Propose(id) => {
                let position = self
                    .nodes
                    .get_mut(&id)
                    .expect("commands only select running leaders")
                    .propose_command();
                if position == LogPosition::INVALID {
                    return Err(format!(
                        "leader {} rejected a command proposal",
                        u64::from(id)
                    ));
                }
                self.drain_actions(id)?;
                Ok(false)
            }
            Cmd::TakeSnapshot(id) => {
                let (position, config) = {
                    let node = self
                        .nodes
                        .get(&id)
                        .expect("commands only select running nodes");
                    let commit_index = node.commit_index();
                    node.log()
                        .get_position_and_config(commit_index)
                        .map(|(pos, cfg)| (pos, cfg.clone()))
                        .expect("TakeSnapshot precondition guarantees commit_index is in log range")
                };
                let node = self
                    .nodes
                    .get_mut(&id)
                    .expect("commands only select running nodes");
                let ok = node.handle_snapshot_installed(position, config);
                if !ok {
                    return Err(format!(
                        "TakeSnapshot: node {} rejected the local snapshot despite \
                         the sample-time precondition",
                        u64::from(id),
                    ));
                }
                self.fold_snapshot_install_into_durable(id);
                self.drain_actions(id)?;
                Ok(false)
            }
            Cmd::DeliverSnapshot(id) => {
                let (position, config) = self
                    .snapshot_queues
                    .get_mut(&id)
                    .and_then(VecDeque::pop_front)
                    .expect("commands only select non-empty snapshot queues");
                // `handle_snapshot_installed` returning `false` is a valid
                // no-op contract (term not yet caught up, or log disagrees
                // with the boundary). The return value flows through to
                // `cases_with_remote_snapshot_installed`.
                let ok = self
                    .nodes
                    .get_mut(&id)
                    .expect("snapshot queues only exist for running nodes")
                    .handle_snapshot_installed(position, config);
                if ok {
                    self.fold_snapshot_install_into_durable(id);
                }
                self.drain_actions(id)?;
                Ok(ok)
            }
            Cmd::DuplicateSnapshot(id) => {
                let (position, config) = self
                    .snapshot_queues
                    .get(&id)
                    .and_then(|queue| queue.front().cloned())
                    .expect("commands only select non-empty snapshot queues");
                let ok = self
                    .nodes
                    .get_mut(&id)
                    .expect("snapshot queues only exist for running nodes")
                    .handle_snapshot_installed(position, config);
                if ok {
                    self.fold_snapshot_install_into_durable(id);
                }
                self.drain_actions(id)?;
                Ok(ok)
            }
            Cmd::DropSnapshot(id) => {
                self.snapshot_queues
                    .get_mut(&id)
                    .and_then(VecDeque::pop_front)
                    .expect("commands only select non-empty snapshot queues");
                Ok(false)
            }
            Cmd::PersistStorage(id) => {
                self.persist_pending(id);
                Ok(false)
            }
        }
    }
}

#[derive(Default)]
struct InvariantState {
    leaders_by_term: BTreeMap<Term, NodeId>,
    leader_logs: BTreeMap<(NodeId, Term), Vec<(LogPosition, LogEntry)>>,
    // Position, entry value, and the term in which the commit was
    // first observed. The observation term matters for leader
    // completeness when an older-term entry is committed by a newer
    // leader.
    committed: BTreeMap<LogIndex, (LogPosition, LogEntry, Term)>,
    leader_with_committed_history_seen: bool,
    // Highest match_index observed per (leader, term) tenure, used by
    // `check_follower_match_index_monotonic` to detect within-tenure
    // regressions. Safe only while `Cmd` does not include membership
    // changes: a follower dropped from the configuration and later
    // re-added during the same tenure would legitimately reset its
    // match_index to 0 and trigger a false positive here.
    follower_match_indexes: BTreeMap<(NodeId, Term), BTreeMap<NodeId, LogIndex>>,
    // Term ever observed as a `snapshot_position.term` at a given index.
    // `check_snapshot_boundary_consistency` uses it to catch bugs that
    // corrupt the boundary term across time or across nodes; the per-step
    // `check_*` methods above never look at the boundary (`iter_with_positions`
    // skips it and the relaxed prefix comparisons intentionally stop at
    // `common_start`).
    snapshot_boundary_terms: BTreeMap<LogIndex, Term>,
    // Set to `true` whenever `check_snapshot_boundary_consistency`
    // actually compared a boundary against another node's log or the
    // committed oracle (i.e. the cross-node / cross-time comparison
    // fired at least once). Feeds `cases_with_boundary_cross_check`.
    boundary_cross_check_fired: bool,
}

impl InvariantState {
    fn check(&mut self, cluster: &Cluster) -> Result<(), String> {
        self.check_election_safety(cluster)?;
        self.check_log_matching(cluster)?;
        self.check_leader_append_only(cluster)?;
        self.check_state_machine_safety(cluster)?;
        self.check_leader_completeness(cluster)?;
        self.check_follower_match_index_monotonic(cluster)?;
        self.check_snapshot_boundary_consistency(cluster)?;
        Ok(())
    }

    fn check_election_safety(&mut self, cluster: &Cluster) -> Result<(), String> {
        for (id, node) in &cluster.nodes {
            if node.role() != Role::Leader {
                continue;
            }
            let term = node.current_term();
            if let Some(previous) = self.leaders_by_term.insert(term, *id)
                && previous != *id
            {
                return Err(format!(
                    "election safety violated: nodes {} and {} were leaders in term {:?}",
                    u64::from(previous),
                    u64::from(*id),
                    term
                ));
            }
        }
        Ok(())
    }

    fn check_log_matching(&self, cluster: &Cluster) -> Result<(), String> {
        let nodes: Vec<(&NodeId, &Node)> = cluster.nodes.iter().collect();
        for left_index in 0..nodes.len() {
            for right_index in left_index + 1..nodes.len() {
                let (left_id, left) = nodes[left_index];
                let (right_id, right) = nodes[right_index];
                let left_entries: Vec<(LogPosition, LogEntry)> = left
                    .log()
                    .entries()
                    .iter_with_positions()
                    .map(|(position, entry)| (position, entry.clone()))
                    .collect();
                let right_entries: Vec<(LogPosition, LogEntry)> = right
                    .log()
                    .entries()
                    .iter_with_positions()
                    .map(|(position, entry)| (position, entry.clone()))
                    .collect();

                // The lower bound below which entries have been absorbed by
                // one side's snapshot. Log matching is only meaningful
                // strictly above this boundary: indices at or below can be
                // present on the un-snapshotted side but gone on the other,
                // which is not a violation.
                let common_start = left
                    .log()
                    .snapshot_position()
                    .index
                    .max(right.log().snapshot_position().index);

                for (shared_position, _) in &left_entries {
                    if shared_position.index <= common_start {
                        continue;
                    }
                    let same_position = right_entries
                        .iter()
                        .any(|(position, _)| position == shared_position);
                    if !same_position {
                        continue;
                    }
                    let left_prefix: Vec<_> = left_entries
                        .iter()
                        .filter(|(position, _)| position.index > common_start)
                        .take_while(|(position, _)| position.index <= shared_position.index)
                        .collect();
                    let right_prefix: Vec<_> = right_entries
                        .iter()
                        .filter(|(position, _)| position.index > common_start)
                        .take_while(|(position, _)| position.index <= shared_position.index)
                        .collect();
                    if left_prefix != right_prefix {
                        return Err(format!(
                            "log matching violated for nodes {} and {} at {shared_position:?}: \
                             left={left_prefix:?}, right={right_prefix:?}",
                            u64::from(*left_id),
                            u64::from(*right_id)
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    fn check_leader_append_only(&mut self, cluster: &Cluster) -> Result<(), String> {
        for (id, node) in &cluster.nodes {
            if node.role() != Role::Leader {
                continue;
            }
            let key = (*id, node.current_term());
            let current: Vec<(LogPosition, LogEntry)> = node
                .log()
                .entries()
                .iter_with_positions()
                .map(|(position, entry)| (position, entry.clone()))
                .collect();
            // A local snapshot taken within the same tenure legitimately
            // deletes a prefix of the previously observed log. Compare only
            // the portion strictly above the current snapshot boundary.
            let current_snapshot = node.log().snapshot_position().index;
            if let Some(previous) = self.leader_logs.get(&key) {
                let previous_suffix: Vec<(LogPosition, LogEntry)> = previous
                    .iter()
                    .filter(|(position, _)| position.index > current_snapshot)
                    .cloned()
                    .collect();
                if !current.starts_with(&previous_suffix) {
                    return Err(format!(
                        "leader {} changed its term {:?} log prefix: previous={previous:?}, \
                         current={current:?}",
                        u64::from(*id),
                        node.current_term()
                    ));
                }
            }
            self.leader_logs.insert(key, current);
        }
        Ok(())
    }

    fn check_state_machine_safety(&mut self, cluster: &Cluster) -> Result<(), String> {
        for (id, node) in &cluster.nodes {
            for (position, entry) in node.log().entries().iter_with_positions() {
                if position.index > node.commit_index() {
                    break;
                }
                match self.committed.get(&position.index) {
                    Some((previous_position, previous_entry, _))
                        if previous_position != &position || previous_entry != &entry =>
                    {
                        return Err(format!(
                            "state machine safety violated at {:?}: node {} has ({position:?}, \
                             {entry:?}), history has ({previous_position:?}, {previous_entry:?})",
                            position.index,
                            u64::from(*id)
                        ));
                    }
                    None => {
                        self.committed.insert(
                            position.index,
                            (position, entry.clone(), node.current_term()),
                        );
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    fn check_leader_completeness(&mut self, cluster: &Cluster) -> Result<(), String> {
        for (id, node) in &cluster.nodes {
            if node.role() != Role::Leader {
                continue;
            }
            if self
                .committed
                .values()
                .any(|(_, _, commit_term)| *commit_term <= node.current_term())
            {
                self.leader_with_committed_history_seen = true;
            }
            for (index, expected) in &self.committed {
                // A stale leader from an older term is not required to
                // contain entries committed later. Leader completeness
                // applies from the term in which the commit occurred.
                if node.current_term() < expected.2 {
                    continue;
                }
                // Committed entries at or below this leader's snapshot
                // boundary are absorbed by the snapshot itself, so the
                // per-entry log lookup is not required (and would
                // legitimately return `None`).
                if *index <= node.log().snapshot_position().index {
                    continue;
                }
                let actual = node.log().entries().get_entry(*index).map(|entry| {
                    (
                        LogPosition::new(
                            node.log()
                                .entries()
                                .get_term(*index)
                                .expect("an existing entry must have a term"),
                            *index,
                        ),
                        entry.clone(),
                    )
                });
                let expected_entry = (&expected.0, &expected.1);
                if actual.as_ref().map(|(position, entry)| (position, entry))
                    != Some(expected_entry)
                {
                    return Err(format!(
                        "leader completeness violated: leader {} lacks committed entry at \
                         {index:?}; actual={actual:?}, expected={expected:?}",
                        u64::from(*id)
                    ));
                }
            }
        }
        Ok(())
    }

    // Within a single (leader, term) tenure, `follower_match_index` must not
    // regress. `(leader_id, term)` uniquely identifies a tenure because every
    // path to `transition_to_leader` goes through `transition_to_candidate`,
    // which bumps `current_term`.
    fn check_follower_match_index_monotonic(&mut self, cluster: &Cluster) -> Result<(), String> {
        for (leader_id, node) in &cluster.nodes {
            if node.role() != Role::Leader {
                continue;
            }
            let term = node.current_term();
            let recorded = self
                .follower_match_indexes
                .entry((*leader_id, term))
                .or_default();
            for follower_id in node.peers() {
                let Some(current) = node.follower_match_index(follower_id) else {
                    continue;
                };
                if let Some(previous) = recorded.get(&follower_id)
                    && current < *previous
                {
                    return Err(format!(
                        "follower_match_index regressed within a single tenure: leader {} in \
                         {term:?}, follower {}: previous={previous:?}, current={current:?}",
                        u64::from(*leader_id),
                        u64::from(follower_id)
                    ));
                }
                recorded.insert(follower_id, current);
            }
        }
        Ok(())
    }

    // The snapshot boundary `(term, index)` is a committed marker but is
    // not returned by `iter_with_positions`, and the relaxed prefix
    // comparisons in `check_log_matching` / `check_leader_completeness`
    // stop strictly above it. Cross-check it against three references:
    // the same index observed at a previous step (cross-time), every
    // other node's log at that index (cross-node, using
    // `entries().get_term` which includes the peer's own boundary if
    // any), and the `committed` oracle entry at that index.
    fn check_snapshot_boundary_consistency(&mut self, cluster: &Cluster) -> Result<(), String> {
        for (id, node) in &cluster.nodes {
            let snap_pos = node.log().snapshot_position();
            if snap_pos == LogPosition::ZERO {
                continue;
            }
            if let Some(previous_term) = self
                .snapshot_boundary_terms
                .insert(snap_pos.index, snap_pos.term)
            {
                self.boundary_cross_check_fired = true;
                if previous_term != snap_pos.term {
                    return Err(format!(
                        "snapshot boundary term diverged at {:?}: node {} reports \
                         {:?}, previously observed {:?}",
                        snap_pos.index,
                        u64::from(*id),
                        snap_pos.term,
                        previous_term,
                    ));
                }
            }
            for (other_id, other_node) in &cluster.nodes {
                if other_id == id {
                    continue;
                }
                // Uncommitted entries on `other_node` may legitimately have
                // a different `term` at the same index while a stale log is
                // being repaired (log matching only ties `(term, index)`
                // pairs, not `index` alone). Restrict the peer check to
                // indices at or below `other_node`'s `commit_index`, which
                // is the range where terms must agree.
                if snap_pos.index > other_node.commit_index() {
                    continue;
                }
                let Some(other_term) = other_node.log().entries().get_term(snap_pos.index) else {
                    continue;
                };
                self.boundary_cross_check_fired = true;
                if other_term != snap_pos.term {
                    return Err(format!(
                        "snapshot boundary term mismatch across nodes at {:?}: \
                         node {} snapshot term {:?}, node {} log term {:?}",
                        snap_pos.index,
                        u64::from(*id),
                        snap_pos.term,
                        u64::from(*other_id),
                        other_term,
                    ));
                }
            }
            if let Some((oracle_pos, _, _)) = self.committed.get(&snap_pos.index) {
                self.boundary_cross_check_fired = true;
                if oracle_pos.term != snap_pos.term {
                    return Err(format!(
                        "snapshot boundary term mismatch with committed oracle at \
                         {:?}: node {} snapshot term {:?}, oracle term {:?}",
                        snap_pos.index,
                        u64::from(*id),
                        snap_pos.term,
                        oracle_pos.term,
                    ));
                }
            }
        }
        Ok(())
    }
}

fn sample_steps(ctx: &mut noprop::TestCaseContext) -> usize {
    noprop::sample_with_boundaries(
        ctx,
        &[MIN_STEPS, MAX_STEPS],
        noprop::Ratio::one_nth(5),
        |ctx| noprop::sample_usize_in(ctx, MIN_STEPS + 1..MAX_STEPS),
    )
}

/// The Raft election-safety, log-matching, leader-append-only,
/// state-machine-safety, and leader-completeness properties, plus the
/// within-tenure monotonicity of `Node::follower_match_index`, hold
/// after every state-dependent cluster command.
#[test]
fn cluster_invariants_hold() -> noprop::TestResult {
    // Bumped from 128 to 256: `Cmd::PersistStorage` added a new
    // storage-scoped path that must appear per case, so more cases are
    // needed to reliably exercise every coverage counter (in particular
    // `cases_with_snapshot_dropped` / `cases_with_remote_snapshot_installed`,
    // both of which depend on `snapshot_queued` being non-empty first).
    let config = run_config(256)?;
    let cases_with_command_commit = Cell::new(0usize);
    let cases_with_leader_history = Cell::new(0usize);
    let cases_with_crash = Cell::new(0usize);
    let cases_with_restart = Cell::new(0usize);
    let cases_with_duplicate = Cell::new(0usize);
    let cases_with_multiple_leader_terms = Cell::new(0usize);
    let cases_with_local_snapshot = Cell::new(0usize);
    let cases_with_remote_snapshot_installed = Cell::new(0usize);
    let cases_with_snapshot_dropped = Cell::new(0usize);
    let cases_with_boundary_cross_check = Cell::new(0usize);
    let cases_with_storage_persisted = Cell::new(0usize);
    let cases_with_crash_before_persist = Cell::new(0usize);
    let cases_with_restart_after_snapshot_fold = Cell::new(0usize);
    let mut runner = noprop::Runner::new(config.seed);

    runner.run(config.cases, |ctx| {
        let node_count = noprop::sample_usize_in(ctx, 3..=5) as u64;
        let node_ids: Vec<NodeId> = (0..node_count).map(NodeId::new).collect();
        let mut cluster = Cluster::bootstrap(&node_ids)?;
        let mut invariants = InvariantState::default();
        let mut history = Vec::new();
        let mut crashed = false;
        let mut restarted = false;
        let mut duplicated = false;
        let mut local_snapshot = false;
        let mut remote_snapshot_installed = false;
        let mut snapshot_dropped = false;
        let mut storage_persisted = false;
        let mut crash_before_persist = false;
        let mut restart_after_snapshot_fold = false;
        invariants.check(&cluster)?;

        for step in 0..sample_steps(ctx) {
            let command = cluster.sample_command(ctx);
            crashed |= matches!(command, Cmd::Crash(_));
            restarted |= matches!(command, Cmd::Restart(_));
            duplicated |= matches!(command, Cmd::DuplicateNext(_));
            snapshot_dropped |= matches!(command, Cmd::DropSnapshot(_));
            storage_persisted |= matches!(command, Cmd::PersistStorage(_));
            // Crash with pending storage on the same node = crash window
            // exercise. Detected before `apply` because `apply(Crash)`
            // clears the pending entry.
            if let Cmd::Crash(id) = command
                && cluster.pending_storage.contains_key(&id)
            {
                crash_before_persist = true;
            }
            // Restart from a durable state whose log already carries a
            // snapshot boundary confirms the
            // `fold_snapshot_install_into_durable` path was exercised:
            // a prior `TakeSnapshot` / `DeliverSnapshot` /
            // `DuplicateSnapshot` folded the install into
            // `durable_states[id]`, that state survived a crash, and
            // this restart is now restoring from it. Detected before
            // `apply(Restart)` consumes the `CrashedNode` entry.
            if let Cmd::Restart(id) = command
                && let Some(crashed_node) = cluster.crashed.get(&id)
                && crashed_node.durable.log.snapshot_position().index != LogIndex::ZERO
            {
                restart_after_snapshot_fold = true;
            }
            let remote_install_ok = cluster.apply(command).map_err(|error| {
                format!("command failed at step {step}: {error}; history={history:?}")
            })?;
            local_snapshot |= matches!(command, Cmd::TakeSnapshot(_));
            if remote_install_ok {
                remote_snapshot_installed = true;
            }
            history.push(command.to_string());
            invariants.check(&cluster).map_err(|error| {
                format!("invariant failed at step {step}: {error}; history={history:?}")
            })?;
        }

        if invariants
            .committed
            .values()
            .any(|(_, entry, _)| matches!(entry, LogEntry::Command))
        {
            cases_with_command_commit.set(cases_with_command_commit.get() + 1);
        }
        if invariants.leader_with_committed_history_seen {
            cases_with_leader_history.set(cases_with_leader_history.get() + 1);
        }
        if crashed {
            cases_with_crash.set(cases_with_crash.get() + 1);
        }
        if restarted {
            cases_with_restart.set(cases_with_restart.get() + 1);
        }
        if duplicated {
            cases_with_duplicate.set(cases_with_duplicate.get() + 1);
        }
        if invariants.leaders_by_term.len() > 1 {
            cases_with_multiple_leader_terms.set(cases_with_multiple_leader_terms.get() + 1);
        }
        if local_snapshot {
            cases_with_local_snapshot.set(cases_with_local_snapshot.get() + 1);
        }
        if remote_snapshot_installed {
            cases_with_remote_snapshot_installed
                .set(cases_with_remote_snapshot_installed.get() + 1);
        }
        if snapshot_dropped {
            cases_with_snapshot_dropped.set(cases_with_snapshot_dropped.get() + 1);
        }
        if invariants.boundary_cross_check_fired {
            cases_with_boundary_cross_check.set(cases_with_boundary_cross_check.get() + 1);
        }
        if storage_persisted {
            cases_with_storage_persisted.set(cases_with_storage_persisted.get() + 1);
        }
        if crash_before_persist {
            cases_with_crash_before_persist.set(cases_with_crash_before_persist.get() + 1);
        }
        if restart_after_snapshot_fold {
            cases_with_restart_after_snapshot_fold
                .set(cases_with_restart_after_snapshot_fold.get() + 1);
        }
        Ok(())
    })?;

    let coverage = [
        (cases_with_command_commit.get(), "a committed command"),
        (
            cases_with_leader_history.get(),
            "a leader with committed history",
        ),
        (cases_with_crash.get(), "an effective crash"),
        (cases_with_restart.get(), "an effective restart"),
        (
            cases_with_duplicate.get(),
            "an effective duplicated message delivery",
        ),
        (
            cases_with_multiple_leader_terms.get(),
            "leaders in multiple terms",
        ),
        (cases_with_local_snapshot.get(), "a local snapshot"),
        (
            cases_with_remote_snapshot_installed.get(),
            "a remote snapshot install that succeeded",
        ),
        (cases_with_snapshot_dropped.get(), "a dropped snapshot"),
        (
            cases_with_boundary_cross_check.get(),
            "a snapshot boundary that was cross-checked against another observation",
        ),
        (
            cases_with_storage_persisted.get(),
            "a storage transaction persisted via PersistStorage",
        ),
        (
            cases_with_crash_before_persist.get(),
            "a crash while a storage transaction was still pending",
        ),
        (
            cases_with_restart_after_snapshot_fold.get(),
            "a restart from a durable state that carries a snapshot boundary",
        ),
    ];
    for (count, label) in coverage {
        assert!(count > 0, "no case exercised {label}\n{runner}");
    }
    Ok(())
}
