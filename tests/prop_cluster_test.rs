//! Stateful property-based tests for a multi-node noraft cluster,
//! driven by noprop (https://github.com/sile/noprop).
//!
//! A 3-5 node deterministic cluster is driven by an arbitrary event
//! sequence (election timeout, message deliver / drop, node crash /
//! restart, propose command) and every Raft-paper invariant is checked
//! after every step:
//!
//! - Election safety: at most one leader per term
//! - Log matching: the same index never carries two terms across the
//!   cluster's committed prefixes
//! - Leader append-only: a leader never truncates its own log
//! - State machine safety: any two nodes' committed prefixes agree on
//!   every index (proxy for "no two nodes apply different entries at
//!   the same index")
//! - Leader completeness: every leader has every entry that has ever
//!   been committed by any node
//!
//! Seed / case budget come from the `NORAFT_PBT_SEED` and
//! `NORAFT_PBT_CASES` environment variables; unset means
//! "clock-derived seed" and "100 cases". A failing seed can be
//! re-run with:
//!
//! ```text
//! NORAFT_PBT_SEED=<seed> cargo test --test prop_cluster_test
//! ```

use noraft::{Log, Message, Node, NodeGeneration, NodeId, Role, Term};
use std::cell::Cell;
use std::collections::{BTreeMap, VecDeque};
use std::fmt;

const MIN_STEPS: usize = 20;
const MAX_STEPS: usize = 200;

#[derive(Clone, Debug)]
enum Cmd {
    TickElection(NodeId),
    DeliverNext(NodeId),
    DropNext(NodeId),
    CrashNode(NodeId),
    RestartNode(NodeId),
    Propose,
}

impl fmt::Display for Cmd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Cmd::TickElection(id) => write!(f, "TickElection({})", u64::from(*id)),
            Cmd::DeliverNext(id) => write!(f, "DeliverNext({})", u64::from(*id)),
            Cmd::DropNext(id) => write!(f, "DropNext({})", u64::from(*id)),
            Cmd::CrashNode(id) => write!(f, "CrashNode({})", u64::from(*id)),
            Cmd::RestartNode(id) => write!(f, "RestartNode({})", u64::from(*id)),
            Cmd::Propose => write!(f, "Propose"),
        }
    }
}

/// Persistent state of a crashed node, kept so it can be restored on
/// restart.
struct CrashedNode {
    generation: NodeGeneration,
    term: Term,
    voted_for: Option<NodeId>,
    log: Log,
}

/// A deterministic 3-5 node cluster driven by explicit commands.
///
/// The harness applies each command by calling the [`Node`] methods
/// directly and delivering the produced [`Message`]s through bounded
/// per-node queues. Crash / restart keep the persistent state
/// (generation, term, voted-for, log) in `crashed` so that a restart
/// restores exactly what the node had persisted.
struct Cluster {
    nodes: BTreeMap<NodeId, Node>,
    queues: BTreeMap<NodeId, VecDeque<Message>>,
    crashed: BTreeMap<NodeId, CrashedNode>,
}

impl Cluster {
    fn bootstrap(node_ids: &[NodeId]) -> Self {
        let mut nodes = BTreeMap::new();
        for &id in node_ids {
            nodes.insert(id, Node::start(id));
        }
        let mut queues = BTreeMap::new();
        for &id in node_ids {
            queues.insert(id, VecDeque::new());
        }
        let bootstrap_id = *node_ids.first().expect("at least one node");
        nodes
            .get_mut(&bootstrap_id)
            .expect("bootstrap node exists")
            .create_cluster(node_ids);
        let mut cluster = Self {
            nodes,
            queues,
            crashed: BTreeMap::new(),
        };
        cluster.drain_all_actions();
        cluster
    }

    fn drain_all_actions(&mut self) {
        let ids: Vec<_> = self.nodes.keys().copied().collect();
        for id in ids {
            self.drain_actions(id);
        }
    }

    fn drain_actions(&mut self, id: NodeId) {
        let peers: Vec<_> = self.nodes.get(&id).expect("node exists").peers().collect();
        let pending: Vec<_> = self
            .nodes
            .get_mut(&id)
            .expect("node exists")
            .actions_mut()
            .by_ref()
            .collect();
        for action in pending {
            use noraft::Action::*;
            match action {
                SetElectionTimeout | SaveCurrentTerm | SaveVotedFor | AppendLogEntries(_) => {
                    // Persistence-like operations are no-ops in this
                    // harness: the crashed state is kept by the
                    // harness instead.
                }
                BroadcastMessage(msg) => {
                    for peer in &peers {
                        self.queues.entry(*peer).or_default().push_back(msg.clone());
                    }
                }
                SendMessage(to, msg) => {
                    self.queues.entry(to).or_default().push_back(msg);
                }
                InstallSnapshot(_) => {
                    // Out of scope for this harness.
                }
            }
        }
    }

    fn tick_election(&mut self, id: NodeId) {
        if self.nodes.contains_key(&id) {
            self.nodes
                .get_mut(&id)
                .expect("checked contains_key")
                .handle_election_timeout();
            self.drain_actions(id);
        }
    }

    fn deliver_next(&mut self, id: NodeId) {
        let Some(msg) = self.queues.get_mut(&id).and_then(|q| q.pop_front()) else {
            return;
        };
        if let Some(node) = self.nodes.get_mut(&id) {
            let _ = node.handle_message(&msg);
            self.drain_actions(id);
        }
    }

    fn drop_next(&mut self, id: NodeId) {
        if let Some(queue) = self.queues.get_mut(&id) {
            let _ = queue.pop_front();
        }
    }

    fn crash_node(&mut self, id: NodeId) {
        let Some(node) = self.nodes.remove(&id) else {
            return;
        };
        self.crashed.insert(
            id,
            CrashedNode {
                generation: NodeGeneration::new(node.generation().get() + 1),
                term: node.current_term(),
                voted_for: node.voted_for(),
                log: node.log().clone(),
            },
        );
        self.queues.remove(&id);
    }

    fn restart_node(&mut self, id: NodeId) {
        let Some(state) = self.crashed.remove(&id) else {
            return;
        };
        let node = Node::restart(id, state.generation, state.term, state.voted_for, state.log);
        self.nodes.insert(id, node);
        self.queues.insert(id, VecDeque::new());
        self.drain_actions(id);
    }

    fn propose_on_any_leader(&mut self) {
        let leader_id = self.nodes.iter().find_map(|(id, node)| {
            if node.role() == Role::Leader {
                Some(*id)
            } else {
                None
            }
        });
        if let Some(id) = leader_id {
            self.nodes
                .get_mut(&id)
                .expect("checked find_map")
                .propose_command();
            self.drain_actions(id);
        }
    }

    fn leaders_per_term(&self) -> BTreeMap<Term, Vec<NodeId>> {
        let mut per_term: BTreeMap<Term, Vec<NodeId>> = BTreeMap::new();
        for (id, node) in &self.nodes {
            if node.role() == Role::Leader {
                per_term.entry(node.current_term()).or_default().push(*id);
            }
        }
        per_term
    }

    /// Log matching (committed-only variant): the same log index must
    /// never carry two different terms across the cluster's committed
    /// prefixes.
    ///
    /// Raft's Log Matching Property is "same `(index, term)` implies
    /// same entry / same prefix", which permits divergent
    /// *uncommitted* terms at the same index during ongoing
    /// replication (a follower that has not yet caught up to a later
    /// term still keeps its older uncommitted entries locally). We
    /// therefore restrict the check to committed indices, where Raft
    /// additionally guarantees agreement (this is the state machine
    /// safety piece).
    fn check_log_matching(&self) -> Result<(), String> {
        let mut term_at: BTreeMap<u64, (NodeId, Term)> = BTreeMap::new();
        for (id, node) in &self.nodes {
            let commit_index = node.commit_index().get();
            for (position, _) in node.log().entries().iter_with_positions() {
                let index = position.index.get();
                if index == 0 || index > commit_index {
                    continue;
                }
                if let Some((other, term)) = term_at.get(&index) {
                    if *term != position.term {
                        return Err(format!(
                            "log matching violated at committed index {index}: node {} has \
                             term {:?}, node {} has term {term:?}",
                            u64::from(*id),
                            position.term,
                            u64::from(*other),
                        ));
                    }
                } else {
                    term_at.insert(index, (*id, position.term));
                }
            }
        }
        Ok(())
    }

    /// Leader append-only: the last log index of each leader must be
    /// non-decreasing across the steps where the node stays leader.
    fn check_leader_append_only(&self, last: &mut BTreeMap<NodeId, u64>) -> Result<(), String> {
        for (id, node) in &self.nodes {
            if node.role() == Role::Leader {
                let last_index = node.log().entries().last_position().index.get();
                if let Some(prev) = last.get(id)
                    && last_index < *prev
                {
                    return Err(format!(
                        "leader append-only violated: node {} truncated its log from index \
                         {prev} to {last_index}",
                        u64::from(*id),
                    ));
                }
                last.insert(*id, last_index);
            }
        }
        Ok(())
    }

    /// State machine safety (proxy): any two nodes whose committed
    /// prefixes cover the same index must have identical terms at that
    /// index. Extends the invariant across time by rolling observed
    /// commits into `history`, so a value that was once committed at
    /// index `i` cannot later be replaced by a different value at the
    /// same `i`.
    ///
    /// We check terms only: log matching guarantees that the same
    /// `(index, term)` pair implies the same entry, so equal terms
    /// across nodes at the same committed index implies equal entries.
    fn check_state_machine_safety(&self, history: &mut BTreeMap<u64, Term>) -> Result<(), String> {
        let mut current: BTreeMap<u64, (NodeId, Term)> = BTreeMap::new();
        for (id, node) in &self.nodes {
            let commit_index = node.commit_index().get();
            for (position, _) in node.log().entries().iter_with_positions() {
                let index = position.index.get();
                if index == 0 || index > commit_index {
                    continue;
                }
                match current.get(&index) {
                    Some((other, other_term)) if *other_term != position.term => {
                        return Err(format!(
                            "state machine safety violated at index {index}: node {} committed \
                             with term {:?}, node {} committed with term {:?}",
                            u64::from(*id),
                            position.term,
                            u64::from(*other),
                            other_term,
                        ));
                    }
                    None => {
                        current.insert(index, (*id, position.term));
                    }
                    _ => {}
                }
            }
        }
        for (index, (_, term)) in current {
            match history.get(&index) {
                Some(prev) if *prev != term => {
                    return Err(format!(
                        "state machine safety violated across time at index {index}: history \
                         says term {:?}, cluster now committed with term {:?}",
                        prev, term,
                    ));
                }
                None => {
                    history.insert(index, term);
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Leader completeness: every leader must have every entry that
    /// has ever been committed by any node. In Raft, a candidate can
    /// only win an election if its log is at least as up-to-date as a
    /// quorum's, which guarantees the current leader (whatever term)
    /// holds every past-committed entry.
    ///
    /// Checked by scanning each leader's log for the recorded
    /// `(index, term)` of every entry in `history`. A missing pair is
    /// a violation.
    fn check_leader_completeness(&self, history: &BTreeMap<u64, Term>) -> Result<(), String> {
        for (leader_id, node) in &self.nodes {
            if node.role() != Role::Leader {
                continue;
            }
            let leader_term = node.current_term();
            for (&index, &term) in history {
                let has = node
                    .log()
                    .entries()
                    .iter_with_positions()
                    .any(|(pos, _)| pos.index.get() == index && pos.term == term);
                if !has {
                    return Err(format!(
                        "leader completeness violated: leader {} (term {:?}) is missing \
                         committed entry at index {index} (committed with term {:?})",
                        u64::from(*leader_id),
                        leader_term,
                        term,
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Samples a command with weights tuned to the cluster's behavior.
///
/// Uniform 1/6 across all six commands starves the log-replication
/// path (drops + crashes disrupt it faster than deliveries can
/// converge), so commits almost never happen and state machine
/// safety / leader completeness never actually get exercised
/// (silent success). Weight DeliverNext / Propose much heavier and
/// keep drops / crashes / restarts as low-rate disruptors so log
/// replication has room to converge to commits.
fn sample_command(ctx: &mut noprop::TestCaseContext, node_ids: &[NodeId]) -> Cmd {
    match noprop::sample_weighted_index(ctx, &[3, 12, 1, 1, 1, 5]) {
        0 => Cmd::TickElection(noprop::sample_choice(ctx, node_ids)),
        1 => Cmd::DeliverNext(noprop::sample_choice(ctx, node_ids)),
        2 => Cmd::DropNext(noprop::sample_choice(ctx, node_ids)),
        3 => Cmd::CrashNode(noprop::sample_choice(ctx, node_ids)),
        4 => Cmd::RestartNode(noprop::sample_choice(ctx, node_ids)),
        _ => Cmd::Propose,
    }
}

/// The five Raft invariants must hold after every step of every case.
#[test]
fn cluster_invariants_hold() -> noprop::TestResult {
    let seed = noprop::seed_from_env_or_time("NORAFT_PBT_SEED")?;
    let cases = std::env::var("NORAFT_PBT_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_usize);

    // Coverage gates: without these, a run whose cases never commit
    // an entry and never see a leader would silently pass state
    // machine safety and leader completeness (both checks turn into
    // no-ops on an empty `committed_history` / no-leader state). The
    // gates turn "should reach" into "must reach" so a regression in
    // event sampling / cluster wiring cannot hide.
    let cases_with_commit: Cell<usize> = Cell::new(0);
    let cases_with_leader_and_history: Cell<usize> = Cell::new(0);
    let mut runner = noprop::Runner::new(seed);
    runner.run(cases, |ctx| {
        let node_count = noprop::sample_usize_in(ctx, 3..=5) as u64;
        let node_ids: Vec<NodeId> = (0..node_count).map(NodeId::new).collect();
        let mut cluster = Cluster::bootstrap(&node_ids);
        let steps = noprop::sample_usize_in(ctx, MIN_STEPS..=MAX_STEPS);
        let mut history: Vec<String> = Vec::with_capacity(steps);
        let mut leader_last: BTreeMap<NodeId, u64> = BTreeMap::new();
        let mut committed_history: BTreeMap<u64, Term> = BTreeMap::new();

        for step in 0..steps {
            let cmd = sample_command(ctx, &node_ids);
            match &cmd {
                Cmd::TickElection(id) => cluster.tick_election(*id),
                Cmd::DeliverNext(id) => cluster.deliver_next(*id),
                Cmd::DropNext(id) => cluster.drop_next(*id),
                Cmd::CrashNode(id) => cluster.crash_node(*id),
                Cmd::RestartNode(id) => cluster.restart_node(*id),
                Cmd::Propose => cluster.propose_on_any_leader(),
            }
            history.push(cmd.to_string());

            // Election safety.
            let leaders_per_term = cluster.leaders_per_term();
            for (term, leaders) in &leaders_per_term {
                if leaders.len() > 1 {
                    let leaders_list: Vec<u64> = leaders.iter().copied().map(u64::from).collect();
                    return Err(format!(
                        "election safety violated at step {step}: term {} has {} leaders {:?}; \
                         history=[{}]",
                        u64::from(*term),
                        leaders.len(),
                        leaders_list,
                        history.join(", "),
                    )
                    .into());
                }
            }

            // Log matching.
            cluster
                .check_log_matching()
                .map_err(|e| format!("at step {step}: {e}; history=[{}]", history.join(", ")))?;

            // Leader append-only.
            cluster
                .check_leader_append_only(&mut leader_last)
                .map_err(|e| format!("at step {step}: {e}; history=[{}]", history.join(", ")))?;

            // State machine safety (updates committed_history in
            // place).
            cluster
                .check_state_machine_safety(&mut committed_history)
                .map_err(|e| format!("at step {step}: {e}; history=[{}]", history.join(", ")))?;

            // Leader completeness (uses committed_history built
            // above).
            cluster
                .check_leader_completeness(&committed_history)
                .map_err(|e| format!("at step {step}: {e}; history=[{}]", history.join(", ")))?;
        }

        // Case-level coverage bookkeeping: track whether this case
        // actually exercised the two invariant paths that can pass
        // vacuously.
        if !committed_history.is_empty() {
            cases_with_commit.set(cases_with_commit.get() + 1);
            let has_leader = cluster.nodes.values().any(|n| n.role() == Role::Leader);
            if has_leader {
                cases_with_leader_and_history.set(cases_with_leader_and_history.get() + 1);
            }
        }
        Ok(())
    })?;

    // Run-after assertions: at least one case must have committed
    // something (state machine safety exercised) and at least one
    // case must have both a leader and a non-empty committed history
    // simultaneously (leader completeness exercised). A budget of 100
    // cases with 20-200 steps and a 3-5 node cluster typically hits
    // both dozens of times; a total of zero indicates the harness or
    // sampling is broken.
    assert!(
        cases_with_commit.get() > 0,
        "no case committed an entry — state machine safety was never exercised \
         (seed={seed:#018x}, stats={:?})",
        runner.stats(),
    );
    assert!(
        cases_with_leader_and_history.get() > 0,
        "no case ended with both a leader and a non-empty committed history — leader \
         completeness was never exercised (seed={seed:#018x}, stats={:?})",
        runner.stats(),
    );
    Ok(())
}
