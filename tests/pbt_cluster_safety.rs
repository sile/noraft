//! Stateful Raft properties driven by state-dependent cluster commands.
//!
//! Commands are only drawn when their preconditions hold, and the five
//! Raft safety properties are checked after every effective transition.
//! The oracle retains leaders and committed entries across the complete
//! case so sequential violations cannot disappear with current state.

pub mod pbt_harness;

use noraft::{Action, Log, LogEntry, LogIndex, LogPosition, Message, Node, NodeId, Role, Term};
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
}

struct CrashedNode {
    term: Term,
    voted_for: Option<NodeId>,
    log: Log,
}

struct Cluster {
    nodes: BTreeMap<NodeId, Node>,
    queues: BTreeMap<NodeId, VecDeque<Message>>,
    crashed: BTreeMap<NodeId, CrashedNode>,
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
            crashed: BTreeMap::new(),
        };
        cluster.drain_actions(bootstrap_id)?;
        Ok(cluster)
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
        for action in actions {
            match action {
                Action::SetElectionTimeout
                | Action::SaveCurrentTerm
                | Action::SaveVotedFor
                | Action::AppendLogEntries(_) => {
                    // This harness models persistence as synchronous.
                    // Node already contains the state represented by
                    // these actions when it is crashed.
                }
                Action::BroadcastMessage(message) => {
                    for peer in &peers {
                        if let Some(queue) = self.queues.get_mut(peer) {
                            queue.push_back(message.clone());
                        }
                    }
                }
                Action::SendMessage(to, message) => {
                    if let Some(queue) = self.queues.get_mut(&to) {
                        queue.push_back(message);
                    }
                }
                Action::InstallSnapshot(to) => {
                    return Err(format!(
                        "snapshot installation for node {} is outside this harness",
                        u64::from(to)
                    ));
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

    fn leader_ids(&self) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter_map(|(id, node)| (node.role() == Role::Leader).then_some(*id))
            .collect()
    }

    fn sample_command(&self, ctx: &mut noprop::TestCaseContext) -> Cmd {
        let running = self.running_ids();
        let crashed = self.crashed_ids();
        let queued = self.queued_ids();
        let leaders = self.leader_ids();
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

        let kind = kinds[noprop::sample_weighted_index(ctx, &weights)];
        match kind {
            CmdKind::TickElection => Cmd::TickElection(noprop::sample_choice(ctx, &running)),
            CmdKind::DeliverNext => Cmd::DeliverNext(noprop::sample_choice(ctx, &queued)),
            CmdKind::DuplicateNext => Cmd::DuplicateNext(noprop::sample_choice(ctx, &queued)),
            CmdKind::DropNext => Cmd::DropNext(noprop::sample_choice(ctx, &queued)),
            CmdKind::Crash => Cmd::Crash(noprop::sample_choice(ctx, &running)),
            CmdKind::Restart => Cmd::Restart(noprop::sample_choice(ctx, &crashed)),
            CmdKind::Propose => Cmd::Propose(noprop::sample_choice(ctx, &leaders)),
        }
    }

    fn apply(&mut self, command: Cmd) -> Result<(), String> {
        match command {
            Cmd::TickElection(id) => {
                self.nodes
                    .get_mut(&id)
                    .expect("commands only select running nodes")
                    .handle_election_timeout();
                self.drain_actions(id)
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
                self.drain_actions(id)
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
                self.drain_actions(id)
            }
            Cmd::DropNext(id) => {
                self.queues
                    .get_mut(&id)
                    .and_then(VecDeque::pop_front)
                    .expect("commands only select non-empty queues");
                Ok(())
            }
            Cmd::Crash(id) => {
                let node = self
                    .nodes
                    .remove(&id)
                    .expect("commands only select running nodes");
                self.queues.remove(&id);
                self.crashed.insert(
                    id,
                    CrashedNode {
                        term: node.current_term(),
                        voted_for: node.voted_for(),
                        log: node.log().clone(),
                    },
                );
                Ok(())
            }
            Cmd::Restart(id) => {
                let state = self
                    .crashed
                    .remove(&id)
                    .expect("commands only select crashed nodes");
                let node = Node::restart(id, state.term, state.voted_for, state.log);
                self.nodes.insert(id, node);
                self.queues.insert(id, VecDeque::new());
                self.drain_actions(id)
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
                self.drain_actions(id)
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
}

impl InvariantState {
    fn check(&mut self, cluster: &Cluster) -> Result<(), String> {
        self.check_election_safety(cluster)?;
        self.check_log_matching(cluster)?;
        self.check_leader_append_only(cluster)?;
        self.check_state_machine_safety(cluster)?;
        self.check_leader_completeness(cluster)?;
        self.check_follower_match_index_monotonic(cluster)?;
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

                for (shared_position, _) in &left_entries {
                    let same_position = right_entries
                        .iter()
                        .any(|(position, _)| position == shared_position);
                    if !same_position {
                        continue;
                    }
                    let left_prefix: Vec<_> = left_entries
                        .iter()
                        .take_while(|(position, _)| position.index <= shared_position.index)
                        .collect();
                    let right_prefix: Vec<_> = right_entries
                        .iter()
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
            if let Some(previous) = self.leader_logs.get(&key)
                && !current.starts_with(previous)
            {
                return Err(format!(
                    "leader {} changed its term {:?} log prefix: previous={previous:?}, \
                     current={current:?}",
                    u64::from(*id),
                    node.current_term()
                ));
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
    let config = run_config(128)?;
    let cases_with_command_commit = Cell::new(0usize);
    let cases_with_leader_history = Cell::new(0usize);
    let cases_with_crash = Cell::new(0usize);
    let cases_with_restart = Cell::new(0usize);
    let cases_with_duplicate = Cell::new(0usize);
    let cases_with_multiple_leader_terms = Cell::new(0usize);
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
        invariants.check(&cluster)?;

        for step in 0..sample_steps(ctx) {
            let command = cluster.sample_command(ctx);
            crashed |= matches!(command, Cmd::Crash(_));
            restarted |= matches!(command, Cmd::Restart(_));
            duplicated |= matches!(command, Cmd::DuplicateNext(_));
            cluster.apply(command).map_err(|error| {
                format!("command failed at step {step}: {error}; history={history:?}")
            })?;
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
    ];
    for (count, label) in coverage {
        assert!(count > 0, "no case exercised {label}\n{runner}");
    }
    Ok(())
}
