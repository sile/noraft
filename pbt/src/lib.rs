//! Shared runner, generators, and tick-driven cluster harness for PBTs.

use std::collections::BTreeMap;
use std::io::{Error, ErrorKind};

pub const SEED_ENV: &str = "NORAFT_PBT_SEED";
pub const CASES_ENV: &str = "NORAFT_PBT_CASES";

#[derive(Debug, Clone, Copy)]
pub struct RunConfig {
    pub seed: u64,
    pub cases: usize,
}

/// Loads a fresh-or-reproducible seed and a strictly positive case budget.
///
/// [`noprop::seed_from_env_or_time`] reads `NORAFT_PBT_SEED` only when
/// explicitly set for failure reproduction. Otherwise each invocation
/// derives a new seed from the current time, including on CI.
///
/// An unset case-budget variable selects `default_cases`. A malformed
/// value or zero is an error so a misspelled override cannot silently
/// fall back or turn a property into a zero-case success.
pub fn run_config(default_cases: usize) -> noprop::TestResult<RunConfig> {
    assert!(
        default_cases > 0,
        "the default case budget must be positive"
    );

    let seed = noprop::seed_from_env_or_time(SEED_ENV)?;
    let cases = match std::env::var(CASES_ENV) {
        Ok(value) => value.parse::<usize>().map_err(|error| {
            Error::new(
                ErrorKind::InvalidInput,
                format!("invalid {CASES_ENV} value {value:?}: {error}"),
            )
        })?,
        Err(std::env::VarError::NotPresent) => default_cases,
        Err(error) => return Err(error.into()),
    };
    if cases == 0 {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!("{CASES_ENV} must be greater than zero"),
        )
        .into());
    }

    Ok(RunConfig { seed, cases })
}

/// Runs a property with a time-derived seed unless reproduction is requested.
pub fn run<F>(default_cases: usize, property: F) -> noprop::TestResult
where
    F: Fn(&mut noprop::TestCaseContext) -> noprop::TestResult,
{
    let config = run_config(default_cases)?;
    noprop::Runner::new(config.seed).run(config.cases, property)?;
    Ok(())
}

/// Samples a bounded length while giving empty, singleton, and maximum
/// lengths explicit probability.
pub fn sample_len(ctx: &mut noprop::TestCaseContext, max: usize) -> usize {
    assert!(max >= 3, "sample_len requires max >= 3");
    noprop::sample_with_boundaries(ctx, &[0, 1, max], noprop::Ratio::one_nth(5), |ctx| {
        noprop::sample_usize_in(ctx, 2..max)
    })
}

/// Samples every representable `u64` except `u64::MAX`, with extra
/// weight on values relevant to `next()` boundary behavior.
pub fn sample_u64_before_max(ctx: &mut noprop::TestCaseContext) -> u64 {
    noprop::sample_with_boundaries(
        ctx,
        &[0, 1, u64::MAX - 1],
        noprop::Ratio::one_nth(5),
        |ctx| {
            noprop::sample_with_rejection(ctx, 8, |ctx| {
                let value = noprop::sample_u64(ctx);
                (value < u64::MAX).then_some(value)
            })
        },
    )
}

/// Samples an arbitrary public `noraft::ClusterConfig`, including overlaps
/// between any of the three node sets.
pub fn sample_config(ctx: &mut noprop::TestCaseContext) -> noraft::ClusterConfig {
    let mut config = noraft::ClusterConfig::new();
    for value in 0..6 {
        let id = noraft::NodeId::new(value);
        let membership = noprop::sample_usize_in(ctx, 0..8);
        if membership & 1 != 0 {
            config.voters.insert(id);
        }
        if membership & 2 != 0 {
            config.new_voters.insert(id);
        }
        if membership & 4 != 0 {
            config.non_voters.insert(id);
        }
    }
    config
}

/// Samples a non-joint configuration whose voters and non-voters are
/// disjoint by construction.
pub fn sample_normal_config(ctx: &mut noprop::TestCaseContext) -> noraft::ClusterConfig {
    let mut config = noraft::ClusterConfig::new();
    for value in 0..6 {
        let id = noraft::NodeId::new(value);
        match noprop::sample_usize_in(ctx, 0..3) {
            0 => {
                config.voters.insert(id);
            }
            1 => {
                config.non_voters.insert(id);
            }
            _ => {}
        }
    }
    config
}

pub fn sample_log_entry(ctx: &mut noprop::TestCaseContext) -> noraft::LogEntry {
    match noprop::sample_weighted_index(ctx, &[3, 1, 1]) {
        0 => noraft::LogEntry::Command,
        1 => noraft::LogEntry::Term(noraft::Term::new(noprop::sample_u64(ctx))),
        _ => noraft::LogEntry::ClusterConfig(sample_config(ctx)),
    }
}

#[derive(Debug)]
pub struct TestCluster {
    pub nodes: Vec<TestNode>,
    pub clock: Clock,
    pub default_link_options: TestLinkOptions,
    seqno: u64,
    leaders_by_term: BTreeMap<noraft::Term, noraft::NodeId>,
}

impl TestCluster {
    pub fn new(node_ids: &[noraft::NodeId]) -> Self {
        Self {
            nodes: node_ids.iter().map(|&id| TestNode::new(id)).collect(),
            clock: Clock::new(),
            default_link_options: TestLinkOptions::default(),
            seqno: 0,
            leaders_by_term: BTreeMap::new(),
        }
    }

    pub fn leader_node(&self) -> Option<&noraft::Node> {
        self.nodes
            .iter()
            .find(|node| node.running && node.inner.role().is_leader())
            .map(|node| &node.inner)
    }

    pub fn leader_node_mut(&mut self) -> Option<&mut noraft::Node> {
        self.nodes
            .iter_mut()
            .find(|node| node.running && node.inner.role().is_leader())
            .map(|node| &mut node.inner)
    }

    pub fn random_node_mut(&mut self, ctx: &mut noprop::TestCaseContext) -> &mut noraft::Node {
        let index = noprop::sample_usize_in(ctx, 0..self.nodes.len());
        &mut self.nodes[index].inner
    }

    pub fn run_while_leader_absent(
        &mut self,
        ctx: &mut noprop::TestCaseContext,
        deadline: Clock,
    ) -> bool {
        self.run_until(ctx, deadline, |cluster| cluster.leader_node().is_some())
    }

    pub fn run(&mut self, ctx: &mut noprop::TestCaseContext, deadline: Clock) {
        self.run_until(ctx, deadline, |_| false);
    }

    pub fn run_until<F>(
        &mut self,
        ctx: &mut noprop::TestCaseContext,
        deadline: Clock,
        condition: F,
    ) -> bool
    where
        F: Fn(&TestCluster) -> bool,
    {
        while self.clock < deadline {
            if condition(self) {
                return true;
            }
            self.run_tick(ctx);
        }
        condition(self)
    }

    pub fn run_tick(&mut self, ctx: &mut noprop::TestCaseContext) {
        self.clock.tick();
        let mut messages = Vec::new();
        let mut snapshots = Vec::new();

        for node in &mut self.nodes {
            node.run_tick(ctx, self.clock);

            let source = node.inner.id();

            // Release any outbound that was held for a transaction which
            // has now committed. `run_tick` clears `pending_durable`
            // when storage completes, so a `None` here with a non-empty
            // `pending_outbound` means the batch just became durable.
            if node.pending_durable.is_none() && !node.pending_outbound.is_empty() {
                for held in node.pending_outbound.drain(..) {
                    match held {
                        PendingOutbound::Broadcast(message) => {
                            for destination in node.inner.peers() {
                                messages.push((source, destination, message.clone()));
                            }
                        }
                        PendingOutbound::Send(destination, message) => {
                            messages.push((source, destination, message));
                        }
                        PendingOutbound::InstallSnapshot(destination, position, config) => {
                            snapshots.push((source, destination, position, config));
                        }
                    }
                }
            }

            let mut actions = std::mem::take(node.inner.actions_mut());
            let holding = node.pending_durable.is_some();
            if let Some(message) = actions.broadcast_message.take() {
                if holding {
                    node.pending_outbound
                        .push(PendingOutbound::Broadcast(message));
                } else {
                    for destination in node.inner.peers() {
                        messages.push((source, destination, message.clone()));
                    }
                }
            }
            for (destination, message) in actions.send_messages {
                if holding {
                    node.pending_outbound
                        .push(PendingOutbound::Send(destination, message));
                } else {
                    messages.push((source, destination, message));
                }
            }
            for destination in actions.install_snapshots {
                let position = node.inner.log().snapshot_position();
                let config = node.inner.log().snapshot_config().clone();
                if holding {
                    node.pending_outbound.push(PendingOutbound::InstallSnapshot(
                        destination,
                        position,
                        config,
                    ));
                } else {
                    snapshots.push((source, destination, position, config));
                }
            }
        }

        for (source, destination, message) in messages {
            self.send_message(ctx, source, destination, message);
        }
        for (source, destination, position, config) in snapshots {
            self.send_snapshot(ctx, source, destination, position, config);
        }
        self.check_election_safety();
    }

    fn send_message(
        &mut self,
        ctx: &mut noprop::TestCaseContext,
        _source: noraft::NodeId,
        destination: noraft::NodeId,
        message: noraft::Message,
    ) {
        let options = &self.default_link_options;
        if noprop::sample_ratio(ctx, options.drop_rate) {
            return;
        }

        let latency = options.latency_ticks.sample(ctx) * message_size(&message);
        for node in &mut self.nodes {
            if node.inner.id() == destination {
                node.incoming_messages
                    .insert((self.clock.after(latency), self.seqno), message);
                self.seqno += 1;
                return;
            }
        }
    }

    fn send_snapshot(
        &mut self,
        ctx: &mut noprop::TestCaseContext,
        _source: noraft::NodeId,
        destination: noraft::NodeId,
        position: noraft::LogPosition,
        config: noraft::ClusterConfig,
    ) {
        if noprop::sample_ratio(ctx, self.default_link_options.drop_rate) {
            return;
        }

        for node in &mut self.nodes {
            if node.inner.id() == destination {
                if node.snapshot_finish_time.is_some() {
                    return;
                }
                node.snapshot_finish_time = Some((
                    self.clock
                        .after(node.options.install_snapshot_ticks.sample(ctx)),
                    position,
                    config,
                ));
                return;
            }
        }
    }

    fn check_election_safety(&mut self) {
        for node in &self.nodes {
            if !node.inner.role().is_leader() {
                continue;
            }
            let id = node.inner.id();
            let term = node.inner.current_term();
            if let Some(previous) = self.leaders_by_term.insert(term, id) {
                assert_eq!(
                    previous, id,
                    "election safety violated: two leaders observed in term {term:?}"
                );
            }
        }
    }
}

/// Runs the cluster until `position` reaches a terminal commit status
/// (committed / rejected / unknown), or the round budget is exhausted.
pub fn wait_until_terminal(
    cluster: &mut TestCluster,
    ctx: &mut noprop::TestCaseContext,
    position: noraft::LogPosition,
    max_rounds: usize,
) -> Option<noraft::CommitStatus> {
    for _ in 0..max_rounds {
        let found = cluster.run_while_leader_absent(ctx, cluster.clock.after(100_000));
        if !found {
            return None;
        }
        let leader = cluster
            .leader_node()
            .expect("a leader was found immediately above");
        let status = leader.get_commit_status(position);
        if !status.is_in_progress() {
            return Some(status);
        }
        cluster.run(ctx, cluster.clock.after(10));
    }
    None
}

/// Requires all positions to reach an allowed terminal status and
/// returns the number that committed.
pub fn assert_all_terminal(
    cluster: &mut TestCluster,
    ctx: &mut noprop::TestCaseContext,
    positions: &[noraft::LogPosition],
    allow_rejected: bool,
    allow_unknown: bool,
) -> Result<usize, String> {
    let mut committed = 0;
    for (i, position) in positions.iter().enumerate() {
        let status = wait_until_terminal(cluster, ctx, *position, 1000)
            .ok_or_else(|| format!("proposal {i} did not reach a terminal status"))?;
        if status.is_committed() {
            committed += 1;
        } else if (!status.is_rejected() || !allow_rejected)
            && (!status.is_unknown() || !allow_unknown)
        {
            return Err(format!(
                "proposal {i} ended with unexpected status {status:?}"
            ));
        }
    }
    Ok(committed)
}

fn message_size(message: &noraft::Message) -> usize {
    match message {
        noraft::Message::AppendEntriesCall { entries, .. } => entries.len(),
        noraft::Message::AppendEntriesReply { .. }
        | noraft::Message::RequestVoteCall { .. }
        | noraft::Message::RequestVoteReply { .. } => 1,
    }
}

#[derive(Debug, Clone)]
pub struct TestLinkOptions {
    pub latency_ticks: MinMax,
    pub drop_rate: noprop::Ratio,
}

impl Default for TestLinkOptions {
    fn default() -> Self {
        Self {
            latency_ticks: MinMax::new(5, 20),
            drop_rate: noprop::Ratio::new(1, 100),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TestNodeOptions {
    pub election_timeout_ticks: MinMax,
    pub storage_latency_ticks: MinMax,
    pub install_snapshot_ticks: MinMax,
    pub running_ticks: MinMax,
    pub stopping_ticks: MinMax,
}

impl Default for TestNodeOptions {
    fn default() -> Self {
        Self {
            election_timeout_ticks: MinMax::new(100, 1000),
            storage_latency_ticks: MinMax::new(1, 10),
            install_snapshot_ticks: MinMax::new(1000, 10_000),
            running_ticks: MinMax::constant(usize::MAX),
            stopping_ticks: MinMax::constant(usize::MAX),
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct MinMax {
    pub min: usize,
    pub max: usize,
}

impl MinMax {
    pub fn new(min: usize, max: usize) -> Self {
        assert!(min <= max, "MinMax requires min <= max");
        Self { min, max }
    }

    pub fn constant(value: usize) -> Self {
        Self::new(value, value)
    }

    pub fn sample(self, ctx: &mut noprop::TestCaseContext) -> usize {
        noprop::sample_usize_in(ctx, self.min..=self.max)
    }
}

/// Test-only durable state snapshot: what `noraft::Node::restart` needs when a
/// crashed node comes back. Mirrors the persistent-state contract in
/// `src/node.rs::Node::restart`.
#[derive(Debug, Clone)]
struct DurableSnapshot {
    current_term: noraft::Term,
    voted_for: Option<noraft::NodeId>,
    log: noraft::Log,
}

impl DurableSnapshot {
    fn from_node(node: &noraft::Node) -> Self {
        Self {
            current_term: node.current_term(),
            voted_for: node.voted_for(),
            log: node.log().clone(),
        }
    }
}

/// Outbound produced by a `noraft::Node` while a storage transaction was in
/// flight. Held per node until the transaction commits, at which point
/// `TestCluster::run_tick` moves it onto the delivery queues.
///
/// `InstallSnapshot` carries the `(noraft::LogPosition, noraft::ClusterConfig)` that
/// were live on the source node at hold time. Reading them at release
/// time instead would deliver a snapshot boundary the source never
/// asked the user to send: a successful `noraft::Node::handle_snapshot_installed`
/// on the same source (folded into `durable` above) can advance
/// `snapshot_position` while the transfer is still held.
#[derive(Debug)]
enum PendingOutbound {
    Broadcast(noraft::Message),
    Send(noraft::NodeId, noraft::Message),
    InstallSnapshot(noraft::NodeId, noraft::LogPosition, noraft::ClusterConfig),
}

#[derive(Debug)]
pub struct TestNode {
    pub inner: noraft::Node,
    pub options: TestNodeOptions,
    pub voter: bool,
    running: bool,
    timeout_expire_time: Option<Clock>,
    storage_finish_time: Option<Clock>,
    snapshot_finish_time: Option<(Clock, noraft::LogPosition, noraft::ClusterConfig)>,
    incoming_messages: BTreeMap<(Clock, u64), noraft::Message>,
    stop_time: Option<Clock>,
    start_time: Option<Clock>,
    restarts: u64,
    // Last state that has been fully persisted. Restart restores the
    // node from this snapshot, not from live `noraft::Node` state.
    durable: DurableSnapshot,
    // In-flight storage transaction target. Set when a storage action
    // (`SaveCurrentTerm` / `SaveVotedFor` / `AppendLogEntries`) is added
    // and cleared into `durable` when `storage_finish_time` expires.
    // Discarded on crash so an unfinished write is not restored.
    pending_durable: Option<DurableSnapshot>,
    // Outbound produced while `pending_durable` was `Some`. Held here
    // until the transaction commits, then handed to `TestCluster` for
    // delivery. Cleared on crash.
    pending_outbound: Vec<PendingOutbound>,
}

impl TestNode {
    pub fn new(id: noraft::NodeId) -> Self {
        let inner = noraft::Node::start(id);
        let durable = DurableSnapshot::from_node(&inner);
        Self {
            inner,
            options: TestNodeOptions::default(),
            voter: true,
            running: true,
            timeout_expire_time: None,
            storage_finish_time: None,
            snapshot_finish_time: None,
            incoming_messages: BTreeMap::new(),
            stop_time: None,
            start_time: None,
            restarts: 0,
            durable,
            pending_durable: None,
            pending_outbound: Vec::new(),
        }
    }

    /// Monotonically increasing count of the times this node has been
    /// restarted through `noraft::Node::restart` (excluding the initial `noraft::Node::start`
    /// in `TestNode::new`).
    pub fn restarts(&self) -> u64 {
        self.restarts
    }

    fn run_tick(&mut self, ctx: &mut noprop::TestCaseContext, now: Clock) {
        if !self.voter {
            assert!(self.inner.role().is_follower());
        }

        if !self.running {
            if self.start_time.take_if(|time| *time <= now).is_some() {
                self.running = true;
                while let Some(entry) = self.incoming_messages.first_entry() {
                    if entry.key().0 < now {
                        entry.remove();
                    } else {
                        break;
                    }
                }
                // Restore from the last committed durable snapshot, not
                // from live `noraft::Node` state. This makes the harness respect
                // the persistence contract: an unfinished write that was
                // still in `pending_durable` at crash time is not
                // resurrected.
                self.inner = noraft::Node::restart(
                    self.inner.id(),
                    self.durable.current_term,
                    self.durable.voted_for,
                    self.durable.log.clone(),
                );
                self.restarts = self.restarts.saturating_add(1);
            } else {
                return;
            }
        }

        if self.stop_time.is_none() {
            self.stop_time = Some(now.after(self.options.running_ticks.sample(ctx)));
        }
        if self.stop_time.take_if(|time| *time <= now).is_some() {
            self.running = false;
            self.timeout_expire_time = None;
            self.storage_finish_time = None;
            self.snapshot_finish_time = None;
            // In-flight storage transaction is lost on crash. The held
            // outbound is dropped for the same reason: those messages
            // depended on the transaction that never became durable.
            self.pending_durable = None;
            self.pending_outbound.clear();
            self.start_time = Some(now.after(self.options.stopping_ticks.sample(ctx)));
            return;
        }

        if self
            .storage_finish_time
            .take_if(|time| *time <= now)
            .is_some()
            && let Some(committed) = self.pending_durable.take()
        {
            self.durable = committed;
        }
        if self.storage_finish_time.is_some() {
            return;
        }

        if self
            .timeout_expire_time
            .take_if(|time| *time <= now)
            .is_some()
        {
            self.inner.handle_election_timeout();
        }
        if let Some((_, position, config)) = self
            .snapshot_finish_time
            .take_if(|(time, _, _)| *time <= now)
        {
            // `noraft::Node::handle_snapshot_installed` rejects snapshots whose
            // term exceeds `current_term`. `noraft::Action::InstallSnapshot`
            // only fires from `handle_append_entries_reply`, which the
            // follower cannot reach before catching up to the leader's
            // term via a prior `AppendEntriesCall`. So by the time the
            // snapshot arrives here, `current_term >= position.term` is
            // an invariant of the harness. Pin it with an assertion so
            // future scenarios that would violate it fail loudly.
            debug_assert!(
                position.term <= self.inner.current_term(),
                "snapshot term {:?} exceeds receiver current_term {:?}",
                position.term,
                self.inner.current_term()
            );
            if self.inner.handle_snapshot_installed(position, config) {
                // A successful snapshot install atomically persists the
                // new log / term boundary from the user's perspective
                // (it acknowledges an already-persisted snapshot). Fold
                // it into `durable` so a subsequent crash keeps the
                // installed state.
                self.durable = DurableSnapshot::from_node(&self.inner);
            }
        }
        while let Some(entry) = self.incoming_messages.first_entry() {
            if entry.key().0 <= now {
                let message = entry.remove();
                self.inner
                    .handle_message(&message)
                    .expect("harness-produced messages must be valid");
            } else {
                break;
            }
        }

        if std::mem::take(&mut self.inner.actions_mut().set_election_timeout) {
            self.reset_election_timeout(ctx, now);
        }
        let mut generated_storage = false;
        if std::mem::take(&mut self.inner.actions_mut().save_current_term) {
            self.extend_storage_finish_time(ctx, now, 1);
            generated_storage = true;
        }
        if std::mem::take(&mut self.inner.actions_mut().save_voted_for) {
            self.extend_storage_finish_time(ctx, now, 1);
            generated_storage = true;
        }
        if let Some(entries) = self.inner.actions_mut().append_log_entries.take() {
            self.extend_storage_finish_time(ctx, now, entries.len());
            generated_storage = true;
        }
        if generated_storage {
            // Capture the current live `noraft::Node` state as the target of the
            // in-flight transaction. If more storage actions are added
            // in later ticks before `storage_finish_time` expires, the
            // target is refreshed each time so it always reflects the
            // latest queued write.
            self.pending_durable = Some(DurableSnapshot::from_node(&self.inner));
        }
    }

    fn reset_election_timeout(&mut self, ctx: &mut noprop::TestCaseContext, now: Clock) {
        let timeout = match self.inner.role() {
            noraft::Role::Leader => self.options.election_timeout_ticks.min,
            noraft::Role::Candidate => self.options.election_timeout_ticks.sample(ctx),
            noraft::Role::Follower => self.options.election_timeout_ticks.max,
        };
        self.timeout_expire_time = Some(now.after(timeout));
    }

    fn extend_storage_finish_time(
        &mut self,
        ctx: &mut noprop::TestCaseContext,
        now: Clock,
        count: usize,
    ) {
        let remaining_latency = self.storage_finish_time.map_or(0, |time| time.0 - now.0);
        let additional_latency = self.options.storage_latency_ticks.sample(ctx) * count;
        self.storage_finish_time = Some(now.after(remaining_latency + additional_latency));
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Clock(usize);

impl Clock {
    pub fn new() -> Self {
        Self(0)
    }

    pub fn tick(&mut self) {
        self.0 += 1;
    }

    pub fn after(self, ticks: usize) -> Self {
        Self(self.0.saturating_add(ticks))
    }
}
