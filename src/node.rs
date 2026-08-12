use crate::{
    CommitStatus, Log, Role, Term,
    action::{Action, Actions},
    config::ClusterConfig,
    log::{LogEntries, LogEntry, LogIndex, LogPosition},
    message::Message,
    quorum::Quorum,
};
use alloc::collections::{BTreeMap, BTreeSet};

/// Node identifier ([`u64`]).
///
/// Note that if you want to distinguish nodes by their names (not integers),
/// mapping node names to identifiers is out of the scope of this crate.
///
/// Besides, each [`Node`] in a cluster can have a different mapping of names to identifiers.
/// In this case, it is necessary to remap [`NodeId`]s in [`Message`]s before delivering them to other nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(u64);

impl NodeId {
    /// Makes a new [`NodeId`] instance.
    pub const fn new(id: u64) -> Self {
        NodeId(id)
    }

    /// Returns the value of this identifier.
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl From<u64> for NodeId {
    fn from(value: u64) -> Self {
        Self::new(value)
    }
}

impl From<NodeId> for u64 {
    fn from(value: NodeId) -> Self {
        value.get()
    }
}

impl core::ops::Add for NodeId {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.0 + rhs.0)
    }
}

impl core::ops::AddAssign for NodeId {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl core::ops::Sub for NodeId {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.0 - rhs.0)
    }
}

impl core::ops::SubAssign for NodeId {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

/// Error returned by fallible node operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Error {
    /// The local snapshot position conflicts with a leader log.
    ///
    /// This cannot happen in a valid Raft execution and suggests corrupted or
    /// inconsistent persistent state. The crate user should stop using the
    /// affected node and inspect or rebuild its persistent state.
    LocalSnapshotMismatch,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LocalSnapshotMismatch => {
                f.write_str("local snapshot position conflicts with leader log")
            }
        }
    }
}

impl core::error::Error for Error {}

/// Cumulative counters for internal Raft-protocol events observed by a [`Node`].
///
/// Each [`Node`] owns a [`NodeMetrics`] instance, obtainable via [`Node::metrics`].
/// All counters are [`u64`] and are updated with saturating addition, so they clamp
/// at [`u64::MAX`] rather than wrapping.
///
/// # Interpretation
///
/// An increase in any counter alone does not confirm a protocol violation, a
/// storage failure, or a bug. Message reordering, duplication, delayed delivery,
/// and leader changes are all part of normal operation in a distributed system,
/// and each of them can legitimately increase one or more counters. Treat
/// sustained or unusually rapid growth as a signal to start an investigation
/// rather than as a definitive diagnosis.
///
/// [`NodeMetrics::append_entries_replies_ignored_behind_match_index`] and
/// [`NodeMetrics::term_advances_from_messages`] deserve particular care in
/// interpretation; see their field documentation for details.
///
/// # Lifecycle
///
/// Counters accumulate over the lifetime of the [`Node`] instance. They are not
/// reset on role transitions between follower, candidate, and leader. They are
/// not carried across [`Node::start`] or [`Node::restart`], and they are not part
/// of the persistent Raft state.
///
/// Cloning a [`Node`] copies the current counter values.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NodeMetrics {
    /// Number of times [`Node::handle_message`] observed a message whose sender
    /// is this node itself.
    pub self_messages_ignored: u64,

    /// Number of times an incoming message carried a term greater than the
    /// current term, causing this node to step down to follower.
    ///
    /// This counter also increases during normal leader elections, so a single
    /// increment is not by itself abnormal. Rapid growth over a short window can
    /// indicate an election storm or a network partition.
    pub term_advances_from_messages: u64,

    /// Number of times a `RequestVoteCall` was rejected because its term was
    /// smaller than the current term.
    pub request_vote_calls_rejected_by_old_term: u64,

    /// Number of times a `RequestVoteCall` was rejected because the
    /// candidate's last log position was older than this node's local log.
    pub request_vote_calls_rejected_by_log: u64,

    /// Number of times a `RequestVoteCall` was rejected because this node had
    /// already voted for another node in the current term.
    pub request_vote_calls_rejected_by_existing_vote: u64,

    /// Number of times a `RequestVoteReply` was ignored because its term was
    /// smaller than the current term.
    pub request_vote_replies_ignored_from_old_terms: u64,

    /// Number of times a `RequestVoteReply` was ignored because this node was
    /// not a candidate (that is, it was a follower or a leader).
    pub request_vote_replies_ignored_while_not_candidate: u64,

    /// Number of times an `AppendEntriesCall` was rejected because its term was
    /// smaller than the current term.
    pub append_entries_calls_rejected_by_old_term: u64,

    /// Number of times this node, acting as leader, received an
    /// `AppendEntriesCall` from another leader in the same term. A correct Raft
    /// execution never produces this situation.
    pub same_term_append_entries_calls_received_by_leader: u64,

    /// Number of times this node, acting as follower, detected that an incoming
    /// `AppendEntriesCall` anchored at an index that exists in the local log but
    /// with a different term (log divergence detected on the follower side).
    pub log_divergences_detected_by_follower: u64,

    /// Number of times an `AppendEntriesReply` was ignored because its term was
    /// smaller than the current term.
    pub append_entries_replies_ignored_from_old_terms: u64,

    /// Number of times an `AppendEntriesReply` was ignored because this node
    /// was not a leader (that is, it was a follower or a candidate).
    pub append_entries_replies_ignored_while_not_leader: u64,

    /// Number of times an `AppendEntriesReply` was ignored because it came from
    /// a node that the current leader does not track as a follower.
    pub append_entries_replies_ignored_from_unknown_nodes: u64,

    /// Number of times a follower reported a last log index smaller than the
    /// index that the leader had already acknowledged for that follower during
    /// the same leader's tenure.
    ///
    /// Under normal operation this counter increases occasionally due to
    /// message reordering. Sustained or acute growth during a single leader's
    /// tenure, combined with a follower whose `match_index` does not advance,
    /// can indicate log reordering, incorrect persistence ordering, or a lost
    /// log tail on the follower. See the "Signs that log loss may have
    /// occurred" section on [`Node::restart`] for how to interpret sustained
    /// growth.
    pub append_entries_replies_ignored_behind_match_index: u64,

    /// Number of times a follower reported a last log index greater than the
    /// leader's own last log index.
    pub append_entries_replies_ahead_of_leader: u64,

    /// Number of times this node, acting as leader, detected that a follower's
    /// reported last log position had an index present in the leader's log but
    /// with a different term (log divergence detected on the leader side).
    pub log_divergences_detected_by_leader: u64,

    /// Number of times this node, being a voter, started a new election by
    /// transitioning to candidate. The counter is not incremented when a
    /// non-voter or a removed node attempts to start an election.
    pub elections_started: u64,

    /// Number of times this node transitioned to leader.
    pub leaderships_started: u64,
}

/// Raft node.
#[derive(Debug, Clone)]
pub struct Node {
    id: NodeId,
    voted_for: Option<NodeId>,
    current_term: Term,
    log: Log,
    commit_index: LogIndex,
    actions: Actions,
    role: RoleState,
    metrics: NodeMetrics,
}

impl Node {
    /// Starts a new node.
    ///
    /// To create a new cluster, please call [`Node::create_cluster()`] after starting the node.
    ///
    /// If the node has already been part of a cluster, please use [`Node::restart()`] instead.
    /// The "Persistence requirements" section on [`Node::restart`] documents
    /// the contract that any process running a [`Node`] must satisfy,
    /// including newly-started ones.
    ///
    /// # Examples
    ///
    /// ```
    /// // Starts three nodes.
    /// let mut node0 = noraft::Node::start(noraft::NodeId::new(0));
    /// let node1 = noraft::Node::start(noraft::NodeId::new(1));
    /// let node2 = noraft::Node::start(noraft::NodeId::new(2));
    ///
    /// for node in [&node0, &node1, &node2] {
    ///     assert!(node.role().is_follower());
    ///     assert_eq!(node.config().unique_nodes().count(), 0);
    ///     assert_eq!(node.log().last_position(), noraft::LogPosition::ZERO);
    ///     assert!(node.actions().is_empty());
    /// }
    ///
    /// // Creates a new cluster.
    /// node0.create_cluster(&[node0.id(), node1.id(), node2.id()]);
    ///
    /// assert!(node0.role().is_candidate());
    /// assert_eq!(node0.config().unique_nodes().count(), 3);
    /// assert_ne!(node0.log().last_position(), noraft::LogPosition::ZERO);
    /// assert!(!node0.actions().is_empty());
    ///
    /// // [NOTE] To complete the cluster creation, the user needs to handle the queued actions.
    /// ```
    pub fn start(id: NodeId) -> Self {
        Self::new(id)
    }

    /// Restarts a node.
    ///
    /// `current_term`, `voted_for`, and `log` are restored from persistent storage.
    ///
    /// # Persistence requirements
    ///
    /// The Raft algorithm relies on the persistent state being reliable. This crate
    /// makes the reliance explicit at the API boundary:
    ///
    /// - `current_term`, `voted_for`, and `log` must be restored without loss from
    ///   the state that noraft last requested to be persisted.
    /// - Before sending an outbound message that a batch of [`Action`]s yielded,
    ///   the crate user must complete the persistence requested by the
    ///   preceding storage [`Action`]s in the same batch (`SaveCurrentTerm`,
    ///   `SaveVotedFor`, `AppendLogEntries`). See [`Actions`] for the ordering
    ///   contract that makes this well-defined.
    /// - A node whose persistent state is suspected of loss or corruption must
    ///   not be restarted into the cluster; see [Signs that log loss may have
    ///   occurred](#signs-that-log-loss-may-have-occurred) below.
    /// - If any of the above requirements are violated, the situation is outside
    ///   the support scope of this crate and neither safety nor liveness is
    ///   guaranteed.
    ///
    /// noraft cannot verify the integrity of the storage itself. Checksums,
    /// redundancy, corruption detection, and the decision to discard a node
    /// belong to the crate user.
    ///
    /// # Signs that log loss may have occurred
    ///
    /// If the tail of a node's log is lost and the node restarts while the same
    /// leader is still in office, the leader still holds the pre-failure
    /// `match_index` for that node. Any subsequent `AppendEntries` reply carrying
    /// a smaller last log index is ignored as a delayed reply, so the restarted
    /// node's log synchronization may be retried repeatedly without progress.
    ///
    /// This state is not a confirmed diagnosis of a lost log tail. Message loss
    /// or delay, and stalls in the persistence pipeline, can also produce the
    /// same symptom. Treat it as a situation that warrants investigation.
    ///
    /// The crate user should investigate in the following order:
    ///
    /// 1. Confirm that `AppendEntries` deliveries to the restarted node and its
    ///    persistence requests are still making progress.
    /// 2. Confirm from the integration layer's own bookkeeping that the leader
    ///    and the term have not changed across the failure boundary (noraft
    ///    itself does not retain this information across restarts).
    /// 3. Compare the restarted node's local last index with the leader's view
    ///    of that node's replication progress: query the current leader with
    ///    [`Node::follower_match_index`] for the restarted node's id. A local
    ///    last index that is strictly smaller than the returned `match_index`
    ///    means the follower cannot still hold the log tail the current-tenure
    ///    leader has already acknowledged. Note that a `None` result — the
    ///    queried node is no longer leader, the tenure has changed, or the id
    ///    is not tracked — leaves this step inconclusive.
    /// 4. Check whether the leader's
    ///    [`NodeMetrics::append_entries_replies_ignored_behind_match_index`]
    ///    counter (the number of `AppendEntries` replies ignored for reporting a
    ///    last log index smaller than `match_index`) is growing continuously.
    /// 5. If the local last index is smaller, inspect the storage log, the
    ///    checksums, and the failure history.
    ///
    /// [`NodeMetrics::append_entries_replies_ignored_behind_match_index`] is a
    /// leader-wide aggregate counter and does not identify which follower is
    /// responsible. Treat it as an auxiliary signal to combine with the stalled
    /// restarted node and the transport log the integration layer keeps.
    ///
    /// noraft does not attempt any automatic recovery from this state. The
    /// leader keeps dropping the follower's replies until the operator
    /// intervenes or a leader change resets `match_index`. Once log loss or
    /// corruption is confirmed, the affected node must not be reused as-is to
    /// catch up: stop and isolate it, remove it from the cluster configuration,
    /// and re-add it as a fresh node backed by healthy storage if needed.
    ///
    /// If a leader change happens during the outage, the follower's
    /// `match_index` is reset by the new leader, so the stall does not appear
    /// and log loss can escape observation. Successful catch-up also does not
    /// prove storage integrity, so the decision to reuse a node after an
    /// abnormal termination must ultimately rest on the storage guarantees the
    /// crate user provides.
    ///
    /// # Examples
    /// ```
    /// // Loads the persistent state.
    /// let current_term = /* ... ; */
    /// # noraft::Term::new(1);
    /// let voted_for = /* ... ; */
    /// # None;
    /// let log = /* ... ; */
    /// # noraft::Log::new(noraft::ClusterConfig::new(), noraft::LogEntries::new(noraft::LogPosition::ZERO));
    ///
    /// // Restarts a node.
    /// let snapshot_index = log.snapshot_position().index;
    /// let node = noraft::Node::restart(noraft::NodeId::new(0), current_term, voted_for, log);
    /// assert!(node.role().is_follower());
    /// assert_eq!(node.commit_index(), snapshot_index);
    ///
    /// // Unlike `Node::start()`, the restarted node has actions to execute.
    /// assert!(!node.actions().is_empty());
    /// ```
    pub fn restart(id: NodeId, current_term: Term, voted_for: Option<NodeId>, log: Log) -> Self {
        let mut node = Self::new(id);

        node.current_term = current_term;
        node.voted_for = voted_for;
        node.log = log;
        node.commit_index = node.log.snapshot_position().index;
        node.actions.set(Action::SetElectionTimeout);

        node
    }

    /// Creates a new cluster.
    ///
    /// This method returns a [`LogPosition`] associated with a log entry.
    /// The log entry will be accepted when the initial cluster configuration is successfully committed.
    ///
    /// To proceed the cluster creation, the user needs to handle the queued actions after calling this method.
    ///
    /// # Preconditions
    ///
    /// This method returns [`LogPosition::INVALID`] if the following preconditions are not met:
    /// - This node (`self`) is a newly started node.
    /// - `initial_voters` contains at least one node.
    ///
    /// Theoretically, it is acceptable to exclude the self node from `initial_voters`
    /// (although it is not practical).
    ///
    /// # Notes
    ///
    /// Raft algorithm assumes that each node in a cluster belongs to only one cluster at a time.
    /// Therefore, including nodes that are already part of another cluster in the `initial_voters`
    /// will result in undefined behavior.
    pub fn create_cluster(&mut self, initial_voters: &[NodeId]) -> LogPosition {
        if self.log.last_position() != LogPosition::ZERO {
            return LogPosition::INVALID;
        }
        if !self.config().voters.is_empty() {
            return LogPosition::INVALID;
        }
        if initial_voters.is_empty() {
            return LogPosition::INVALID;
        }

        let mut config = ClusterConfig::new();
        config.voters.extend(initial_voters.iter().copied());
        let entry = LogEntry::ClusterConfig(config);
        self.actions
            .set(Action::AppendLogEntries(LogEntries::from_iter(
                LogPosition::ZERO,
                core::iter::once(entry.clone()),
            )));
        self.log.entries_mut().push(entry.clone());

        self.transition_to_candidate();

        self.log.last_position()
    }

    fn new(id: NodeId) -> Self {
        let config = ClusterConfig::new();
        Self {
            id,
            voted_for: None,
            current_term: Term::ZERO,
            log: Log::new(config, LogEntries::new(LogPosition::ZERO)),
            commit_index: LogIndex::ZERO,
            actions: Actions::default(),
            role: RoleState::Follower,
            metrics: NodeMetrics::default(),
        }
    }

    /// Returns the identifier of this node.
    pub fn id(&self) -> NodeId {
        self.id
    }

    /// Returns the role of this node.
    pub fn role(&self) -> Role {
        match self.role {
            RoleState::Follower => Role::Follower,
            RoleState::Candidate { .. } => Role::Candidate,
            RoleState::Leader { .. } => Role::Leader,
        }
    }

    /// Returns the current term of this node.
    pub fn current_term(&self) -> Term {
        self.current_term
    }

    /// Returns the identifier of the node for which this node voted in the current term.
    ///
    /// If `self.role()` is not [`Role::Candidate`], the returned node may be the leader of the current term.
    pub fn voted_for(&self) -> Option<NodeId> {
        self.voted_for
    }

    /// Returns the in-memory representation of the local log of this node.
    pub fn log(&self) -> &Log {
        &self.log
    }

    /// Returns the commit index of this node.
    ///
    /// [`LogEntry::Command`] entries up to this index are safely applied to the state machine managed by the user.
    pub fn commit_index(&self) -> LogIndex {
        self.commit_index
    }

    /// Returns the current cluster configuration of this node.
    ///
    /// This is shorthand for `self.log().latest_config()`.
    pub fn config(&self) -> &ClusterConfig {
        self.log.latest_config()
    }

    /// Returns an iterator over the identifiers of the peers of this node.
    ///
    /// "Peers" means all unique nodes in the current cluster configuration except for this node.
    pub fn peers(&self) -> impl '_ + Iterator<Item = NodeId> {
        self.config()
            .unique_nodes()
            .filter(move |&node| node != self.id)
    }

    /// Returns a reference to the pending actions for this node.
    pub fn actions(&self) -> &Actions {
        &self.actions
    }

    /// Returns a mutable reference to the pending actions for this node.
    ///
    /// # Note
    ///
    /// It is the user's responsibility to execute these actions.
    pub fn actions_mut(&mut self) -> &mut Actions {
        &mut self.actions
    }

    /// Returns a shared reference to the cumulative counters for this node.
    ///
    /// The returned [`NodeMetrics`] is owned by the node and is updated in place
    /// as protocol events occur. Callers only receive read access, so the
    /// counters cannot be reset or modified from outside. Scraping the counters
    /// therefore never copies the whole snapshot.
    ///
    /// See [`NodeMetrics`] for the meaning of each counter and for guidance on
    /// how to interpret their values.
    pub fn metrics(&self) -> &NodeMetrics {
        &self.metrics
    }

    /// Returns the `match_index` this node, acting as leader, currently tracks
    /// for the follower identified by `id`.
    ///
    /// `Some(match_index)` is returned when all of the following hold:
    ///
    /// - `self.role()` is [`Role::Leader`].
    /// - `id` appears in `self.config().unique_nodes()` (i.e. it is a voter,
    ///   a `new_voters`-only node in a joint consensus, or a non-voter) and
    ///   is not `self.id()`.
    ///
    /// In every other case (any follower or candidate role, an unknown `id`,
    /// `self.id()` itself, or a node that has been dropped from the current
    /// configuration) [`None`] is returned.
    ///
    /// The value is the largest `last_position.index` this leader has observed
    /// from a successful `AppendEntries` reply from that follower during the
    /// current leader tenure. It grows monotonically for as long as the node
    /// remains leader, and it is reset to [`LogIndex::ZERO`] whenever the node
    /// transitions to leader — including a re-election of the same node —
    /// because [`Node::restart`] does not carry `match_index` across a leader
    /// change.
    ///
    /// # Interpreting the value
    ///
    /// - The value only says that a successful reply was received. It is not a
    ///   claim about what the follower has already committed nor about what its
    ///   state machine has applied.
    /// - Provided the persistence contract on [`Node::restart`] (see the
    ///   "Persistence requirements" section) is honored on the follower side,
    ///   `match_index` is a lower bound on the log the follower has already
    ///   made durable. If the contract is violated (for example a reply is sent
    ///   before its preceding `AppendLogEntries` `Action` is persisted), the
    ///   lower-bound interpretation no longer holds.
    /// - The value cannot detect every kind of divergence on its own. In
    ///   particular:
    ///     - A leader change during the outage resets `match_index` back to 0
    ///       on the next leader.
    ///     - The counter is an index only; it cannot distinguish an entry
    ///       whose term or payload has since been overwritten.
    ///     - A complete loss of persistent storage is handled by the
    ///       "Signs that log loss may have occurred" section on
    ///       [`Node::restart`], not by this value alone.
    ///
    /// # Deciding when to promote a non-voter
    ///
    /// [`ClusterConfig::non_voters`] recommends adding a new node as a
    /// non-voter first, letting it catch up, and then promoting it to a voter.
    /// This method lets the integration layer observe that catch-up directly:
    /// once `follower_match_index(new_node)` is close enough to
    /// `self.log().last_position().index` (the exact threshold is an
    /// integration-side policy), the node is ready to be promoted with
    /// [`Node::propose_config`].
    ///
    /// # Diagnosing suspected log loss before rejoining
    ///
    /// The "Signs that log loss may have occurred" section on [`Node::restart`]
    /// covers the *after-the-fact* case where a restarted node has already
    /// rejoined and its replication has stalled. This method also enables a
    /// *before-rejoining* check: given a still-isolated node whose persistent
    /// state was recovered from disk, an operator can query the current leader
    /// for that node's `match_index` and refuse to bring the node back if its
    /// local log is shorter than what the leader has already acknowledged.
    ///
    /// The concrete procedure is:
    ///
    /// 1. Read `local_last_index` from the recovered persistent storage.
    /// 2. Ask the current leader instance for `Node::id`, `Node::current_term`,
    ///    and `Node::follower_match_index(target)` in the same tenure. Because
    ///    they are consecutive `&self` calls on the same [`Node`], the three
    ///    values naturally form a single snapshot; every leader tenure starts
    ///    with a fresh per-follower `match_index` (initialized to 0), so the
    ///    returned value is guaranteed to belong to the current tenure.
    /// 3. If `local_last_index < follower_match_index`, the follower cannot
    ///    still hold a log tail that the current-tenure leader has already
    ///    seen. Isolate the node and refuse to restart it.
    ///
    /// The converse does not hold: `local_last_index >= follower_match_index`
    /// is not a proof of storage integrity. A leader change during the outage,
    /// or a leader that has not yet learned the follower's pre-failure
    /// progress, can hide the loss. Continuously mirroring the value into
    /// external monitoring gives more coverage.
    ///
    /// This method is a signal; it is not a decision procedure for whether a
    /// node is safe to restart. The final call to isolate, discard, or re-add
    /// belongs to the integration layer's storage guarantees.
    pub fn follower_match_index(&self, id: NodeId) -> Option<LogIndex> {
        let RoleState::Leader { followers, .. } = &self.role else {
            return None;
        };
        followers.get(&id).map(|f| f.match_index)
    }

    fn transition_to_leader(&mut self) {
        debug_assert_eq!(self.voted_for, Some(self.id));

        self.metrics.leaderships_started = self.metrics.leaderships_started.saturating_add(1);

        let quorum = Quorum::new(self.config());
        let followers = BTreeMap::new();
        let solo_voter = self.is_self_solo_voter();
        self.role = RoleState::Leader {
            followers,
            quorum,
            solo_voter,
        };
        self.rebuild_followers();
        self.rebuild_quorum();

        self.propose(LogEntry::Term(self.current_term));
    }

    fn is_self_solo_voter(&self) -> bool {
        self.config().unique_voters().count() == 1 && self.config().voters.contains(&self.id)
    }

    fn refresh_solo_voter(&mut self) {
        let is_self_solo_voter = self.is_self_solo_voter();
        if let RoleState::Leader { solo_voter, .. } = &mut self.role {
            *solo_voter = is_self_solo_voter;
        }
    }

    fn transition_to_candidate(&mut self) {
        if !self.log.latest_config().is_voter(self.id) {
            // Non voter or removed node cannot become a candidate.
            return;
        }

        self.metrics.elections_started = self.metrics.elections_started.saturating_add(1);

        self.set_current_term(self.current_term.next());
        self.set_voted_for(Some(self.id));

        let solo_voter = self.is_self_solo_voter();
        if solo_voter {
            self.transition_to_leader();
            return;
        }

        self.role = RoleState::Candidate {
            granted_votes: core::iter::once(self.id).collect(),
        };

        self.actions
            .set(Action::BroadcastMessage(Message::request_vote_call(
                self.current_term,
                self.id,
                self.log.last_position(),
            )));
        self.actions.set(Action::SetElectionTimeout);
    }

    fn transition_to_follower(&mut self, term: Term) {
        debug_assert!(self.current_term <= term);

        self.set_current_term(term);
        self.set_voted_for(None);
        self.role = RoleState::Follower;
        self.actions.set(Action::SetElectionTimeout);
    }

    /// Proposes a user-defined command ([`LogEntry::Command`]).
    ///
    /// This method returns a [`LogPosition`] that associated with the log entry for the proposed command.
    /// To determine whether the command has been committed, you can use the [`Node::get_commit_status()`] method.
    /// To known where the command is commited or not, you can use [`Node::get_commit_status()`] method.
    /// Committed commands can be applied to the state machine managed by the user.
    ///
    /// [`Node::get_commit_status()`] is useful for determining when to send the command result back to the client
    /// that triggered the command (if such a client exists).
    /// To detect all committed commands that need to be applied to the state machine,
    /// it is recommended to use [`Node::commit_index()`] since it considers commands proposed by other nodes.
    ///
    /// Note that this crate does not manage the detail of user-defined commands,
    /// so this method takes no arguments.
    /// It is the user's responsibility to mapping the log index of the proposed command to
    /// the actual command data.
    ///
    /// # Preconditions
    ///
    /// This method returns [`LogPosition::INVALID`] if the following preconditions are not met:
    /// - `self.role().is_leader()` is [`true`].
    ///
    /// # Pipelining
    ///
    /// [`Node::propose_command()`] can be called multiple times before any action is executed.
    /// In such cases, the pending actions are consolidated, reducing the overall I/O cost.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut node = /* ... ; */
    /// # noraft::Node::start(noraft::NodeId::new(0));
    ///
    /// let commit_position = node.propose_command();
    /// if commit_position.is_invalid() {
    ///     // `node` is not the leader.
    ///     if let Some(maybe_leader) = node.voted_for() {
    ///         // Retry with the possible leader or reply to the client that the command is rejected.
    ///         // ...
    ///     }
    ///     return;
    /// }
    ///
    /// // Need to map the log index to the actual command data for
    /// // exeucting `Action::AppendLogEntries(_)` queued by the node.
    /// assert!(node.actions().append_log_entries.is_some());
    /// let index = commit_position.index;
    /// # let _ = index;
    /// // ... executing actions ...
    ///
    /// while node.get_commit_status(commit_position).is_in_progress() {
    ///     // ... executing actions ...
    /// }
    ///
    /// if node.get_commit_status(commit_position).is_rejected() {
    ///    // Retry with another node or reply to the client that the command is rejected.
    ///    // ...
    ///    return;
    /// }
    /// assert!(node.get_commit_status(commit_position).is_committed());
    ///
    /// // Apply all committed commands to the state machine.
    /// let last_applied_index = /* ...; */
    /// # noraft::LogIndex::ZERO;
    /// for index in (last_applied_index.get() - 1)..=node.commit_index().get() {
    ///     let index = noraft::LogIndex::new(index);
    ///     if node.log().entries().get_entry(index) != Some(noraft::LogEntry::Command) {
    ///         continue;
    ///     }
    ///     // Apply the command to the state machine.
    ///     // ...
    ///
    ///     if index == commit_position.index {
    ///         // Reply to the client that the command is committed.
    ///         // ...
    ///     }
    /// }
    /// ```
    pub fn propose_command(&mut self) -> LogPosition {
        if !matches!(self.role, RoleState::Leader { .. }) {
            return LogPosition::INVALID;
        }
        self.propose(LogEntry::Command)
    }

    fn propose(&mut self, entry: LogEntry) -> LogPosition {
        debug_assert!(self.role().is_leader());

        let old_last_position = self.log.last_position();
        self.append_proposed_log_entry(&entry);

        let RoleState::Leader { followers, .. } = &self.role else {
            unreachable!();
        };
        if !followers.is_empty() {
            let call = Message::append_entries_call(
                self.current_term,
                self.id,
                self.commit_index,
                LogEntries::from_iter(old_last_position, core::iter::once(entry)),
            );
            self.actions.set(Action::BroadcastMessage(call));
        }
        self.actions.set(Action::SetElectionTimeout);

        self.log.last_position()
    }

    // Preserves the `match_index` of existing followers and only inserts a
    // fresh `Follower::new()` for newly added members. Followers dropped from
    // the latest configuration are removed.
    //
    // Keeping existing entries untouched is what makes `match_index`
    // monotonically non-decreasing within a single leader's tenure, which the
    // behind-match_index check in `handle_append_entries_reply` relies on.
    fn rebuild_followers(&mut self) {
        let RoleState::Leader { followers, .. } = &mut self.role else {
            unreachable!();
        };

        let config = self.log.latest_config();

        for id in config.unique_nodes() {
            if id == self.id || followers.contains_key(&id) {
                continue;
            }
            followers.insert(id, Follower::new());
        }

        followers.retain(|id, _| config.contains(*id));
    }

    fn rebuild_quorum(&mut self) {
        let self_id = self.id;
        let self_last = self.log.last_position().index;
        let config = self.log.latest_config();

        let RoleState::Leader {
            quorum, followers, ..
        } = &mut self.role
        else {
            unreachable!();
        };

        *quorum = Quorum::new(config);
        quorum.update_match_index(config, self_id, LogIndex::ZERO, self_last);
        for (&id, follower) in &*followers {
            quorum.update_match_index(config, id, LogIndex::ZERO, follower.match_index);
        }
    }

    fn update_commit_index_if_possible(&mut self) {
        let RoleState::Leader { quorum, .. } = &mut self.role else {
            unreachable!();
        };

        let new_commit_index = quorum.smallest_majority_index();
        if new_commit_index <= self.commit_index
            || self.log.entries().get_term(new_commit_index) != Some(self.current_term)
        {
            return;
        }
        // [NOTE] Commit index is updated.

        self.commit_index = new_commit_index;

        if new_commit_index < self.log.latest_config_index() {
            return;
        }
        // [NOTE] The latest configuration has been committed.

        if self.log.latest_config().is_joint_consensus() {
            self.finalize_joint_consensus();
        } else if !self.log.latest_config().voters.contains(&self.id) {
            // The leader, who is not a voter in the latest committed configuration, steps down here.
            //
            // The new election will begin after the followers detect the leader's absence
            // (i.e., when the election timeout expires on the followers).
            self.transition_to_follower(self.current_term);
        }
    }

    fn finalize_joint_consensus(&mut self) {
        debug_assert!(self.role().is_leader());
        debug_assert!(self.log.latest_config().is_joint_consensus());

        let mut new_config = self.log.latest_config().clone();
        new_config.voters = core::mem::take(&mut new_config.new_voters);
        debug_assert!(!new_config.voters.is_empty());

        self.propose(LogEntry::ClusterConfig(new_config));
    }

    /// Proposes a new cluster configuration ([`LogEntry::ClusterConfig`]).
    ///
    /// If `new_config.new_voters` is not empty, the cluster will transition into a joint consensus state.
    /// In this state, leader elections and commit proposals require a majority from both the old and
    /// new voters independently.
    /// Once `new_config` is committed, a new configuration, which includes only the new voters
    /// (and any non-voters, if any), will be automatically proposed to finalize the joint consensus.
    ///
    /// `new_config.new_voters` does not need to include the self node.
    /// If it does not, the leader self node will transition to a follower
    /// when the final configuration is committed.
    ///
    /// Note that a change in `new_config.non_voters` does not require a joint consensus.
    ///
    /// # Preconditions
    ///
    /// This method returns [`LogPosition::INVALID`] if the following preconditions are not met:
    /// - `self.role().is_leader()` is [`true`].
    /// - `new_config.voters` is equal to `self.config().voters`.
    /// - A node is either a voter or a non-voter in the new configuration (not both).
    /// - `self.config().is_joint_consensus()` is [`false`] (i.e., there is no other configuration change in progress).
    ///
    /// # Examples
    ///
    /// ```
    /// let mut node = /* ... ; */
    /// # noraft::Node::start(noraft::NodeId::new(1));
    ///
    /// // Propose a new configuration with adding node 4 and removing node 2.
    /// let new_config =
    ///     node.config().to_joint_consensus(&[noraft::NodeId::new(4)], &[noraft::NodeId::new(2)]);
    /// node.propose_config(new_config);
    /// ```
    pub fn propose_config(&mut self, new_config: ClusterConfig) -> LogPosition {
        if !self.role().is_leader() {
            return LogPosition::INVALID;
        }
        if self.log.latest_config().voters != new_config.voters {
            return LogPosition::INVALID;
        }
        if !new_config.voters.is_disjoint(&new_config.non_voters)
            || !new_config.new_voters.is_disjoint(&new_config.non_voters)
        {
            return LogPosition::INVALID;
        }
        if self.log.latest_config().is_joint_consensus() {
            return LogPosition::INVALID;
        }

        self.propose(LogEntry::ClusterConfig(new_config))
    }

    /// Returns the commit status of a log entry associated with the given position.
    pub fn get_commit_status(&self, position: LogPosition) -> CommitStatus {
        if position.index < self.log().entries().prev_position().index {
            return CommitStatus::Unknown;
        } else if position.index <= self.commit_index() {
            if self.log().entries().contains(position) {
                return CommitStatus::Committed;
            } else {
                return CommitStatus::Rejected;
            }
        } else if let Some(term) = self.log().entries().get_term(self.commit_index())
            && position.term < term
        {
            return CommitStatus::Rejected;
        }
        CommitStatus::InProgress
    }

    /// Sends a heartbeat (i.e, an empty `AppendEntriesCall` message) to all followers.
    ///
    /// This method returns `false` if this node is not the leader.
    ///
    /// This method can be used to perform consistent queries through the following steps:
    /// 1. Invoke `heartbeat()`.
    /// 2. Attach a user-defined request identifier (e.g., timestamp) to the next message to be broadcast.
    /// 3. Wait until this node receives the majority of response messages that match the identifier,
    ///    to confirm that this node is still the leader of the cluster.
    /// 4. Execute the consistent query.
    pub fn heartbeat(&mut self) -> bool {
        let RoleState::Leader { followers, .. } = &self.role else {
            return false;
        };

        if !followers.is_empty() {
            let call = Message::append_entries_call(
                self.current_term,
                self.id,
                self.commit_index,
                LogEntries::new(self.log.entries().last_position()),
            );
            self.actions.set(Action::BroadcastMessage(call));
        }
        self.actions.set(Action::SetElectionTimeout);

        true
    }

    fn append_proposed_log_entry(&mut self, entry: &LogEntry) {
        let RoleState::Leader { quorum, .. } = &mut self.role else {
            unreachable!();
        };

        let old_last_index = self.log.last_position().index;
        self.actions
            .set(Action::AppendLogEntries(LogEntries::from_iter(
                self.log.last_position(),
                core::iter::once(entry.clone()),
            )));
        self.log.entries_mut().push(entry.clone());

        quorum.update_match_index(
            self.log.latest_config(),
            self.id,
            old_last_index,
            self.log.last_position().index,
        );

        if matches!(entry, LogEntry::ClusterConfig(_)) {
            self.rebuild_followers();
            self.rebuild_quorum();
            self.refresh_solo_voter();
        }

        if matches!(
            self.role,
            RoleState::Leader {
                solo_voter: true,
                ..
            }
        ) {
            self.update_commit_index_if_possible();
        }
    }

    fn append_log_entries_from_leader(&mut self, entries: &LogEntries) -> Result<bool, Error> {
        debug_assert!(self.role().is_follower());

        if self.log.entries().contains(entries.last_position()) {
            // Already up-to-date.
            return Ok(self.log().last_position() == entries.last_position());
        }
        if !self.log.entries().contains(entries.prev_position()) {
            // Cannot append.
            if self
                .log
                .entries()
                .contains_index(entries.prev_position().index)
            {
                self.metrics.log_divergences_detected_by_follower = self
                    .metrics
                    .log_divergences_detected_by_follower
                    .saturating_add(1);
                // Remove the divergence entries.
                // Note that `Action::AppendLogEntries` is not triggered until
                // the root of the divergence point is identified.
                let new_len = entries
                    .prev_position()
                    .index
                    .get()
                    .checked_sub(self.log.snapshot_position().index.get() + 1);
                if let Some(new_len) = new_len {
                    self.log.entries_mut().truncate(new_len as usize);
                    debug_assert_eq!(
                        self.log.last_position().index.get() + 1,
                        entries.prev_position().index.get()
                    );
                } else {
                    return Err(Error::LocalSnapshotMismatch);
                }
            }
            return Ok(false);
        }

        // Append.
        let entries = entries.strip_common_prefix(self.log.entries());
        self.log.entries_mut().append(&entries);
        self.actions.set(Action::AppendLogEntries(entries));

        Ok(true)
    }

    fn set_current_term(&mut self, term: Term) {
        self.current_term = term;
        self.actions.set(Action::SaveCurrentTerm);
    }

    fn set_voted_for(&mut self, voted_for: Option<NodeId>) {
        self.voted_for = voted_for;
        self.actions.set(Action::SaveVotedFor);
    }

    /// Returns `true` if this node considers the given message a potentially disruptive
    /// `RequestVoteCall`.
    ///
    /// This method is intended for pre-filtering at the integration layer.
    /// The returned value is `true` when all of the following conditions are met:
    /// - `msg` is [`Message::RequestVoteCall`]
    /// - `msg.term()` is greater than this node's current term
    /// - this node is not a candidate
    /// - this node has already voted for another node in the current term
    ///
    /// Such a message might be sent from a removed node and could disrupt an active leader.
    /// For details, see section 6 of the Raft paper.
    ///
    /// If your integration already guarantees in advance (for example, by a pre-vote
    /// mechanism) that disruptive `RequestVoteCall` messages are not sent, avoid using
    /// this method for filtering. Otherwise, valid `RequestVoteCall` messages may be
    /// dropped unexpectedly.
    ///
    /// Example scenario: A follower's election timeout has already expired, but the
    /// integration runs pre-vote first and delays calling
    /// [`Node::handle_election_timeout`] so that the node does not immediately become a
    /// candidate and increment its term. During that gap, the node stays as a follower.
    /// If `RequestVoteCall` messages are dropped by this method in that period and the
    /// pre-vote does not succeed, the node can remain follower longer than necessary.
    /// If many running followers stay in that state, leader election can stall for a
    /// very long time because no candidate is able to gather votes.
    ///
    /// Note that [`Node::handle_message`] does not automatically ignore this case.
    /// If you want to drop these messages, call this method before passing the message to
    /// [`Node::handle_message`].
    pub fn could_be_disruptive_request_vote(&self, msg: &Message) -> bool {
        self.current_term < msg.term()
            && matches!(msg, Message::RequestVoteCall { .. })
            && !matches!(self.role, RoleState::Candidate { .. })
            && self.voted_for.is_some_and(|id| id != msg.from())
    }

    /// Handles an incoming message from other nodes.
    ///
    /// # Note
    ///
    /// This method processes potentially disruptive `RequestVoteCall` messages as normal input.
    /// To pre-filter such messages, call [`Node::could_be_disruptive_request_vote`]
    /// before invoking this method.
    ///
    /// # Errors
    ///
    /// If this method returns [`Err`], the node has detected a state that cannot
    /// occur in a valid Raft execution. The crate user should stop using this node
    /// and inspect or rebuild its persistent state.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut node = /* ... ; */
    /// # noraft::Node::start(noraft::NodeId::new(1));
    ///
    /// let msg = /* ... ; */
    /// # noraft::Message::RequestVoteReply { from: noraft::NodeId::new(1), term: noraft::Term::new(1), vote_granted: true };
    /// node.handle_message(&msg).expect("message handling should succeed");
    ///
    /// // Execute actions queued by the message handling.
    /// for action in node.actions_mut() {
    ///     // ...
    /// }
    /// ```
    pub fn handle_message(&mut self, msg: &Message) -> Result<(), Error> {
        if msg.from() == self.id {
            self.metrics.self_messages_ignored =
                self.metrics.self_messages_ignored.saturating_add(1);
            return Ok(());
        }
        if self.current_term < msg.term() {
            self.metrics.term_advances_from_messages =
                self.metrics.term_advances_from_messages.saturating_add(1);
            self.transition_to_follower(msg.term());
        }

        match msg {
            Message::RequestVoteCall {
                from,
                term,
                last_position,
            } => self.handle_request_vote_call(*from, *term, *last_position),
            Message::RequestVoteReply {
                from,
                term,
                vote_granted,
            } => self.handle_request_vote_reply(*from, *term, *vote_granted),
            Message::AppendEntriesCall {
                from,
                term,
                commit_index,
                entries,
            } => self.handle_append_entries_call(*from, *term, *commit_index, entries)?,
            Message::AppendEntriesReply {
                from,
                term,
                last_position,
            } => self.handle_append_entries_reply(*from, *term, *last_position),
        }

        Ok(())
    }

    fn handle_request_vote_call(&mut self, from: NodeId, term: Term, last_position: LogPosition) {
        if term < self.current_term {
            self.metrics.request_vote_calls_rejected_by_old_term = self
                .metrics
                .request_vote_calls_rejected_by_old_term
                .saturating_add(1);
            // Needs to reply to update the sender's term.
            let reply = Message::request_vote_reply(self.current_term, self.id, false);
            self.actions.set(Action::SendMessage(from, reply));
            return;
        }

        if self.log.last_position() > last_position {
            self.metrics.request_vote_calls_rejected_by_log = self
                .metrics
                .request_vote_calls_rejected_by_log
                .saturating_add(1);
            // Deny the vote without sending an explicit false reply. A negative reply in
            // the candidate's current term would not change the candidate's state in this
            // implementation; the candidate will retry after its election timeout if it
            // cannot collect a majority.
            return;
        }

        if self.voted_for.is_none() {
            self.set_voted_for(Some(from));
        }

        if self.voted_for != Some(from) {
            self.metrics.request_vote_calls_rejected_by_existing_vote = self
                .metrics
                .request_vote_calls_rejected_by_existing_vote
                .saturating_add(1);
            // Deny the vote without sending an explicit false reply. This node is either a
            // candidate, a leader, or has already voted for another node in this term.
            return;
        }
        debug_assert!(self.role().is_follower());

        // This follower votes for the candidate.
        let reply = Message::request_vote_reply(self.current_term, self.id, true);
        self.actions.set(Action::SendMessage(from, reply));
        self.actions.set(Action::SetElectionTimeout);
    }

    fn handle_request_vote_reply(&mut self, from: NodeId, term: Term, vote_granted: bool) {
        let RoleState::Candidate { granted_votes } = &mut self.role else {
            self.metrics
                .request_vote_replies_ignored_while_not_candidate = self
                .metrics
                .request_vote_replies_ignored_while_not_candidate
                .saturating_add(1);
            return;
        };
        if !vote_granted {
            return;
        }
        if term < self.current_term {
            // Delayed (obsolete) reply from an old term.
            self.metrics.request_vote_replies_ignored_from_old_terms = self
                .metrics
                .request_vote_replies_ignored_from_old_terms
                .saturating_add(1);
            return;
        }
        granted_votes.insert(from);

        let config = self.log.latest_config();
        let n = config
            .voters
            .iter()
            .filter(|v| granted_votes.contains(v))
            .count();
        if n < self.log.latest_config().voter_majority_count() {
            return;
        }

        let n = config
            .new_voters
            .iter()
            .filter(|v| granted_votes.contains(v))
            .count();
        if n < config.new_voter_majority_count() {
            return;
        }

        self.transition_to_leader();
    }

    fn handle_append_entries_call(
        &mut self,
        from: NodeId,
        term: Term,
        leader_commit: LogIndex,
        entries: &LogEntries,
    ) -> Result<(), Error> {
        if term < self.current_term {
            self.metrics.append_entries_calls_rejected_by_old_term = self
                .metrics
                .append_entries_calls_rejected_by_old_term
                .saturating_add(1);
            // Needs to reply to update the sender's term.
            self.reply_append_entries(from);
            return Ok(());
        }

        if self.role().is_leader() {
            self.metrics
                .same_term_append_entries_calls_received_by_leader = self
                .metrics
                .same_term_append_entries_calls_received_by_leader
                .saturating_add(1);
            // A same-term leader conflict cannot happen in a correct Raft execution.
            return Ok(());
        }

        if !self.role().is_follower() {
            // A candidate recognizes the sender as the legitimate leader for this term.
            self.role = RoleState::Follower;
        }

        if self.voted_for != Some(from) {
            self.set_voted_for(Some(from));
        }

        let no_divergence = self.append_log_entries_from_leader(entries)?;
        if no_divergence {
            let next_commit_index = leader_commit.min(self.log.last_position().index);
            if self.commit_index < next_commit_index {
                self.commit_index = next_commit_index;
            }
        }

        self.reply_append_entries(from);
        self.actions.set(Action::SetElectionTimeout);
        Ok(())
    }

    fn handle_append_entries_reply(
        &mut self,
        from: NodeId,
        term: Term,
        follower_last_position: LogPosition,
    ) {
        if term < self.current_term {
            self.metrics.append_entries_replies_ignored_from_old_terms = self
                .metrics
                .append_entries_replies_ignored_from_old_terms
                .saturating_add(1);
            // Delayed (obsolete) reply from an old term.
            return;
        }

        let RoleState::Leader {
            followers, quorum, ..
        } = &mut self.role
        else {
            self.metrics.append_entries_replies_ignored_while_not_leader = self
                .metrics
                .append_entries_replies_ignored_while_not_leader
                .saturating_add(1);
            return;
        };

        let Some(follower) = followers.get_mut(&from) else {
            self.metrics
                .append_entries_replies_ignored_from_unknown_nodes = self
                .metrics
                .append_entries_replies_ignored_from_unknown_nodes
                .saturating_add(1);
            // Replies from unknown nodes are ignored.
            return;
        };

        if follower_last_position.index < follower.match_index {
            self.metrics
                .append_entries_replies_ignored_behind_match_index = self
                .metrics
                .append_entries_replies_ignored_behind_match_index
                .saturating_add(1);
            // Delayed reply behind the acknowledged match index. The
            // persistence contract on `Node::restart` guarantees this cannot
            // happen legitimately within a single leader's tenure.
            return;
        }

        if !self.log.entries().contains(follower_last_position) {
            if let Some(term) = self.log.entries().get_term(follower_last_position.index) {
                self.metrics.log_divergences_detected_by_leader = self
                    .metrics
                    .log_divergences_detected_by_leader
                    .saturating_add(1);
                // Delete the follower's last log entry.
                let index = follower_last_position.index;
                let call = Message::append_entries_call(
                    self.current_term,
                    self.id,
                    self.commit_index,
                    LogEntries::new(LogPosition { term, index }),
                );
                self.actions.set(Action::SendMessage(from, call));
            } else if self.log.last_position().index < follower_last_position.index {
                self.metrics.append_entries_replies_ahead_of_leader = self
                    .metrics
                    .append_entries_replies_ahead_of_leader
                    .saturating_add(1);
                // Something seems strange.
                // However, as the leader log grows, a divergence point will be detected.
            } else {
                // The follower's log is too old. Needs to install a snapshot.
                debug_assert!(follower_last_position.index <= self.log.snapshot_position().index);
                self.actions.set(Action::InstallSnapshot(from));
            }

            return;
        }

        // [NOTE]
        // This check should be done here because `self.log.last_position()` may be updated in
        // `self.update_commit_index_if_possible()`.
        let is_follower_up_to_date = follower_last_position.index == self.log.last_position().index;

        if follower.match_index < follower_last_position.index {
            let old_match_index = follower.match_index;
            follower.match_index = follower_last_position.index;

            quorum.update_match_index(
                self.log.latest_config(),
                from,
                old_match_index,
                follower.match_index,
            );

            if self.commit_index < follower.match_index {
                self.update_commit_index_if_possible();
            }
        }

        if is_follower_up_to_date {
            // The follower's log is up-to-date.
            return;
        }
        debug_assert!(self.log.entries().contains(follower_last_position));

        let Some(delta) = self.log.entries().since(follower_last_position) else {
            unreachable!();
        };
        let call =
            Message::append_entries_call(self.current_term, self.id, self.commit_index, delta);
        self.actions.set(Action::SendMessage(from, call));
    }

    fn reply_append_entries(&mut self, to: NodeId) {
        let reply =
            Message::append_entries_reply(self.current_term, self.id, self.log.last_position());
        self.actions.set(Action::SendMessage(to, reply));
    }

    /// Handles an election timeout.
    ///
    /// This method is typically invoked when the timeout set by [`Action::SetElectionTimeout`] expires.
    /// However, it can also be invoked by other means, such as to trigger a new election
    /// as quickly as possible when the crate user knows there is no leader.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut node = /* ... ; */
    /// # noraft::Node::start(noraft::NodeId::new(1));
    ///
    /// node.handle_election_timeout();
    ///
    /// // Execute actions queued by the timeout handling.
    /// for action in node.actions_mut() {
    ///     // ...
    /// }
    /// ```
    pub fn handle_election_timeout(&mut self) {
        match self.role {
            RoleState::Follower => {
                self.transition_to_candidate();
            }
            RoleState::Candidate { .. } => {
                self.transition_to_candidate();
            }
            RoleState::Leader { .. } => {
                self.heartbeat();
            }
        }
    }

    /// Updates this node's log ([`Log`]) to reflect the installation of a snapshot.
    ///
    /// If the node log contains `last_included_position`, log entries up to `last_included_position` are removed.
    /// If `last_included_position` is greater than the last log position, the log is replaced with an empty log starting at `last_included_position`.
    ///
    /// Note that how to install a snapshot is outside of the scope of this crate.
    ///
    /// # Preconditions
    ///
    /// This method returns [`false`] and ignores the installation if the following conditions are not met:
    /// - `last_included_position` is valid, which means:
    ///   - `self.log.entries().contains(last_included_position)` is [`true`].
    ///   - Additionally, if `self.role().is_leader()` is [`false`], it is also acceptable if `last_included_position.index` is greater than `self.commit_index()`.
    /// - `last_included_config` is the configuration at `last_included_position.index`.
    pub fn handle_snapshot_installed(
        &mut self,
        last_included_position: LogPosition,
        last_included_config: ClusterConfig,
    ) -> bool {
        if !self.is_valid_snapshot(&last_included_config, last_included_position) {
            return false;
        }
        if let Some(entries) = self.log.entries().since(last_included_position) {
            self.log = Log::new(last_included_config, entries);
        } else {
            self.log = Log::new(
                last_included_config,
                LogEntries::new(last_included_position),
            );
        }

        if let Some(entries) = &mut self.actions.append_log_entries {
            entries.handle_snapshot_installed(last_included_position);
            if entries.is_empty() {
                self.actions.append_log_entries = None;
            }
        }

        if let Some(msg) = &mut self.actions.broadcast_message {
            msg.handle_snapshot_installed(last_included_position);
        }
        for msg in self.actions.send_messages.values_mut() {
            msg.handle_snapshot_installed(last_included_position);
        }

        // Per-follower `match_index` is intentionally not updated here: what a
        // follower actually holds is only known from its `AppendEntriesReply`,
        // and taking a snapshot on the leader does not change that observation.
        true
    }

    fn is_valid_snapshot(
        &self,
        last_included_config: &ClusterConfig,
        last_included_position: LogPosition,
    ) -> bool {
        if self.commit_index() < last_included_position.index {
            return self.role() != Role::Leader;
        }
        if !self.log.entries().contains(last_included_position) {
            return false;
        }
        self.log.get_config(last_included_position.index) == Some(last_included_config)
    }
}

#[derive(Debug, Clone)]
enum RoleState {
    Follower,
    Candidate {
        granted_votes: BTreeSet<NodeId>,
    },
    Leader {
        followers: BTreeMap<NodeId, Follower>,
        quorum: Quorum,
        solo_voter: bool,
    },
}

#[derive(Debug, Clone)]
struct Follower {
    pub match_index: LogIndex,
}

impl Follower {
    pub fn new() -> Self {
        Self {
            match_index: LogIndex::new(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Message;

    fn pos(term: u64, index: u64) -> LogPosition {
        LogPosition {
            term: Term::new(term),
            index: LogIndex::new(index),
        }
    }

    fn config(voters: &[NodeId]) -> ClusterConfig {
        let mut c = ClusterConfig::new();
        for &v in voters {
            c.voters.insert(v);
        }
        c
    }

    fn follower(id: NodeId, voters: &[NodeId], term: u64, voted_for: Option<NodeId>) -> Node {
        let cfg = config(voters);
        let entries = LogEntries::from_iter(
            LogPosition::ZERO,
            core::iter::once(LogEntry::ClusterConfig(cfg.clone())),
        );
        Node::restart(id, Term::new(term), voted_for, Log::new(cfg, entries))
    }

    fn drain(node: &mut Node) {
        while node.actions_mut().next().is_some() {}
    }

    fn leader_with(id: NodeId, others: &[NodeId]) -> Node {
        let mut voters = alloc::vec::Vec::from([id]);
        voters.extend_from_slice(others);
        let mut node = Node::start(id);
        node.create_cluster(&voters);
        drain(&mut node);
        for &f in others {
            let msg = Message::request_vote_reply(node.current_term(), f, true);
            node.handle_message(&msg).unwrap();
            if node.role().is_leader() {
                break;
            }
        }
        assert!(node.role().is_leader());
        drain(&mut node);
        node.metrics = NodeMetrics::default();
        node
    }

    fn candidate_with(id: NodeId, others: &[NodeId]) -> Node {
        let mut voters = alloc::vec::Vec::from([id]);
        voters.extend_from_slice(others);
        let mut node = follower(id, &voters, 0, None);
        drain(&mut node);
        node.handle_election_timeout();
        assert!(node.role().is_candidate());
        drain(&mut node);
        node.metrics = NodeMetrics::default();
        node
    }

    #[test]
    fn metrics_default_after_start() {
        let node = Node::start(NodeId::new(0));
        assert_eq!(*node.metrics(), NodeMetrics::default());
    }

    #[test]
    fn metrics_default_after_restart() {
        let node = follower(NodeId::new(0), &[NodeId::new(0)], 3, Some(NodeId::new(1)));
        assert_eq!(*node.metrics(), NodeMetrics::default());
    }

    #[test]
    fn self_messages_ignored_counter() {
        let mut node = Node::start(NodeId::new(0));
        let msg = Message::request_vote_call(Term::new(1), NodeId::new(0), pos(0, 0));
        node.handle_message(&msg).unwrap();
        assert_eq!(node.metrics().self_messages_ignored, 1);
        assert_eq!(node.metrics().term_advances_from_messages, 0);
    }

    #[test]
    fn term_advances_from_messages_counter() {
        let mut node = follower(NodeId::new(0), &[NodeId::new(0), NodeId::new(1)], 1, None);
        drain(&mut node);
        let msg = Message::request_vote_call(Term::new(5), NodeId::new(1), pos(5, 1));
        node.handle_message(&msg).unwrap();
        assert_eq!(node.metrics().term_advances_from_messages, 1);
    }

    #[test]
    fn request_vote_calls_rejected_by_old_term_counter() {
        let mut node = follower(NodeId::new(0), &[NodeId::new(0), NodeId::new(1)], 5, None);
        drain(&mut node);
        let msg = Message::request_vote_call(Term::new(2), NodeId::new(1), pos(2, 5));
        node.handle_message(&msg).unwrap();
        assert_eq!(node.metrics().request_vote_calls_rejected_by_old_term, 1);
        assert_eq!(node.metrics().request_vote_calls_rejected_by_log, 0);
        assert_eq!(
            node.metrics().request_vote_calls_rejected_by_existing_vote,
            0
        );
    }

    #[test]
    fn request_vote_calls_rejected_by_log_counter() {
        let mut node = follower(NodeId::new(0), &[NodeId::new(0), NodeId::new(1)], 1, None);
        drain(&mut node);
        // Follower's log has ClusterConfig at (term=0, index=1); candidate reports (0, 0).
        let msg = Message::request_vote_call(Term::new(1), NodeId::new(1), pos(0, 0));
        node.handle_message(&msg).unwrap();
        assert_eq!(node.metrics().request_vote_calls_rejected_by_log, 1);
        assert_eq!(node.metrics().request_vote_calls_rejected_by_old_term, 0);
    }

    #[test]
    fn request_vote_calls_rejected_by_existing_vote_counter() {
        let mut node = follower(
            NodeId::new(0),
            &[NodeId::new(0), NodeId::new(1), NodeId::new(2)],
            3,
            Some(NodeId::new(1)),
        );
        drain(&mut node);
        // Same term, log check passes (candidate's last_position >= follower's).
        let msg = Message::request_vote_call(Term::new(3), NodeId::new(2), pos(3, 5));
        node.handle_message(&msg).unwrap();
        assert_eq!(
            node.metrics().request_vote_calls_rejected_by_existing_vote,
            1
        );
    }

    #[test]
    fn request_vote_replies_ignored_from_old_terms_counter() {
        let mut node = candidate_with(NodeId::new(0), &[NodeId::new(1), NodeId::new(2)]);
        // Candidate's term after election timeout is 1; reply carries term 0.
        assert_eq!(node.current_term(), Term::new(1));
        let msg = Message::request_vote_reply(Term::new(0), NodeId::new(1), true);
        node.handle_message(&msg).unwrap();
        assert_eq!(
            node.metrics().request_vote_replies_ignored_from_old_terms,
            1
        );
        assert_eq!(
            node.metrics()
                .request_vote_replies_ignored_while_not_candidate,
            0
        );
    }

    #[test]
    fn request_vote_replies_ignored_while_not_candidate_counter() {
        let mut node = follower(NodeId::new(0), &[NodeId::new(0), NodeId::new(1)], 3, None);
        drain(&mut node);
        let msg = Message::request_vote_reply(Term::new(3), NodeId::new(1), true);
        node.handle_message(&msg).unwrap();
        assert_eq!(
            node.metrics()
                .request_vote_replies_ignored_while_not_candidate,
            1
        );
        assert_eq!(
            node.metrics().request_vote_replies_ignored_from_old_terms,
            0
        );
    }

    #[test]
    fn append_entries_calls_rejected_by_old_term_counter() {
        let mut node = follower(NodeId::new(0), &[NodeId::new(0), NodeId::new(1)], 5, None);
        drain(&mut node);
        let call = Message::append_entries_call(
            Term::new(2),
            NodeId::new(1),
            LogIndex::ZERO,
            LogEntries::new(pos(2, 1)),
        );
        node.handle_message(&call).unwrap();
        assert_eq!(node.metrics().append_entries_calls_rejected_by_old_term, 1);
    }

    #[test]
    fn same_term_append_entries_calls_received_by_leader_counter() {
        let mut node = leader_with(NodeId::new(0), &[NodeId::new(1)]);
        // Another node claims to be leader at the same term.
        let call = Message::append_entries_call(
            node.current_term(),
            NodeId::new(1),
            LogIndex::ZERO,
            LogEntries::new(node.log().last_position()),
        );
        node.handle_message(&call).unwrap();
        assert_eq!(
            node.metrics()
                .same_term_append_entries_calls_received_by_leader,
            1
        );
    }

    #[test]
    fn log_divergences_detected_by_follower_counter() {
        let mut node = follower(NodeId::new(0), &[NodeId::new(0), NodeId::new(1)], 2, None);
        drain(&mut node);
        // Follower has ClusterConfig at (term=0, index=1). Send a call with prev at
        // (term=999, index=1) — index exists but term differs.
        let call = Message::append_entries_call(
            Term::new(2),
            NodeId::new(1),
            LogIndex::ZERO,
            LogEntries::from_iter(pos(999, 1), core::iter::once(LogEntry::Command)),
        );
        node.handle_message(&call).unwrap();
        assert_eq!(node.metrics().log_divergences_detected_by_follower, 1);
    }

    #[test]
    fn append_entries_replies_ignored_from_old_terms_counter() {
        let mut node = leader_with(NodeId::new(0), &[NodeId::new(1)]);
        let reply = Message::append_entries_reply(
            Term::new(node.current_term().get() - 1),
            NodeId::new(1),
            node.log().last_position(),
        );
        node.handle_message(&reply).unwrap();
        assert_eq!(
            node.metrics().append_entries_replies_ignored_from_old_terms,
            1
        );
    }

    #[test]
    fn append_entries_replies_ignored_while_not_leader_counter() {
        let mut node = follower(NodeId::new(0), &[NodeId::new(0), NodeId::new(1)], 3, None);
        drain(&mut node);
        let reply = Message::append_entries_reply(Term::new(3), NodeId::new(1), pos(3, 1));
        node.handle_message(&reply).unwrap();
        assert_eq!(
            node.metrics()
                .append_entries_replies_ignored_while_not_leader,
            1
        );
    }

    #[test]
    fn append_entries_replies_ignored_from_unknown_nodes_counter() {
        let mut node = leader_with(NodeId::new(0), &[NodeId::new(1)]);
        let reply = Message::append_entries_reply(
            node.current_term(),
            NodeId::new(9),
            node.log().last_position(),
        );
        node.handle_message(&reply).unwrap();
        assert_eq!(
            node.metrics()
                .append_entries_replies_ignored_from_unknown_nodes,
            1
        );
    }

    #[test]
    fn append_entries_replies_ignored_behind_match_index_counter() {
        let mut node = leader_with(NodeId::new(0), &[NodeId::new(1)]);
        // Manually advance follower.match_index so a smaller last_position looks stale.
        if let RoleState::Leader { followers, .. } = &mut node.role {
            let f = followers.get_mut(&NodeId::new(1)).unwrap();
            f.match_index = LogIndex::new(5);
        }
        let commit_before = node.commit_index();
        let reply = Message::append_entries_reply(
            node.current_term(),
            NodeId::new(1),
            pos(node.current_term().get(), 3),
        );
        node.handle_message(&reply).unwrap();
        assert_eq!(
            node.metrics()
                .append_entries_replies_ignored_behind_match_index,
            1
        );
        // Dropping the delayed reply must not advance the commit index or
        // emit any new action.
        assert_eq!(node.commit_index(), commit_before);
        assert!(node.actions_mut().next().is_none());
    }

    #[test]
    fn append_entries_replies_ahead_of_leader_counter() {
        let mut node = leader_with(NodeId::new(0), &[NodeId::new(1)]);
        let leader_last = node.log().last_position().index.get();
        let reply = Message::append_entries_reply(
            node.current_term(),
            NodeId::new(1),
            pos(node.current_term().get(), leader_last + 10),
        );
        node.handle_message(&reply).unwrap();
        assert_eq!(node.metrics().append_entries_replies_ahead_of_leader, 1);
    }

    #[test]
    fn log_divergences_detected_by_leader_counter() {
        let mut node = leader_with(NodeId::new(0), &[NodeId::new(1)]);
        let last_index = node.log().last_position().index;
        // The leader's log has an entry at last_index; report a different term at the
        // same index so contains() fails but get_term() returns Some.
        let reply = Message::append_entries_reply(
            node.current_term(),
            NodeId::new(1),
            LogPosition {
                term: Term::new(999),
                index: last_index,
            },
        );
        node.handle_message(&reply).unwrap();
        assert_eq!(node.metrics().log_divergences_detected_by_leader, 1);
    }

    #[test]
    fn elections_started_counter() {
        let mut node = follower(NodeId::new(0), &[NodeId::new(0), NodeId::new(1)], 0, None);
        drain(&mut node);
        assert_eq!(node.metrics().elections_started, 0);
        node.handle_election_timeout();
        assert_eq!(node.metrics().elections_started, 1);
    }

    #[test]
    fn elections_started_not_counted_for_non_voter() {
        // Node 9 is not a voter in the cluster {0}.
        let cfg = config(&[NodeId::new(0)]);
        let entries = LogEntries::from_iter(
            LogPosition::ZERO,
            core::iter::once(LogEntry::ClusterConfig(cfg.clone())),
        );
        let mut node = Node::restart(NodeId::new(9), Term::ZERO, None, Log::new(cfg, entries));
        drain(&mut node);
        node.handle_election_timeout();
        assert_eq!(node.metrics().elections_started, 0);
    }

    #[test]
    fn leaderships_started_counter() {
        // Solo voter transitions to leader immediately in create_cluster.
        let mut node = Node::start(NodeId::new(0));
        node.create_cluster(&[NodeId::new(0)]);
        assert!(node.role().is_leader());
        assert_eq!(node.metrics().leaderships_started, 1);
        assert_eq!(node.metrics().elections_started, 1);
    }

    #[test]
    fn counters_persist_through_role_transitions() {
        let mut node = follower(NodeId::new(0), &[NodeId::new(0), NodeId::new(1)], 0, None);
        drain(&mut node);
        // Trigger a term advance while still a follower.
        let msg = Message::request_vote_call(Term::new(5), NodeId::new(1), pos(5, 1));
        node.handle_message(&msg).unwrap();
        assert_eq!(node.metrics().term_advances_from_messages, 1);
        drain(&mut node);
        // Trigger a candidate → follower transition on election timeout followed by a
        // higher-term message.
        node.handle_election_timeout();
        assert!(node.role().is_candidate());
        assert_eq!(node.metrics().elections_started, 1);
        let msg = Message::request_vote_call(Term::new(10), NodeId::new(1), pos(10, 1));
        node.handle_message(&msg).unwrap();
        assert!(node.role().is_follower());
        // Earlier counter values are preserved across role transitions.
        assert_eq!(node.metrics().elections_started, 1);
        assert_eq!(node.metrics().term_advances_from_messages, 2);
    }

    #[test]
    fn counters_saturate_at_max() {
        let mut node = Node::start(NodeId::new(0));
        node.metrics.self_messages_ignored = u64::MAX;
        let msg = Message::request_vote_call(Term::new(1), NodeId::new(0), pos(0, 0));
        node.handle_message(&msg).unwrap();
        assert_eq!(node.metrics().self_messages_ignored, u64::MAX);
    }

    #[test]
    fn follower_match_index_returns_none_for_non_leader_roles() {
        let f = follower(NodeId::new(0), &[NodeId::new(0), NodeId::new(1)], 1, None);
        assert_eq!(f.follower_match_index(NodeId::new(0)), None);
        assert_eq!(f.follower_match_index(NodeId::new(1)), None);

        let c = candidate_with(NodeId::new(0), &[NodeId::new(1)]);
        assert!(c.role().is_candidate());
        assert_eq!(c.follower_match_index(NodeId::new(0)), None);
        assert_eq!(c.follower_match_index(NodeId::new(1)), None);
    }

    #[test]
    fn follower_match_index_returns_none_for_unknown_and_self_on_leader() {
        let leader = leader_with(NodeId::new(0), &[NodeId::new(1)]);
        assert_eq!(leader.follower_match_index(NodeId::new(0)), None);
        assert_eq!(leader.follower_match_index(NodeId::new(99)), None);
    }

    #[test]
    fn follower_match_index_starts_at_zero_for_tracked_voters() {
        let leader = leader_with(NodeId::new(0), &[NodeId::new(1), NodeId::new(2)]);
        assert_eq!(
            leader.follower_match_index(NodeId::new(1)),
            Some(LogIndex::new(0))
        );
        assert_eq!(
            leader.follower_match_index(NodeId::new(2)),
            Some(LogIndex::new(0))
        );
    }

    #[test]
    fn follower_match_index_starts_at_zero_for_non_voter_and_new_voter() {
        // Solo-voter leader so config changes commit immediately.
        let mut leader = leader_with(NodeId::new(0), &[]);
        drain(&mut leader);

        // Non-voter is tracked as soon as the config change is applied.
        let mut cfg = leader.config().clone();
        cfg.non_voters.insert(NodeId::new(1));
        assert_ne!(leader.propose_config(cfg), LogPosition::INVALID);
        drain(&mut leader);
        assert_eq!(
            leader.follower_match_index(NodeId::new(1)),
            Some(LogIndex::new(0))
        );

        // A new_voters-only node in a joint consensus is also tracked.
        let joint = leader.config().to_joint_consensus(&[NodeId::new(2)], &[]);
        assert_ne!(leader.propose_config(joint), LogPosition::INVALID);
        drain(&mut leader);
        assert_eq!(
            leader.follower_match_index(NodeId::new(2)),
            Some(LogIndex::new(0))
        );
    }

    #[test]
    fn follower_match_index_advances_on_successful_reply() {
        let mut leader = leader_with(NodeId::new(0), &[NodeId::new(1)]);
        let follower_id = NodeId::new(1);
        assert_eq!(
            leader.follower_match_index(follower_id),
            Some(LogIndex::new(0))
        );
        let last = leader.log().last_position();
        let reply = Message::append_entries_reply(leader.current_term(), follower_id, last);
        leader.handle_message(&reply).unwrap();
        assert_eq!(leader.follower_match_index(follower_id), Some(last.index));
    }

    #[test]
    fn follower_match_index_does_not_regress_on_delayed_reply() {
        let mut leader = leader_with(NodeId::new(0), &[NodeId::new(1)]);
        // Manually advance follower.match_index so a smaller last_position looks stale.
        if let RoleState::Leader { followers, .. } = &mut leader.role {
            let f = followers.get_mut(&NodeId::new(1)).unwrap();
            f.match_index = LogIndex::new(5);
        }
        assert_eq!(
            leader.follower_match_index(NodeId::new(1)),
            Some(LogIndex::new(5))
        );
        let reply = Message::append_entries_reply(
            leader.current_term(),
            NodeId::new(1),
            pos(leader.current_term().get(), 3),
        );
        leader.handle_message(&reply).unwrap();
        assert_eq!(
            leader.follower_match_index(NodeId::new(1)),
            Some(LogIndex::new(5))
        );
    }

    #[test]
    fn follower_match_index_returns_none_after_removed_from_config() {
        // Solo-voter leader so config changes commit immediately.
        let mut leader = leader_with(NodeId::new(0), &[]);
        drain(&mut leader);

        let mut cfg = leader.config().clone();
        cfg.non_voters.insert(NodeId::new(1));
        assert_ne!(leader.propose_config(cfg), LogPosition::INVALID);
        drain(&mut leader);
        assert!(leader.follower_match_index(NodeId::new(1)).is_some());

        let mut cfg = leader.config().clone();
        cfg.non_voters.clear();
        assert_ne!(leader.propose_config(cfg), LogPosition::INVALID);
        drain(&mut leader);
        assert_eq!(leader.follower_match_index(NodeId::new(1)), None);
    }

    #[test]
    fn follower_match_index_forgets_previous_values_after_stepdown() {
        let mut leader = leader_with(NodeId::new(0), &[NodeId::new(1)]);
        let follower_id = NodeId::new(1);

        let last = leader.log().last_position();
        let reply = Message::append_entries_reply(leader.current_term(), follower_id, last);
        leader.handle_message(&reply).unwrap();
        assert_eq!(leader.follower_match_index(follower_id), Some(last.index));

        // Step down via a higher-term RequestVote.
        let higher = Term::new(leader.current_term().get() + 10);
        let msg = Message::request_vote_call(higher, follower_id, pos(higher.get(), 1));
        leader.handle_message(&msg).unwrap();
        assert!(leader.role().is_follower());
        assert_eq!(leader.follower_match_index(follower_id), None);
    }
}
