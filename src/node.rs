use crate::{
    CommitStatus, Log, Role, Term,
    action::{Action, Actions},
    config::ClusterConfig,
    log::{LogEntries, LogEntry, LogIndex, LogPosition},
    message::Message,
    quorum::Quorum,
};
use std::collections::{BTreeMap, BTreeSet};

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

impl std::ops::Add for NodeId {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.0 + rhs.0)
    }
}

impl std::ops::AddAssign for NodeId {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl std::ops::Sub for NodeId {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.0 - rhs.0)
    }
}

impl std::ops::SubAssign for NodeId {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

/// Node generation ([`u64`]).
///
/// The crate user is responsible for supplying a unique generation on restart.
/// The generation should be monotonically increasing for the same node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeGeneration(u64);

impl NodeGeneration {
    /// The initial generation.
    pub const ZERO: Self = Self(0);

    /// Makes a new [`NodeGeneration`] instance.
    pub const fn new(generation: u64) -> Self {
        Self(generation)
    }

    /// Returns the value of this generation.
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl From<u64> for NodeGeneration {
    fn from(value: u64) -> Self {
        Self::new(value)
    }
}

impl From<NodeGeneration> for u64 {
    fn from(value: NodeGeneration) -> Self {
        value.get()
    }
}

/// Raft node.
#[derive(Debug, Clone)]
pub struct Node {
    id: NodeId,
    generation: NodeGeneration,
    voted_for: Option<NodeId>,
    current_term: Term,
    log: Log,
    commit_index: LogIndex,
    actions: Actions,
    role: RoleState,
}

impl Node {
    /// Starts a new node.
    ///
    /// To create a new cluster, please call [`Node::create_cluster()`] after starting the node.
    ///
    /// If the node has already been part of a cluster, please use [`Node::restart()`] instead.
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
        Self::new(id, NodeGeneration::ZERO)
    }

    /// Restarts a node.
    ///
    /// `current_term`, `voted_for`, and `log` are restored from persistent storage.
    /// The generation must be a value that is unique across restarts of the same node,
    /// and should be monotonically increasing.
    /// It can be derived from persistent storage or an external source such as a system clock,
    /// as long as it satisfies the uniqueness and monotonicity requirements.
    /// Note that managing the persistent storage is outside the scope of this crate.
    ///
    /// # Notes
    ///
    /// Raft algorithm assumes the persistent storage is reliable.
    /// So, for example, if the local log of the node has corrupted or lost some log tail entries,
    /// it is safest to remove the node from the cluster then add it back as a new node.
    ///
    /// In practice, such storage failures are usually tolerable when the majority of nodes in the cluster
    /// are healthy (i.e., the restarted node can restored its previous state).
    ///
    /// But be careful, whether the degraded safety guarantee is acceptable or not highly depends on
    /// the application.
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
    /// let generation = noraft::NodeGeneration::new(1);
    /// let node = noraft::Node::restart(noraft::NodeId::new(0), generation, current_term, voted_for, log);
    /// assert!(node.role().is_follower());
    /// assert_eq!(node.commit_index(), snapshot_index);
    ///
    /// // Unlike `Node::start()`, the restarted node has actions to execute.
    /// assert!(!node.actions().is_empty());
    /// ```
    pub fn restart(
        id: NodeId,
        generation: NodeGeneration,
        current_term: Term,
        voted_for: Option<NodeId>,
        log: Log,
    ) -> Self {
        let mut node = Self::new(id, generation);

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
                std::iter::once(entry.clone()),
            )));
        self.log.entries_mut().push(entry.clone());

        self.transition_to_candidate();

        self.log.last_position()
    }

    fn new(id: NodeId, generation: NodeGeneration) -> Self {
        let config = ClusterConfig::new();
        Self {
            id,
            generation,
            voted_for: None,
            current_term: Term::ZERO,
            log: Log::new(config, LogEntries::new(LogPosition::ZERO)),
            commit_index: LogIndex::ZERO,
            actions: Actions::default(),
            role: RoleState::Follower,
        }
    }

    /// Returns the identifier of this node.
    pub fn id(&self) -> NodeId {
        self.id
    }

    /// Returns the generation of this node.
    pub fn generation(&self) -> NodeGeneration {
        self.generation
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

    fn transition_to_leader(&mut self) {
        debug_assert_eq!(self.voted_for, Some(self.id));

        let quorum = Quorum::new(self.config());
        let followers = BTreeMap::new();
        let solo_voter =
            self.config().unique_voters().count() == 1 && self.config().voters.contains(&self.id);
        self.role = RoleState::Leader {
            followers,
            quorum,
            solo_voter,
        };
        self.rebuild_followers();
        self.rebuild_quorum();

        self.propose(LogEntry::Term(self.current_term));
    }

    fn transition_to_candidate(&mut self) {
        if !self.log.latest_config().is_voter(self.id) {
            // Non voter or removed node cannot become a candidate.
            return;
        }

        self.set_current_term(self.current_term.next());
        self.set_voted_for(Some(self.id));

        let solo_voter =
            self.config().unique_voters().count() == 1 && self.config().voters.contains(&self.id);
        if solo_voter {
            self.transition_to_leader();
            return;
        }

        self.role = RoleState::Candidate {
            granted_votes: std::iter::once(self.id).collect(),
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
                LogEntries::from_iter(old_last_position, std::iter::once(entry)),
            );
            self.actions.set(Action::BroadcastMessage(call));
        }
        self.actions.set(Action::SetElectionTimeout);

        self.log.last_position()
    }

    fn rebuild_followers(&mut self) {
        let RoleState::Leader { followers, .. } = &mut self.role else {
            unreachable!();
        };

        let config = self.log.latest_config();

        // Add new followers.
        for id in config.unique_nodes() {
            if id == self.id || followers.contains_key(&id) {
                continue;
            }
            followers.insert(id, Follower::new());
        }

        // Remove followers not in the latest configuration.
        followers.retain(|id, _| config.contains(*id));
    }

    fn rebuild_quorum(&mut self) {
        let RoleState::Leader {
            quorum, followers, ..
        } = &mut self.role
        else {
            unreachable!();
        };

        let config = self.log.latest_config();
        Self::rebuild_quorum_inner(
            quorum,
            followers,
            config,
            self.id,
            self.log.last_position().index,
        );
    }

    fn rebuild_quorum_inner(
        quorum: &mut Quorum,
        followers: &BTreeMap<NodeId, Follower>,
        config: &ClusterConfig,
        self_id: NodeId,
        self_last: LogIndex,
    ) {
        *quorum = Quorum::new(config);

        quorum.update_match_index(config, self_id, LogIndex::ZERO, self_last);

        for (&id, follower) in followers {
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
        new_config.voters = std::mem::take(&mut new_config.new_voters);
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
                std::iter::once(entry.clone()),
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

    fn append_log_entries_from_leader(&mut self, entries: &LogEntries) -> bool {
        debug_assert!(self.role().is_follower());

        if self.log.entries().contains(entries.last_position()) {
            // Already up-to-date.
            return self.log().last_position() == entries.last_position();
        }
        if !self.log.entries().contains(entries.prev_position()) {
            // Cannot append.
            if self
                .log
                .entries()
                .contains_index(entries.prev_position().index)
            {
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
                    // The local snapshot does not match the leader's log.
                    // Such a situation should never occur if the Raft properties are satisfied.
                    // Although this is very unusual, we will reset the log and request the leader's snapshot.
                    self.log = Log::new(ClusterConfig::new(), LogEntries::new(LogPosition::ZERO));
                }
            }
            return false;
        }

        // Append.
        let entries = entries.strip_common_prefix(self.log.entries());
        self.log.entries_mut().append(&entries);
        self.actions.set(Action::AppendLogEntries(entries));

        true
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
    /// # Examples
    ///
    /// ```
    /// let mut node = /* ... ; */
    /// # noraft::Node::start(noraft::NodeId::new(1));
    ///
    /// let msg = /* ... ; */
    /// # noraft::Message::RequestVoteReply { from: noraft::NodeId::new(1), term: noraft::Term::new(1), vote_granted: true };
    /// node.handle_message(&msg);
    ///
    /// // Execute actions queued by the message handling.
    /// for action in node.actions_mut() {
    ///     // ...
    /// }
    /// ```
    pub fn handle_message(&mut self, msg: &Message) {
        if msg.from() == self.id {
            return;
        }
        if self.current_term < msg.term() {
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
            } => self.handle_append_entries_call(*from, *term, *commit_index, entries),
            Message::AppendEntriesReply {
                from,
                term,
                generation,
                last_position,
            } => self.handle_append_entries_reply(*from, *term, *generation, *last_position),
        }
    }

    fn handle_request_vote_call(&mut self, from: NodeId, term: Term, last_position: LogPosition) {
        if term < self.current_term {
            // Needs to reply to update the sender's term.
            let reply = Message::request_vote_reply(self.current_term, self.id, false);
            self.actions.set(Action::SendMessage(from, reply));
            return;
        }

        if self.log.last_position() > last_position {
            return;
        }

        if self.voted_for.is_none() {
            self.set_voted_for(Some(from));
        }

        if self.voted_for != Some(from) {
            // This node is either a candidate, a leader, or has already voted for another node.
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
            return;
        };
        if !vote_granted {
            return;
        }
        if term < self.current_term {
            // Delayed (obsolete) reply from an old term.
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
    ) {
        if term < self.current_term {
            // Needs to reply to update the sender's term.
            self.reply_append_entries(from);
            return;
        }

        if !self.role().is_follower() {
            return;
        }

        if self.voted_for.is_none() {
            self.set_voted_for(Some(from));
        }

        if self.voted_for != Some(from) {
            return;
        }

        let no_divergence = self.append_log_entries_from_leader(entries);
        if no_divergence {
            let next_commit_index = leader_commit.min(self.log.last_position().index);
            if self.commit_index < next_commit_index {
                self.commit_index = next_commit_index;
            }
        }

        self.reply_append_entries(from);
        self.actions.set(Action::SetElectionTimeout);
    }

    fn handle_append_entries_reply(
        &mut self,
        from: NodeId,
        term: Term,
        generation: NodeGeneration,
        follower_last_position: LogPosition,
    ) {
        if term < self.current_term {
            // Delayed (obsolete) reply from an old term.
            return;
        }

        let RoleState::Leader {
            followers, quorum, ..
        } = &mut self.role
        else {
            return;
        };

        let Some(follower) = followers.get_mut(&from) else {
            // Replies from unknown nodes are ignored.
            return;
        };

        if generation < follower.generation
            || (generation == follower.generation
                && follower_last_position.index < follower.match_index)
        {
            // Delayed reply.
            return;
        }

        let mut should_rebuild_quorum = false;
        if generation > follower.generation {
            follower.generation = generation;
            if follower_last_position.index < follower.match_index {
                follower.match_index = follower_last_position.index;
                should_rebuild_quorum = true;
            }
        }

        if should_rebuild_quorum {
            // Generation update can decrease match_index, and `Quorum::update_match_index()`
            // only supports monotonic increases, so a full rebuild is required.
            //
            // [NOTE]
            // The Raft algorithm assumes that storage is reliable.
            // Therefore, theoretically, this case should not occur.
            // However, in practice, it is possible for the follower's log to be fully or partially lost.
            // To recover log entries as much as possible in such cases,
            // this crate allows proceeding even if the above condition is met.
            // But be cautious, as there is a risk of compromising the properties guaranteed by
            // the Raft algorithm.
            Self::rebuild_quorum_inner(
                quorum,
                followers,
                self.log.latest_config(),
                self.id,
                self.log.last_position().index,
            );
        }

        let follower = followers.get_mut(&from).expect("infallible");

        if !self.log.entries().contains(follower_last_position) {
            if let Some(term) = self.log.entries().get_term(follower_last_position.index) {
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
        let reply = Message::append_entries_reply(
            self.current_term,
            self.id,
            self.generation,
            self.log.last_position(),
        );
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
    pub generation: NodeGeneration,
}

impl Follower {
    pub fn new() -> Self {
        Self {
            match_index: LogIndex::new(0),
            generation: NodeGeneration::ZERO,
        }
    }
}
