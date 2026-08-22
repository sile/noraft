use std::ops::{Deref, DerefMut};

macro_rules! assert_no_action {
    ($node:expr) => {
        assert_eq!($node.actions_mut().next(), None);
        assert!($node.actions().is_empty());
    };
}

macro_rules! assert_action {
    ($node:expr, $action:expr) => {
        let action = $action;
        assert_eq!(
            next_same_kind_action($node.actions_mut(), &action),
            Some(action)
        );
    };
}

#[test]
fn single_node_start() {
    TestNode::asserted_start(id(0), &[id(0)]);
}

#[test]
fn solo_voter_with_non_voter_yields_log_append_before_committed_broadcast() {
    let mut leader = TestNode::asserted_start(id(0), &[id(0)]);
    let mut config = leader.config().clone();
    config.non_voters.insert(id(1));

    let position = leader.propose_config(config);

    assert_ne!(position, noraft::LogPosition::INVALID);
    assert_eq!(leader.commit_index(), position.index);

    let actions: Vec<_> = leader.actions_mut().collect();
    let [
        noraft::Action::SetElectionTimeout,
        noraft::Action::AppendLogEntries(log_entries),
        noraft::Action::BroadcastMessage(noraft::Message::AppendEntriesCall {
            commit_index,
            entries,
            ..
        }),
    ] = actions.as_slice()
    else {
        panic!("unexpected actions: {actions:?}");
    };
    assert_eq!(log_entries.last_position(), position);
    assert_eq!(*commit_index, position.index);
    assert_eq!(entries.last_position(), position);
}

#[test]
fn snapshot_preserves_pending_append_suffix_after_snapshot_position() {
    let mut leader = TestNode::asserted_start(id(0), &[id(0)]);

    let first_position = leader.propose_command();
    let second_position = leader.propose_command();
    assert_eq!(leader.commit_index(), second_position.index);
    assert_eq!(
        leader.actions().append_log_entries.as_ref(),
        Some(&noraft::LogEntries::from_iter(
            log_prev(first_position),
            [noraft::LogEntry::Command, noraft::LogEntry::Command]
        ))
    );

    let snapshot_config = leader.config().clone();
    assert!(leader.handle_snapshot_installed(first_position, snapshot_config));

    // The leader was already committed past `first_position`, so
    // `handle_snapshot_installed` must not regress `commit_index`.
    assert_eq!(leader.commit_index(), second_position.index);
    assert_eq!(leader.log().entries().prev_position(), first_position);
    assert_eq!(leader.log().entries().last_position(), second_position);
    assert_eq!(
        leader.actions().append_log_entries.as_ref(),
        Some(&noraft::LogEntries::from_iter(
            first_position,
            [noraft::LogEntry::Command]
        ))
    );
}

#[test]
fn snapshot_discards_incompatible_pending_append_entries() {
    // `handle_snapshot_installed` rejects snapshots whose term exceeds
    // `current_term`, so mimic the caller precondition and restart the
    // node at the snapshot term.
    let mut follower = noraft::Node::restart(
        id(0),
        t(12),
        None,
        noraft::Log::new(
            noraft::ClusterConfig::new(),
            noraft::LogEntries::new(noraft::LogPosition::ZERO),
        ),
    );
    while follower.actions_mut().next().is_some() {}
    follower.actions_mut().append_log_entries = Some(noraft::LogEntries::from_iter(
        log_pos(t(10), i(2)),
        [noraft::LogEntry::Command],
    ));

    let snapshot_position = log_pos(t(12), i(2));
    assert!(follower.handle_snapshot_installed(snapshot_position, noraft::ClusterConfig::new()));

    // A fresh follower starts at `commit_index == 0`; the install must
    // advance it to the snapshot boundary.
    assert_eq!(follower.commit_index(), snapshot_position.index);
    assert_eq!(follower.log().entries().prev_position(), snapshot_position);
    assert!(follower.actions().append_log_entries.is_none());
}

#[test]
fn local_snapshot_mismatch_returns_error() {
    fn assert_error_trait<E: core::error::Error>() {}
    assert_error_trait::<noraft::Error>();
    assert_eq!(
        noraft::Error::LocalSnapshotMismatch.to_string(),
        "local snapshot position conflicts with leader log"
    );

    let snapshot_position = log_pos(t(1), i(5));
    let log = noraft::Log::new(
        noraft::ClusterConfig::new(),
        noraft::LogEntries::from_iter(snapshot_position, [noraft::LogEntry::Command]),
    );
    let mut follower = noraft::Node::restart(id(0), t(2), Some(id(1)), log);
    assert_action!(follower, set_election_timeout());
    assert_no_action!(follower);

    let original_log = follower.log().clone();
    let call = noraft::Message::AppendEntriesCall {
        from: id(1),
        term: t(2),
        commit_index: i(5),
        entries: noraft::LogEntries::from_iter(log_pos(t(2), i(5)), [noraft::LogEntry::Command]),
    };

    assert_eq!(
        follower.handle_message(&call),
        Err(noraft::Error::LocalSnapshotMismatch)
    );
    assert_eq!(follower.log(), &original_log);
    assert!(follower.actions().is_empty());
}

#[test]
fn create_two_nodes_cluster() {
    let initial_voters = [id(0), id(1)];
    let mut node0 = TestNode::asserted_start(id(0), &initial_voters);
    let mut node1 = TestNode::asserted_start(id(1), &[]);

    // Setup cluster.
    node0.handle_election_timeout();
    assert_eq!(node0.role(), noraft::Role::Candidate);
    assert_action!(node0, set_election_timeout());
    assert_action!(node0, save_current_term());
    assert_action!(node0, save_voted_for());

    let Some(noraft::Action::BroadcastMessage(call @ noraft::Message::RequestVoteCall { .. })) =
        node0.actions_mut().next()
    else {
        panic!("Expected RequestVoteCall message");
    };
    assert_no_action!(node0);

    let reply = node1.asserted_handle_request_vote_call_success(&call);
    let call = node0.asserted_handle_request_vote_reply_majority_vote_granted(&reply);
    let reply = node1.asserted_handle_append_entries_call_failure(&call);
    let call = node0.asserted_handle_append_entries_reply_failure(&reply);

    assert!(!node0.config().is_joint_consensus());
    assert_eq!(node0.config().voters, initial_voters.into_iter().collect());

    assert_eq!(node1.config().unique_nodes().count(), 0);

    let reply = node1.asserted_handle_append_entries_call_success(&call);
    node0.asserted_handle_append_entries_reply_success(&reply, true, false);
    assert_eq!(node0.config(), node1.config());
}

#[test]
fn leader_commits_after_shrinking_to_self_only_voter() {
    let initial_voters = [id(0), id(1)];
    let mut leader = TestNode::asserted_start(id(0), &initial_voters);
    let mut follower = TestNode::asserted_start(id(1), &[]);

    leader.handle_election_timeout();
    assert_eq!(leader.role(), noraft::Role::Candidate);
    assert_action!(leader, set_election_timeout());
    assert_action!(leader, save_current_term());
    assert_action!(leader, save_voted_for());

    let Some(noraft::Action::BroadcastMessage(call @ noraft::Message::RequestVoteCall { .. })) =
        leader.actions_mut().next()
    else {
        panic!("Expected RequestVoteCall message");
    };
    assert_no_action!(leader);

    let reply = follower.asserted_handle_request_vote_call_success(&call);
    let call = leader.asserted_handle_request_vote_reply_majority_vote_granted(&reply);
    let reply = follower.asserted_handle_append_entries_call_failure(&call);
    let call = leader.asserted_handle_append_entries_reply_failure(&reply);
    let reply = follower.asserted_handle_append_entries_call_success(&call);
    leader.asserted_handle_append_entries_reply_success(&reply, true, false);
    assert_eq!(leader.config(), follower.config());

    let joint_config = leader.config().to_joint_consensus(&[], &[follower.id()]);
    let call = leader.asserted_change_cluster_config(joint_config);
    let reply = follower.asserted_handle_append_entries_call_success(&call);

    let final_config_prev_position = leader.log().last_position();
    leader
        .handle_message(&reply)
        .expect("message handling should succeed");

    assert_eq!(leader.config().voters, [leader.id()].into_iter().collect());
    assert!(!leader.config().is_joint_consensus());
    assert_eq!(leader.commit_index(), leader.log().last_position().index);
    assert_action!(
        leader,
        append_log_entry(
            final_config_prev_position,
            cluster_config_entry(leader.config().clone())
        )
    );
    assert_action!(leader, set_election_timeout());
    assert_no_action!(leader);

    let position = leader.propose_command();
    assert_eq!(leader.commit_index(), position.index);
    assert_action!(
        leader,
        append_log_entry(log_prev(position), noraft::LogEntry::Command)
    );
    assert_action!(leader, set_election_timeout());
    assert_no_action!(leader);
}

#[test]
fn create_three_nodes_cluster() {
    let mut cluster = ThreeNodeCluster::new();
    cluster.init_cluster();

    assert!(!cluster.node0.config().is_joint_consensus());
    assert_eq!(cluster.node0.config(), cluster.node1.config());
    assert_eq!(cluster.node0.config(), cluster.node2.config());
}

#[test]
fn self_request_vote_call_is_ignored() {
    let mut node = TestNode::asserted_start(id(0), &[id(0), id(1)]);
    assert_eq!(node.role(), noraft::Role::Candidate);

    let prev_term = node.current_term();
    let prev_voted_for = node.voted_for();
    let prev_role = node.role();
    let msg = request_vote_call(
        node.current_term(),
        node.id(),
        node.log().entries().last_position(),
    );

    node.handle_message(&msg)
        .expect("message handling should succeed");

    assert_eq!(node.current_term(), prev_term);
    assert_eq!(node.voted_for(), prev_voted_for);
    assert_eq!(node.role(), prev_role);
    assert_no_action!(node);
}

#[test]
fn could_be_disruptive_request_vote_true_when_high_term_request_vote_conflicts_with_voted_for() {
    let base = noraft::Node::start(id(0));
    let mut node = noraft::Node::restart(id(0), t(2), Some(id(1)), base.log().clone());
    assert_action!(node, set_election_timeout());
    assert_no_action!(node);

    let msg = request_vote_call(t(3), id(2), node.log().entries().last_position());
    assert!(node.could_be_disruptive_request_vote(&msg));
    assert_no_action!(node);
}

#[test]
fn could_be_disruptive_request_vote_false_for_candidate() {
    let node = TestNode::asserted_start(id(0), &[id(0), id(1), id(2)]);
    assert_eq!(node.role(), noraft::Role::Candidate);

    let msg = request_vote_call(
        next_term(node.current_term()),
        id(2),
        node.log().entries().last_position(),
    );
    assert!(!node.could_be_disruptive_request_vote(&msg));
}

#[test]
fn could_be_disruptive_request_vote_false_for_non_request_vote() {
    let base = noraft::Node::start(id(0));
    let mut node = noraft::Node::restart(id(0), t(2), Some(id(1)), base.log().clone());
    assert_action!(node, set_election_timeout());
    assert_no_action!(node);

    let msg = noraft::Message::AppendEntriesCall {
        from: id(2),
        term: t(3),
        commit_index: node.commit_index(),
        entries: noraft::LogEntries::new(node.log().entries().last_position()),
    };
    assert!(!node.could_be_disruptive_request_vote(&msg));
    assert_no_action!(node);
}

#[test]
fn disruptive_request_vote_is_processed_without_prefilter() {
    let base = noraft::Node::start(id(0));
    let mut node = noraft::Node::restart(id(0), t(2), Some(id(1)), base.log().clone());
    assert_action!(node, set_election_timeout());
    assert_no_action!(node);

    let msg = request_vote_call(t(3), id(2), node.log().entries().last_position());
    assert!(node.could_be_disruptive_request_vote(&msg));

    node.handle_message(&msg)
        .expect("message handling should succeed");

    assert_eq!(node.role(), noraft::Role::Follower);
    assert_eq!(node.current_term(), t(3));
    assert_eq!(node.voted_for(), Some(id(2)));
    let actions: Vec<_> = node.actions_mut().collect();
    assert!(
        actions
            .iter()
            .any(|a| matches!(a, noraft::Action::SaveCurrentTerm))
    );
    assert!(
        actions
            .iter()
            .any(|a| matches!(a, noraft::Action::SaveVotedFor))
    );
    assert!(
        actions
            .iter()
            .any(|a| matches!(a, noraft::Action::SetElectionTimeout))
    );
    assert!(actions.iter().any(|a| matches!(
        a,
        noraft::Action::SendMessage(
            destination,
            noraft::Message::RequestVoteReply {
                term,
                vote_granted: true,
                ..
            }
        ) if *destination == id(2) && *term == t(3)
    )));
}

#[test]
fn high_term_candidate_with_stale_log_is_rejected() {
    // Follower's log ends at (term=5, index=4): committed term-5 entries.
    // Candidate has a higher current term (7) but its last log is (term=3, index=100) —
    // longer by index but staler by term. Per Raft §5.4.1 the follower must reject.
    let entries = noraft::LogEntries::from_iter(
        noraft::LogPosition::ZERO,
        [
            noraft::LogEntry::Term(t(1)),
            noraft::LogEntry::Command,
            noraft::LogEntry::Term(t(5)),
            noraft::LogEntry::Command,
        ],
    );
    assert_eq!(entries.last_position(), log_pos(t(5), i(4)));

    let log = noraft::Log::new(noraft::ClusterConfig::new(), entries);
    let mut node = noraft::Node::restart(id(0), t(5), None, log);
    assert_action!(node, set_election_timeout());
    assert_no_action!(node);

    let stale_but_long = log_pos(t(3), i(100));
    let msg = request_vote_call(t(7), id(1), stale_but_long);

    node.handle_message(&msg)
        .expect("message handling should succeed");

    // noraft::Term must still bump (high term always wins that race), but the candidate
    // must NOT receive a vote because its log is staler by term.
    assert_eq!(node.current_term(), t(7));
    assert_eq!(node.role(), noraft::Role::Follower);
    assert_eq!(node.voted_for(), None);
    let sent_vote = node.actions_mut().any(|a| {
        matches!(
            a,
            noraft::Action::SendMessage(
                _,
                noraft::Message::RequestVoteReply {
                    vote_granted: true,
                    ..
                }
            ),
        )
    });
    assert!(
        !sent_vote,
        "follower must not vote for a stale-log candidate"
    );
}

#[test]
fn election() {
    let mut cluster = ThreeNodeCluster::new();
    cluster.init_cluster();

    // Trigger a new election.
    let _call = cluster.node1.asserted_follower_election_timeout();
    let _call = cluster.node2.asserted_follower_election_timeout();
    let call = cluster.node1.asserted_candidate_election_timeout();

    let reply = cluster
        .node2
        .asserted_handle_request_vote_call_success(&call);

    let call = cluster
        .node1
        .asserted_handle_request_vote_reply_majority_vote_granted(&reply);
    let reply_from_node2 = cluster
        .node2
        .asserted_handle_append_entries_call_success(&call);
    let reply_from_node0 = cluster
        .node0
        .asserted_handle_append_entries_call_success_new_leader(&call);

    cluster
        .node1
        .asserted_handle_append_entries_reply_success(&reply_from_node0, true, false);
    cluster
        .node1
        .asserted_handle_append_entries_reply_success(&reply_from_node2, false, false);

    // Manual heartbeat.
    let call = cluster.node1.asserted_heartbeat();
    let reply = cluster
        .node0
        .asserted_handle_append_entries_call_success(&call);
    cluster
        .node1
        .handle_message(&reply)
        .expect("message handling should succeed");
    assert_no_action!(cluster.node1);

    // Periodic heartbeat.
    cluster.node1.handle_election_timeout();
    let call = append_entries_call(
        &cluster.node1,
        noraft::LogEntries::new(cluster.node1.log().entries().last_position()),
    );
    assert_action!(cluster.node1, set_election_timeout());
    assert_action!(cluster.node1, broadcast_message(&call));

    let reply = cluster
        .node2
        .asserted_handle_append_entries_call_success(&call);
    cluster
        .node1
        .handle_message(&reply)
        .expect("message handling should succeed");
    assert_no_action!(cluster.node1);
}

#[test]
fn candidate_accepts_same_term_append_entries_from_leader() {
    let voters = [id(0), id(1), id(2)];
    let config = joint(&voters, &[]);
    let prefix = noraft::LogEntries::from_iter(
        noraft::LogPosition::ZERO,
        [
            cluster_config_entry(config),
            term_entry(t(2)),
            noraft::LogEntry::Command,
        ],
    );

    let mut leader = TestNode {
        inner: noraft::Node::restart(
            id(0),
            t(3),
            None,
            noraft::Log::new(noraft::ClusterConfig::new(), prefix.clone()),
        ),
        actions: noraft::Actions::default(),
    };
    assert_action!(leader, set_election_timeout());
    assert_no_action!(leader);

    let _call = leader.asserted_follower_election_timeout();
    let vote_reply = request_vote_reply(leader.current_term(), id(2), true);
    let append_entries =
        leader.asserted_handle_request_vote_reply_majority_vote_granted(&vote_reply);
    assert_eq!(leader.role(), noraft::Role::Leader);

    let mut candidate = TestNode {
        inner: noraft::Node::restart(
            id(1),
            t(3),
            None,
            noraft::Log::new(noraft::ClusterConfig::new(), prefix),
        ),
        actions: noraft::Actions::default(),
    };
    assert_action!(candidate, set_election_timeout());
    assert_no_action!(candidate);

    let _call = candidate.asserted_follower_election_timeout();
    assert_eq!(candidate.role(), noraft::Role::Candidate);
    assert_eq!(candidate.current_term(), leader.current_term());

    let _reply = candidate.asserted_handle_append_entries_call_success(&append_entries);

    assert_eq!(candidate.role(), noraft::Role::Follower);
    assert_eq!(candidate.voted_for(), Some(leader.id()));
    assert_eq!(
        candidate.log().last_position(),
        leader.log().last_position()
    );
}

#[test]
fn follower_accepts_same_term_append_entries_after_voting_for_another_candidate() {
    let voters = [id(0), id(1), id(2), id(3), id(4)];
    let config = joint(&voters, &[]);
    let prefix = noraft::LogEntries::from_iter(
        noraft::LogPosition::ZERO,
        [
            cluster_config_entry(config),
            term_entry(t(2)),
            noraft::LogEntry::Command,
        ],
    );

    let mut leader = TestNode {
        inner: noraft::Node::restart(
            id(0),
            t(3),
            None,
            noraft::Log::new(noraft::ClusterConfig::new(), prefix.clone()),
        ),
        actions: noraft::Actions::default(),
    };
    assert_action!(leader, set_election_timeout());
    assert_no_action!(leader);

    let _call = leader.asserted_follower_election_timeout();
    let first_vote_reply = request_vote_reply(leader.current_term(), id(1), true);
    leader
        .handle_message(&first_vote_reply)
        .expect("message handling should succeed");
    assert_eq!(leader.role(), noraft::Role::Candidate);
    assert_no_action!(leader);

    let vote_reply = request_vote_reply(leader.current_term(), id(2), true);
    let append_entries =
        leader.asserted_handle_request_vote_reply_majority_vote_granted(&vote_reply);
    assert_eq!(leader.role(), noraft::Role::Leader);

    let mut follower = TestNode {
        inner: noraft::Node::restart(
            id(3),
            leader.current_term(),
            Some(id(4)),
            noraft::Log::new(noraft::ClusterConfig::new(), prefix),
        ),
        actions: noraft::Actions::default(),
    };
    assert_action!(follower, set_election_timeout());
    assert_no_action!(follower);
    assert_eq!(follower.role(), noraft::Role::Follower);
    assert_ne!(follower.voted_for(), Some(leader.id()));

    let _reply = follower.asserted_handle_append_entries_call_success(&append_entries);

    assert_eq!(follower.role(), noraft::Role::Follower);
    assert_eq!(follower.voted_for(), Some(leader.id()));
    assert_eq!(follower.log().last_position(), leader.log().last_position());
}

#[test]
fn restart() {
    let mut cluster = ThreeNodeCluster::new();
    cluster.init_cluster();
    cluster.propose_command();

    // Restart node1.
    assert_eq!(cluster.node1.role(), noraft::Role::Follower);
    cluster.node1.inner = noraft::Node::restart(
        cluster.node1.id(),
        cluster.node1.current_term(),
        cluster.node1.voted_for(),
        cluster.node1.log().clone(),
    );

    cluster.propose_command();
}

#[test]
fn truncate_log() {
    let mut cluster = ThreeNodeCluster::new();
    cluster.init_cluster();
    cluster.propose_command();

    // Propose a command, but not broadcast the message.
    assert_eq!(cluster.node0.role(), noraft::Role::Leader);
    let commit_position = cluster.node0.propose_command();
    assert_eq!(commit_position, cluster.node0.log().last_position(),);
    while let Some(_) = cluster.node0.actions_mut().next() {}

    // Make node2 the leader.
    let _call = cluster.node2.asserted_follower_election_timeout();
    let call = cluster.node2.asserted_candidate_election_timeout(); // Increase term.

    // Callers can filter out potentially disruptive RequestVoteRPCs.
    let should_ignore = cluster.node0.could_be_disruptive_request_vote(&call);
    assert!(should_ignore);
    if !should_ignore {
        cluster
            .node0
            .handle_message(&call)
            .expect("message handling should succeed");
    }
    assert_eq!(cluster.node0.role(), noraft::Role::Leader);
    assert_no_action!(cluster.node0);

    // The log index of node1 is equal to node2 => granted.
    let _ = cluster.node1.asserted_follower_election_timeout();
    let reply = cluster
        .node1
        .asserted_handle_request_vote_call_success(&call);
    let call = cluster
        .node2
        .asserted_handle_request_vote_reply_majority_vote_granted(&reply);
    assert_eq!(cluster.node2.role(), noraft::Role::Leader);

    // The uncommitted log entries on node0 are truncated.
    let reply = cluster
        .node0
        .asserted_handle_append_entries_call_success(&call);
    assert!(
        cluster
            .node0
            .get_commit_status(commit_position)
            .is_in_progress()
    );

    cluster
        .node2
        .asserted_handle_append_entries_reply_success(&reply, true, false);

    let call = cluster.node2.asserted_heartbeat();
    let _reply = cluster
        .node0
        .asserted_handle_append_entries_call_success(&call);
    assert!(
        cluster
            .node0
            .get_commit_status(commit_position)
            .is_rejected()
    );

    assert_no_action!(cluster.node0);
    assert_no_action!(cluster.node1);
    assert_no_action!(cluster.node2);
}

#[test]
fn same_index_different_term_tail_is_truncated_before_replication() {
    let voters = [id(0), id(1), id(2)];
    let config = joint(&voters, &[]);

    let leader_prefix = noraft::LogEntries::from_iter(
        noraft::LogPosition::ZERO,
        [
            cluster_config_entry(config.clone()),
            term_entry(t(2)),
            noraft::LogEntry::Command,
        ],
    );
    let leader_log = noraft::Log::new(noraft::ClusterConfig::new(), leader_prefix);
    let mut leader = TestNode {
        inner: noraft::Node::restart(id(0), t(3), None, leader_log),
        actions: noraft::Actions::default(),
    };
    assert_action!(leader, set_election_timeout());
    assert_no_action!(leader);

    let _call = leader.asserted_follower_election_timeout();
    assert_eq!(leader.current_term(), t(4));
    let vote_reply = request_vote_reply(leader.current_term(), id(2), true);
    let _initial_append =
        leader.asserted_handle_request_vote_reply_majority_vote_granted(&vote_reply);
    assert_eq!(leader.role(), noraft::Role::Leader);
    assert_eq!(leader.log().last_position(), log_pos(t(4), i(4)));

    let divergent_log = noraft::Log::new(
        noraft::ClusterConfig::new(),
        noraft::LogEntries::from_iter(
            noraft::LogPosition::ZERO,
            [
                cluster_config_entry(config),
                term_entry(t(2)),
                noraft::LogEntry::Command,
                term_entry(t(3)),
            ],
        ),
    );
    let mut follower = TestNode {
        inner: noraft::Node::restart(id(1), t(3), None, divergent_log),
        actions: noraft::Actions::default(),
    };
    assert_action!(follower, set_election_timeout());
    assert_no_action!(follower);
    assert_eq!(follower.log().last_position(), log_pos(t(3), i(4)));

    let divergent_reply = noraft::Message::AppendEntriesReply {
        from: follower.id(),
        term: leader.current_term(),
        last_position: follower.log().last_position(),
    };
    let truncate_call = leader.asserted_handle_append_entries_reply_failure(&divergent_reply);
    let noraft::Message::AppendEntriesCall { entries, .. } = &truncate_call else {
        panic!("Expected AppendEntriesCall");
    };
    assert!(entries.is_empty());
    assert_eq!(entries.prev_position(), log_pos(t(4), i(4)));
    assert_eq!(entries.last_position(), log_pos(t(4), i(4)));

    let truncated_reply = follower.asserted_handle_append_entries_call_failure(&truncate_call);
    assert_eq!(follower.log().last_position(), log_pos(t(2), i(3)));

    leader
        .handle_message(&truncated_reply)
        .expect("message handling should succeed");
    let Some(repair_call) = leader.actions_mut().send_messages.remove(&follower.id()) else {
        panic!("Expected repair AppendEntriesCall");
    };
    assert_no_action!(leader);
    let noraft::Message::AppendEntriesCall { entries, .. } = &repair_call else {
        panic!("Expected AppendEntriesCall");
    };
    assert_eq!(entries.prev_position(), log_pos(t(2), i(3)));
    assert_eq!(entries.last_position(), log_pos(t(4), i(4)));
    assert_eq!(entries.iter().collect::<Vec<_>>(), [term_entry(t(4))]);

    let repaired_reply = follower.asserted_handle_append_entries_call_success(&repair_call);
    assert_eq!(follower.log().last_position(), leader.log().last_position());

    leader.asserted_handle_append_entries_reply_success(&repaired_reply, true, false);
    assert_eq!(leader.commit_index(), i(4));
}

#[test]
fn follower_rewrites_from_common_prefix_when_repair_term_is_outside_local_log() {
    let leader_term = t(4);
    let follower_log = noraft::Log::new(
        noraft::ClusterConfig::new(),
        noraft::LogEntries::from_iter(
            noraft::LogPosition::ZERO,
            [
                term_entry(t(1)),
                noraft::LogEntry::Command,
                noraft::LogEntry::Command,
                term_entry(t(3)),
            ],
        ),
    );
    let mut follower = TestNode {
        inner: noraft::Node::restart(id(1), leader_term, Some(id(0)), follower_log),
        actions: noraft::Actions::default(),
    };
    assert_action!(follower, set_election_timeout());
    assert_no_action!(follower);

    let repair_entries = noraft::LogEntries::from_iter(
        log_pos(t(1), i(3)),
        [
            noraft::LogEntry::Command,
            noraft::LogEntry::Command,
            term_entry(t(2)),
        ],
    );
    let repair_call = noraft::Message::AppendEntriesCall {
        from: id(0),
        term: leader_term,
        commit_index: i(6),
        entries: repair_entries.clone(),
    };

    follower
        .handle_message(&repair_call)
        .expect("message handling should succeed");

    let reply = append_entries_reply(&repair_call, &follower);
    assert_eq!(follower.commit_index(), i(6));
    assert_eq!(
        follower
            .log()
            .entries()
            .iter_with_positions()
            .collect::<Vec<_>>(),
        vec![
            (log_pos(t(1), i(1)), term_entry(t(1))),
            (log_pos(t(1), i(2)), noraft::LogEntry::Command),
            (log_pos(t(1), i(3)), noraft::LogEntry::Command),
            (log_pos(t(1), i(4)), noraft::LogEntry::Command),
            (log_pos(t(1), i(5)), noraft::LogEntry::Command),
            (log_pos(t(2), i(6)), term_entry(t(2))),
        ]
    );
    assert_action!(follower, append_log_entries(&repair_entries));
    assert_action!(follower, send_message(id(0), &reply));
    assert_action!(follower, set_election_timeout());
    assert_no_action!(follower);
}

#[test]
fn snapshot() {
    let mut cluster = ThreeNodeCluster::new();
    cluster.init_cluster();
    cluster.propose_command();
    assert_eq!(cluster.node0.role(), noraft::Role::Leader);

    // Take a snapshot.
    for node in &mut [&mut cluster.node0, &mut cluster.node1, &mut cluster.node2] {
        assert_eq!(
            node.log().entries().prev_position().index,
            noraft::LogIndex::new(0)
        );
        let snapshot_config = node.log().latest_config().clone();
        let snapshot_position = node.log().entries().last_position();
        assert!(node.handle_snapshot_installed(snapshot_position, snapshot_config));
        assert_ne!(
            node.log().entries().prev_position().index,
            noraft::LogIndex::new(0)
        );
        // Every node had already committed the last position, so the
        // install keeps `commit_index` at the snapshot boundary.
        assert_eq!(node.commit_index(), snapshot_position.index);
    }

    // Add a new node and remove two nodes.
    let mut node3 = TestNode::asserted_start(id(3), &[]);
    let config = joint(
        &[cluster.node0.id(), cluster.node1.id(), cluster.node2.id()],
        &[cluster.node0.id(), node3.id()],
    );
    let call = cluster.node0.asserted_change_cluster_config(config);
    for node in &mut [&mut cluster.node1, &mut cluster.node2] {
        let reply = node.asserted_handle_append_entries_call_success(&call);
        cluster
            .node0
            .asserted_handle_append_entries_reply_success(&reply, false, false);
    }

    // Cannot append (need snapshot).
    let reply = node3.asserted_handle_append_entries_call_failure(&call);
    let (snapshot_config, snapshot_position) = cluster
        .node0
        .asserted_handle_append_entries_reply_failure_need_snapshot(&reply);
    assert!(node3.handle_snapshot_installed(snapshot_position, snapshot_config));
    // A fresh member joining via snapshot install lands with
    // `commit_index` at the snapshot boundary.
    assert_eq!(node3.commit_index(), snapshot_position.index);

    // Append after snapshot.
    let call = cluster.node0.asserted_heartbeat();
    let reply = node3.asserted_handle_append_entries_call_failure(&call);

    let call = cluster
        .node0
        .asserted_handle_append_entries_reply_failure(&reply);
    let reply = node3.asserted_handle_append_entries_call_success(&call);
    cluster
        .node0
        .asserted_handle_append_entries_reply_success_with_joint_config_committed(&reply);
}

#[derive(Debug)]
struct ThreeNodeCluster {
    node0: TestNode,
    node1: TestNode,
    node2: TestNode,
}

impl ThreeNodeCluster {
    fn new() -> Self {
        let initial_voters = &[id(0), id(1), id(2)];
        Self {
            node0: TestNode::asserted_start(id(0), initial_voters),
            node1: TestNode::asserted_start(id(1), &[]),
            node2: TestNode::asserted_start(id(2), &[]),
        }
    }

    fn init_cluster(&mut self) {
        // Setup  cluster.
        self.node0.handle_election_timeout();
        assert_eq!(self.node0.role(), noraft::Role::Candidate);
        assert_action!(self.node0, set_election_timeout());
        assert_action!(self.node0, save_current_term());
        assert_action!(self.node0, save_voted_for());
        let call = self
            .node0
            .actions_mut()
            .broadcast_message
            .take()
            .expect("broadcast");
        assert_no_action!(self.node0);

        for node in &mut [&mut self.node1, &mut self.node2] {
            let reply = node.asserted_handle_request_vote_call_success(&call);
            if node.id() == id(1) {
                self.node0
                    .asserted_handle_request_vote_reply_majority_vote_granted(&reply);
            }
        }
        assert_eq!(self.node0.role(), noraft::Role::Leader);

        let call = self.node0.take_broadcast_message();
        for node in &mut [&mut self.node1, &mut self.node2] {
            let reply = node.asserted_handle_append_entries_call_failure(&call);
            let call = self
                .node0
                .asserted_handle_append_entries_reply_failure(&reply);
            let reply = node.asserted_handle_append_entries_call_success(&call);
            if node.id() == id(1) {
                self.node0
                    .asserted_handle_append_entries_reply_success(&reply, true, false);
            }
        }
        assert_eq!(self.node0.config(), self.node1.config());
        assert_eq!(self.node0.config(), self.node2.config());
    }

    fn propose_command(&mut self) {
        let mut commit_position = None;
        let mut call = None;
        for node in &mut [&mut self.node0, &mut self.node1, &mut self.node2] {
            if node.role() != noraft::Role::Leader {
                continue;
            }
            commit_position = Some(node.propose_command());
            assert_action!(
                node.inner,
                append_log_entry(
                    log_prev(node.log().entries().last_position()),
                    noraft::LogEntry::Command
                )
            );
            let msg = append_entries_call(
                &node.inner,
                noraft::LogEntries::from_iter(
                    log_prev(node.log().entries().last_position()),
                    std::iter::once(noraft::LogEntry::Command),
                ),
            );
            assert_action!(node, broadcast_message(&msg));
            assert_action!(node, set_election_timeout());
            assert_no_action!(node);
            call = Some(msg);
            break;
        }

        let (Some(commit_position), Some(call)) = (commit_position, call) else {
            panic!("No leader found.");
        };

        let mut replies = Vec::new();
        for node in &mut [&mut self.node0, &mut self.node1, &mut self.node2] {
            if node.role() == noraft::Role::Leader {
                continue;
            }

            replies.push(node.asserted_handle_append_entries_call_success(&call));
        }

        let mut first = true;
        for node in &mut [&mut self.node0, &mut self.node1, &mut self.node2] {
            if node.role() != noraft::Role::Leader {
                continue;
            }

            for reply in replies {
                node.asserted_handle_append_entries_reply_success(&reply, first, false);
                assert_eq!(node.commit_index(), commit_position.index);
                first = false;
            }
            break;
        }
    }
}

#[derive(Debug)]
struct TestNode {
    inner: noraft::Node,
    actions: noraft::Actions,
}

impl TestNode {
    fn take_broadcast_message(&mut self) -> noraft::Message {
        self.actions
            .broadcast_message
            .take()
            .expect("No broadcast message.")
    }

    fn asserted_start(id: noraft::NodeId, initial_voters: &[noraft::NodeId]) -> Self {
        let mut node = noraft::Node::start(id);
        assert_eq!(node.role(), noraft::Role::Follower);
        assert_eq!(node.current_term(), t(0));
        assert_eq!(node.voted_for(), None);
        assert_no_action!(node);

        if !initial_voters.is_empty() {
            assert_ne!(
                node.create_cluster(initial_voters),
                noraft::LogPosition::INVALID
            );

            assert_action!(node, set_election_timeout());
            assert_action!(node, save_current_term());
            assert_action!(node, save_voted_for());

            if initial_voters == [id] {
                assert_eq!(node.role(), noraft::Role::Leader);
                assert_action!(
                    node,
                    append_log_entries(&noraft::LogEntries::from_iter(
                        prev(t(0), i(0)),
                        [
                            cluster_config_entry(joint(initial_voters, &[])),
                            term_entry(t(1))
                        ]
                    ))
                );
            } else {
                assert_eq!(node.role(), noraft::Role::Candidate);
                assert_action!(
                    node,
                    append_log_entries(&noraft::LogEntries::from_iter(
                        prev(t(0), i(0)),
                        [cluster_config_entry(joint(initial_voters, &[]))]
                    ))
                );
                assert!(matches!(
                    node.actions_mut().next(),
                    Some(noraft::Action::BroadcastMessage(
                        noraft::Message::RequestVoteCall { .. }
                    ))
                ));
            }
            assert_no_action!(node);
        }
        Self {
            inner: node,
            actions: noraft::Actions::default(),
        }
    }

    fn asserted_change_cluster_config(
        &mut self,
        new_config: noraft::ClusterConfig,
    ) -> noraft::Message {
        let prev_entry = self.log().entries().last_position();
        let next_index = next_index(self.log().entries().last_position().index);
        let next_position = log_pos(self.current_term(), next_index);
        assert_eq!(next_position, self.propose_config(new_config.clone()));
        let msg = append_entries_call(
            self,
            noraft::LogEntries::from_iter(
                prev_entry,
                std::iter::once(cluster_config_entry(new_config.clone())),
            ),
        );

        assert_action!(
            self,
            append_log_entry(prev_entry, cluster_config_entry(new_config.clone()))
        );
        assert_action!(self, broadcast_message(&msg));
        assert_action!(self, set_election_timeout());
        assert_no_action!(self);

        msg
    }

    fn asserted_handle_append_entries_call_success(
        &mut self,
        msg: &noraft::Message,
    ) -> noraft::Message {
        assert!(matches!(msg, noraft::Message::AppendEntriesCall { .. }));
        let old_role = self.role();

        let noraft::Message::AppendEntriesCall {
            entries,
            commit_index: leader_commit,
            ..
        } = msg
        else {
            unreachable!();
        };

        let prev_commit_index = self.commit_index();
        let prev_voted_for = self.voted_for();

        self.handle_message(msg)
            .expect("message handling should succeed");
        assert_eq!(
            self.log().entries().last_position(),
            entries.last_position()
        );
        if prev_voted_for != Some(msg.from()) {
            assert_action!(self, save_voted_for());
            assert_eq!(self.voted_for(), Some(msg.from()));
        }

        let reply = append_entries_reply(msg, self);
        if !entries.is_empty() {
            assert_action!(self, append_log_entries(entries));
        }
        if prev_commit_index < *leader_commit
            && prev_commit_index <= self.log().entries().last_position().index
        {
            assert_eq!(
                self.commit_index(),
                self.log()
                    .entries()
                    .last_position()
                    .index
                    .min(*leader_commit)
            );
        }
        assert_action!(self, send_message(msg.from(), &reply));
        assert_action!(self, set_election_timeout());
        if old_role.is_leader() {
            assert_action!(self, save_current_term());
        }
        assert_no_action!(self);

        reply
    }

    fn asserted_handle_append_entries_call_failure(
        &mut self,
        msg: &noraft::Message,
    ) -> noraft::Message {
        assert!(matches!(msg, noraft::Message::AppendEntriesCall { .. }));

        let noraft::Message::AppendEntriesCall { entries, .. } = msg else {
            unreachable!();
        };

        let prev_voted_for = self.voted_for();
        let prev_term = self.current_term();

        self.handle_message(msg)
            .expect("message handling should succeed");
        assert_ne!(
            self.log().entries().last_position(),
            entries.last_position()
        );
        if prev_term < msg.term() {
            assert_action!(self, save_current_term());
            assert_eq!(self.current_term(), msg.term());
        }
        if prev_voted_for != Some(msg.from()) {
            assert_action!(self, save_voted_for());
            assert_eq!(self.voted_for(), Some(msg.from()));
        }
        assert_action!(self, set_election_timeout());

        let reply = append_entries_reply(msg, self);
        assert_action!(self, send_message(msg.from(), &reply));
        assert_no_action!(self);

        reply
    }

    fn asserted_handle_append_entries_reply_failure_need_snapshot(
        &mut self,
        msg: &noraft::Message,
    ) -> (noraft::ClusterConfig, noraft::LogPosition) {
        assert!(matches!(msg, noraft::Message::AppendEntriesReply { .. }));

        let noraft::Message::AppendEntriesReply {
            from,
            last_position,
            ..
        } = msg
        else {
            unreachable!();
        };
        assert!(since(self.log().entries(), *last_position).is_none());

        self.handle_message(msg)
            .expect("message handling should succeed");
        assert_action!(self, noraft::Action::InstallSnapshot(*from));
        assert_no_action!(self);

        (
            self.log().snapshot_config().clone(),
            self.log().entries().prev_position(),
        )
    }

    fn asserted_handle_append_entries_reply_success_with_joint_config_committed(
        &mut self,
        msg: &noraft::Message,
    ) -> noraft::Message {
        assert!(matches!(msg, noraft::Message::AppendEntriesReply { .. }));
        assert!(self.config().is_joint_consensus());

        let prev_entry = self.log().entries().last_position();
        let mut new_config = self.config().clone();
        new_config.voters = std::mem::take(&mut new_config.new_voters);

        let noraft::Message::AppendEntriesReply { last_position, .. } = msg else {
            unreachable!();
        };

        self.handle_message(msg)
            .expect("message handling should succeed");
        let call = append_entries_call(
            self,
            noraft::LogEntries::from_iter(
                prev_entry,
                std::iter::once(cluster_config_entry(new_config.clone())),
            ),
        );
        assert_eq!(self.commit_index(), last_position.index);
        assert_action!(
            self,
            append_log_entry(prev_entry, cluster_config_entry(new_config.clone()))
        );
        assert_action!(self, broadcast_message(&call));
        assert_action!(self, set_election_timeout());
        assert_no_action!(self);

        call
    }

    fn asserted_handle_append_entries_reply_success(
        &mut self,
        reply: &noraft::Message,
        commit_index_will_be_updated: bool,
        joint_consensus_will_be_finalized: bool,
    ) {
        assert!(matches!(reply, noraft::Message::AppendEntriesReply { .. }));

        let old_last_position = self.log().entries().last_position();
        self.handle_message(reply)
            .expect("message handling should succeed");
        self.actions = self.inner.actions().clone();

        let noraft::Message::AppendEntriesReply { last_position, .. } = reply else {
            unreachable!();
        };
        if commit_index_will_be_updated {
            assert_eq!(self.commit_index(), last_position.index);
        }
        if joint_consensus_will_be_finalized {
            assert_action!(self, set_election_timeout());

            let config = self.config().clone();
            assert_action!(
                self,
                append_log_entry(old_last_position, cluster_config_entry(config.clone()))
            );
            assert_action!(
                self,
                broadcast_message(&append_entries_call(
                    self,
                    noraft::LogEntries::from_iter(
                        old_last_position,
                        std::iter::once(cluster_config_entry(config.clone()))
                    )
                ))
            );
        }
        assert_no_action!(self);
    }

    fn asserted_handle_append_entries_reply_failure(
        &mut self,
        reply: &noraft::Message,
    ) -> noraft::Message {
        assert!(matches!(reply, noraft::Message::AppendEntriesReply { .. }));

        self.handle_message(reply)
            .expect("message handling should succeed");
        let Some(call) = self.actions_mut().send_messages.remove(&reply.from()) else {
            panic!("No send message action");
        };
        assert_no_action!(self);

        call
    }

    fn asserted_follower_election_timeout(&mut self) -> noraft::Message {
        assert_eq!(self.role(), noraft::Role::Follower);

        let prev_term = self.current_term();
        self.handle_election_timeout();
        assert_eq!(self.role(), noraft::Role::Candidate);
        assert_eq!(self.current_term(), next_term(prev_term));

        let call = request_vote_call(
            self.current_term(),
            self.id(),
            self.log().entries().last_position(),
        );
        assert_action!(self, save_current_term());
        assert_eq!(self.current_term(), next_term(prev_term));
        assert_action!(self, save_voted_for());
        assert_eq!(self.voted_for(), Some(self.id()));
        assert_action!(self, broadcast_message(&call));
        assert_action!(self, set_election_timeout());
        assert_no_action!(self);

        call
    }

    fn asserted_candidate_election_timeout(&mut self) -> noraft::Message {
        assert_eq!(self.role(), noraft::Role::Candidate);

        let prev_term = self.current_term();
        self.handle_election_timeout();
        assert_eq!(self.role(), noraft::Role::Candidate);
        assert_eq!(self.current_term(), next_term(prev_term));

        let call = request_vote_call(
            self.current_term(),
            self.id(),
            self.log().entries().last_position(),
        );
        assert_action!(self, save_current_term());
        assert_eq!(self.current_term(), next_term(prev_term));
        assert_action!(self, save_voted_for());
        assert_eq!(self.voted_for(), Some(self.id()));
        assert_action!(self, broadcast_message(&call));
        assert_action!(self, set_election_timeout());
        assert_no_action!(self);

        call
    }

    fn asserted_handle_request_vote_call_success(
        &mut self,
        msg: &noraft::Message,
    ) -> noraft::Message {
        assert!(matches!(msg, noraft::Message::RequestVoteCall { .. }));

        self.handle_message(msg)
            .expect("message handling should succeed");

        let reply = request_vote_reply(msg.term(), self.id(), true);
        assert_action!(self, save_current_term());
        assert_eq!(self.current_term(), msg.term());
        assert_action!(self, save_voted_for());
        assert_eq!(self.voted_for(), Some(msg.from()));
        assert_action!(self, set_election_timeout());
        assert_action!(self, send_message(msg.from(), &reply));
        assert_no_action!(self);

        reply
    }

    fn asserted_handle_request_vote_reply_majority_vote_granted(
        &mut self,
        msg: &noraft::Message,
    ) -> noraft::Message {
        assert!(matches!(msg, noraft::Message::RequestVoteReply { .. }));

        let tail = self.log().entries().last_position();
        self.handle_message(msg)
            .expect("message handling should succeed");
        self.actions = self.inner.actions().clone();
        let call = append_entries_call(
            self,
            noraft::LogEntries::from_iter(tail, std::iter::once(term_entry(self.current_term()))),
        );
        assert_action!(
            self,
            append_log_entry(tail, term_entry(self.current_term()))
        );
        assert_action!(self, broadcast_message(&call));
        assert_action!(self, set_election_timeout());
        assert_no_action!(self);

        call
    }

    fn asserted_handle_append_entries_call_success_new_leader(
        &mut self,
        msg: &noraft::Message,
    ) -> noraft::Message {
        assert!(matches!(msg, noraft::Message::AppendEntriesCall { .. }));

        let tail = self.log().entries().last_position();
        self.handle_message(msg)
            .expect("message handling should succeed");
        let reply = append_entries_reply(msg, self);
        assert_action!(self, save_current_term());
        assert_eq!(self.current_term(), msg.term());
        assert_action!(self, save_voted_for());
        assert_eq!(self.voted_for(), Some(msg.from()));
        assert_action!(self, set_election_timeout());
        assert_action!(self, append_log_entry(tail, term_entry(msg.term())));
        assert_action!(self, send_message(msg.from(), &reply));
        assert_no_action!(self);

        reply
    }

    fn asserted_heartbeat(&mut self) -> noraft::Message {
        assert!(self.heartbeat());
        let call = append_entries_call(
            self,
            noraft::LogEntries::new(self.log().entries().last_position()),
        );
        assert_action!(self, set_election_timeout());
        assert_action!(self, broadcast_message(&call));
        assert_no_action!(self);
        call
    }
}

impl Deref for TestNode {
    type Target = noraft::Node;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for TestNode {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

fn id(id: u64) -> noraft::NodeId {
    noraft::NodeId::new(id)
}

fn t(term: u64) -> noraft::Term {
    noraft::Term::new(term)
}

fn i(index: u64) -> noraft::LogIndex {
    noraft::LogIndex::new(index)
}

fn prev(term: noraft::Term, index: noraft::LogIndex) -> noraft::LogPosition {
    log_pos(term, index)
}

fn joint(old: &[noraft::NodeId], new: &[noraft::NodeId]) -> noraft::ClusterConfig {
    let mut config = noraft::ClusterConfig::new();
    config.voters.extend(old.iter().copied());
    config.new_voters.extend(new.iter().copied());
    config
}

fn term_entry(term: noraft::Term) -> noraft::LogEntry {
    noraft::LogEntry::Term(term)
}

fn cluster_config_entry(config: noraft::ClusterConfig) -> noraft::LogEntry {
    noraft::LogEntry::ClusterConfig(config)
}

fn request_vote_call(
    term: noraft::Term,
    from: noraft::NodeId,
    last_position: noraft::LogPosition,
) -> noraft::Message {
    noraft::Message::RequestVoteCall {
        from,
        term,
        last_position,
    }
}

fn request_vote_reply(
    term: noraft::Term,
    from: noraft::NodeId,
    vote_granted: bool,
) -> noraft::Message {
    noraft::Message::RequestVoteReply {
        from,
        term,
        vote_granted,
    }
}

fn append_entries_call(leader: &noraft::Node, entries: noraft::LogEntries) -> noraft::Message {
    let term = leader.current_term();
    let from = leader.id();
    let commit_index = leader.commit_index();
    noraft::Message::AppendEntriesCall {
        from,
        term,
        commit_index,
        entries,
    }
}

fn append_entries_reply(call: &noraft::Message, node: &noraft::Node) -> noraft::Message {
    let noraft::Message::AppendEntriesCall { .. } = call else {
        panic!();
    };

    let term = node.current_term();
    let from = node.id();
    let last_position = node.log().entries().last_position();
    noraft::Message::AppendEntriesReply {
        from,
        term,
        last_position,
    }
}

fn send_message(destination: noraft::NodeId, message: &noraft::Message) -> noraft::Action {
    noraft::Action::SendMessage(destination, message.clone())
}

fn broadcast_message(message: &noraft::Message) -> noraft::Action {
    noraft::Action::BroadcastMessage(message.clone())
}

fn set_election_timeout() -> noraft::Action {
    noraft::Action::SetElectionTimeout
}

fn append_log_entry(prev: noraft::LogPosition, entry: noraft::LogEntry) -> noraft::Action {
    noraft::Action::AppendLogEntries(noraft::LogEntries::from_iter(prev, std::iter::once(entry)))
}

fn append_log_entries(entries: &noraft::LogEntries) -> noraft::Action {
    noraft::Action::AppendLogEntries(entries.clone())
}

fn save_current_term() -> noraft::Action {
    noraft::Action::SaveCurrentTerm
}

fn save_voted_for() -> noraft::Action {
    noraft::Action::SaveVotedFor
}

fn next_term(term: noraft::Term) -> noraft::Term {
    noraft::Term::new(term.get() + 1)
}

fn next_index(index: noraft::LogIndex) -> noraft::LogIndex {
    noraft::LogIndex::new(index.get() + 1)
}

fn log_pos(term: noraft::Term, index: noraft::LogIndex) -> noraft::LogPosition {
    noraft::LogPosition { term, index }
}

fn log_prev(entry: noraft::LogPosition) -> noraft::LogPosition {
    log_pos(entry.term, noraft::LogIndex::new(entry.index.get() - 1))
}

fn since(
    entries: &noraft::LogEntries,
    position: noraft::LogPosition,
) -> Option<noraft::LogEntries> {
    if !entries.contains(position) {
        return None;
    }
    Some(noraft::LogEntries::from_iter(
        position,
        entries
            .iter()
            .skip(position.index.get() as usize - position.index.get() as usize),
    ))
}

fn next_same_kind_action(
    actions: &mut noraft::Actions,
    expected: &noraft::Action,
) -> Option<noraft::Action> {
    match expected {
        noraft::Action::SetElectionTimeout if actions.set_election_timeout => {
            actions.set_election_timeout = false;
            Some(noraft::Action::SetElectionTimeout)
        }
        noraft::Action::SaveCurrentTerm if actions.save_current_term => {
            actions.save_current_term = false;
            Some(noraft::Action::SaveCurrentTerm)
        }
        noraft::Action::SaveVotedFor if actions.save_voted_for => {
            actions.save_voted_for = false;
            Some(noraft::Action::SaveVotedFor)
        }
        noraft::Action::AppendLogEntries(_) => actions
            .append_log_entries
            .take()
            .map(noraft::Action::AppendLogEntries),
        noraft::Action::BroadcastMessage(_) => actions
            .broadcast_message
            .take()
            .map(noraft::Action::BroadcastMessage),
        noraft::Action::SendMessage(node_id, _) => actions
            .send_messages
            .remove(node_id)
            .map(|msg| noraft::Action::SendMessage(*node_id, msg)),
        noraft::Action::InstallSnapshot(node_id) => actions
            .install_snapshots
            .remove(node_id)
            .then_some(noraft::Action::InstallSnapshot(*node_id)),
        _ => None,
    }
}
