//! Stateful property-based tests for a single [`Node`]'s state
//! machine, driven by noprop (https://github.com/sile/noprop).
//!
//! A bare node is driven by an arbitrary sequence of operations
//! (election timeout, propose command / config, incoming RPC message)
//! and the following invariants are checked after every step:
//!
//! - `current_term()` is non-decreasing
//! - `commit_index()` is non-decreasing
//! - a leader does not step down within the same term
//! - a leader does not truncate its own log (leader append-only)
//!
//! Seed / case budget come from the `NORAFT_PBT_SEED` and
//! `NORAFT_PBT_CASES` environment variables; unset means
//! "clock-derived seed" and "512 cases". A failing seed can be
//! re-run with:
//!
//! ```text
//! NORAFT_PBT_SEED=<seed> cargo test --test prop_node_state_machine_test
//! ```

use noraft::{
    ClusterConfig, LogEntries, LogEntry, LogIndex, LogPosition, Message, Node, NodeGeneration,
    NodeId, Role, Term,
};
use std::cell::Cell;
use std::fmt;

const MIN_STEPS: usize = 20;
const MAX_STEPS: usize = 200;

/// One operation applied to the node.
#[derive(Clone, Debug)]
enum Op {
    HandleElectionTimeout,
    ProposeCommand,
    ProposeConfig,
    HandleMessage,
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::HandleElectionTimeout => write!(f, "HandleElectionTimeout"),
            Op::ProposeCommand => write!(f, "ProposeCommand"),
            Op::ProposeConfig => write!(f, "ProposeConfig"),
            Op::HandleMessage => write!(f, "HandleMessage"),
        }
    }
}

/// Samples an operation, weighting the message handling path so that
/// incoming RPCs (the primary way a node's state changes) are
/// exercised often.
fn sample_op(ctx: &mut noprop::TestCaseContext) -> Op {
    match noprop::sample_weighted_index(ctx, &[1, 1, 1, 3]) {
        0 => Op::HandleElectionTimeout,
        1 => Op::ProposeCommand,
        2 => Op::ProposeConfig,
        _ => Op::HandleMessage,
    }
}

fn sample_position(ctx: &mut noprop::TestCaseContext) -> LogPosition {
    LogPosition::new(
        Term::new(noprop::sample_usize_in(ctx, 0..16) as u64),
        LogIndex::new(noprop::sample_usize_in(ctx, 0..16) as u64),
    )
}

fn sample_entries(ctx: &mut noprop::TestCaseContext) -> LogEntries {
    let mut entries = LogEntries::new(sample_position(ctx));
    let count = noprop::sample_usize_in(ctx, 0..8);
    for _ in 0..count {
        let entry = match noprop::sample_usize_in(ctx, 0..3) {
            0 => LogEntry::Term(Term::new(noprop::sample_usize_in(ctx, 0..16) as u64)),
            1 => LogEntry::ClusterConfig(ClusterConfig::new()),
            _ => LogEntry::Command,
        };
        entries.push(entry);
    }
    entries
}

/// A message from some other node, with arbitrary (possibly stale or
/// future) term and log position.
fn sample_message(ctx: &mut noprop::TestCaseContext, from: NodeId) -> Message {
    let term = Term::new(noprop::sample_usize_in(ctx, 0..16) as u64);
    match noprop::sample_usize_in(ctx, 0..4) {
        0 => Message::RequestVoteCall {
            from,
            term,
            last_position: sample_position(ctx),
        },
        1 => Message::RequestVoteReply {
            from,
            term,
            vote_granted: noprop::sample_bool(ctx),
        },
        2 => Message::AppendEntriesCall {
            from,
            term,
            commit_index: LogIndex::new(noprop::sample_usize_in(ctx, 0..16) as u64),
            entries: sample_entries(ctx),
        },
        _ => Message::AppendEntriesReply {
            from,
            term,
            generation: NodeGeneration::new(0),
            last_position: sample_position(ctx),
        },
    }
}

/// The single-node state machine invariants:
///
/// - `current_term` is non-decreasing
/// - `commit_index` is non-decreasing
/// - a leader does not step down within the same term
/// - a leader's log is append-only (its last index never decreases)
#[test]
fn single_node_state_machine_invariants() -> noprop::TestResult {
    let seed = noprop::seed_from_env_or_time("NORAFT_PBT_SEED")?;
    let cases = std::env::var("NORAFT_PBT_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512_usize);

    let cases_with_leader: Cell<usize> = Cell::new(0);
    let cases_with_messages: Cell<usize> = Cell::new(0);
    let cases_with_commits: Cell<usize> = Cell::new(0);
    noprop::Runner::new(seed).run(cases, |ctx| {
        // Start as a solo-voter cluster so the leader state and the
        // commit path are reachable (a bare `Node::start` node is not
        // a voter of any config and can never become a candidate).
        let mut node = Node::start(NodeId::new(0));
        node.create_cluster(&[NodeId::new(0)]);
        let _ = node.actions_mut().by_ref().collect::<Vec<_>>();

        let steps = noprop::sample_usize_in(ctx, MIN_STEPS..=MAX_STEPS);
        let mut history: Vec<String> = Vec::with_capacity(steps);
        let mut prev_term = Term::ZERO;
        let mut prev_commit = LogIndex::ZERO;
        let mut prev_role = Role::Follower;
        let mut leader_last_index = LogIndex::ZERO;
        let mut saw_message = false;

        for step in 0..steps {
            let op = sample_op(ctx);
            match &op {
                Op::HandleElectionTimeout => node.handle_election_timeout(),
                Op::ProposeCommand => {
                    node.propose_command();
                }
                Op::ProposeConfig => {
                    let mut config = ClusterConfig::new();
                    config.voters.insert(NodeId::new(0));
                    config.voters.insert(NodeId::new(1));
                    node.propose_config(config);
                }
                Op::HandleMessage => {
                    let msg = sample_message(ctx, NodeId::new(1));
                    let _ = node.handle_message(&msg);
                    saw_message = true;
                }
            }
            // Drop the produced actions: this property only observes
            // the node state, not the actions it emits.
            let _ = node.actions_mut().by_ref().collect::<Vec<_>>();
            history.push(op.to_string());

            let term = node.current_term();
            let commit = node.commit_index();
            let role = node.role();
            if term < prev_term {
                return Err(format!(
                    "term must be non-decreasing at step {step}: {term:?} < {prev_term:?}; \
                     history=[{}]",
                    history.join(", ")
                )
                .into());
            }
            if commit < prev_commit {
                return Err(format!(
                    "commit_index must be non-decreasing at step {step}: {commit:?} < \
                     {prev_commit:?}; history=[{}]",
                    history.join(", ")
                )
                .into());
            }
            if term == prev_term && prev_role == Role::Leader && role != Role::Leader {
                return Err(format!(
                    "leader must not step down within the same term at step {step}: \
                     {role:?} after {prev_role:?} in term {term:?}; history=[{}]",
                    history.join(", ")
                )
                .into());
            }
            // Leader append-only: while this node is the leader of a
            // term, its log's last index must never decrease. The
            // baseline resets at a term boundary, where the log may
            // legitimately have been truncated while a follower.
            if term != prev_term {
                leader_last_index = node.log().last_position().index;
            }
            if role.is_leader() {
                let last_index = node.log().last_position().index;
                if last_index < leader_last_index {
                    return Err(format!(
                        "leader must not truncate its log at step {step}: last index \
                         decreased from {leader_last_index:?} to {last_index:?}; \
                         history=[{}]",
                        history.join(", ")
                    )
                    .into());
                }
                leader_last_index = last_index;
                cases_with_leader.set(cases_with_leader.get() + 1);
            }
            if commit > LogIndex::ZERO {
                cases_with_commits.set(cases_with_commits.get() + 1);
            }

            prev_term = term;
            prev_commit = commit;
            prev_role = role;
        }
        if saw_message {
            cases_with_messages.set(cases_with_messages.get() + 1);
        }
        Ok(())
    })?;

    assert!(
        cases_with_leader.get() > 0,
        "no case exercised the leader role; the leader invariants were vacuous \
         (seed={seed:#018x})",
    );
    assert!(
        cases_with_messages.get() > 0,
        "no case handled a message; the message handling paths were never exercised \
         (seed={seed:#018x})",
    );
    assert!(
        cases_with_commits.get() > 0,
        "no case advanced the commit index; the commit path was never exercised \
         (seed={seed:#018x})",
    );
    Ok(())
}
