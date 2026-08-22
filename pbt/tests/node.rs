//! Stateful properties for valid single-node cluster transitions.

use noraft::{
    ClusterConfig, CommitStatus, LogIndex, LogPosition, Message, Node, NodeId, Role, Term,
};
use pbt::{run, sample_len};

const MAX_STEPS: usize = 200;

#[derive(Debug, Clone, Copy)]
enum Op {
    ProposeCommand,
    ProposeSameConfig,
    AdvanceTerm,
    Heartbeat,
}

fn sample_op(ctx: &mut noprop::TestCaseContext) -> Op {
    match noprop::sample_weighted_index(ctx, &[4, 1, 1, 1]) {
        0 => Op::ProposeCommand,
        1 => Op::ProposeSameConfig,
        2 => Op::AdvanceTerm,
        _ => Op::Heartbeat,
    }
}

fn drain_actions(node: &mut Node) {
    for _ in node.actions_mut() {}
}

fn advance_term(node: &mut Node) -> noprop::TestResult {
    let requested_term = Term::new(node.current_term().get() + 1);
    let request = Message::RequestVoteCall {
        from: NodeId::new(1),
        term: requested_term,
        last_position: node.log().last_position(),
    };
    node.handle_message(&request)?;
    if node.current_term() != requested_term || node.role() != Role::Follower {
        return Err(format!(
            "higher-term vote request did not produce a follower in {requested_term:?}: \
             term={:?}, role={:?}",
            node.current_term(),
            node.role()
        )
        .into());
    }
    drain_actions(node);

    node.handle_election_timeout();
    if node.current_term() <= requested_term || node.role() != Role::Leader {
        return Err(format!(
            "solo follower did not win the next election: requested={requested_term:?}, \
             term={:?}, role={:?}",
            node.current_term(),
            node.role()
        )
        .into());
    }
    Ok(())
}

fn assert_monotonic_state(
    node: &Node,
    previous_term: Term,
    previous_commit: LogIndex,
    previous_last: LogIndex,
    step: usize,
    op: Op,
) -> noprop::TestResult {
    if node.current_term() < previous_term {
        return Err(format!(
            "term decreased at step {step} after {op:?}: {:?} < {previous_term:?}",
            node.current_term()
        )
        .into());
    }
    if node.commit_index() < previous_commit {
        return Err(format!(
            "commit index decreased at step {step} after {op:?}: {:?} < {previous_commit:?}",
            node.commit_index()
        )
        .into());
    }
    if node.log().last_position().index < previous_last {
        return Err(format!(
            "solo leader truncated its log at step {step} after {op:?}: {:?} < {previous_last:?}",
            node.log().last_position().index
        )
        .into());
    }
    if node.role() != Role::Leader {
        return Err(format!("solo voter stopped being leader at step {step} after {op:?}").into());
    }
    Ok(())
}

/// A solo voter commits every local proposal immediately, remains a
/// leader across election timeouts, and advances term, commit index,
/// and log position monotonically.
#[test]
fn solo_node_valid_transitions_preserve_invariants() -> noprop::TestResult {
    run(512, |ctx| {
        let id = NodeId::new(0);
        let mut node = Node::start(id);
        let cluster_position = node.create_cluster(&[id]);
        if cluster_position == LogPosition::INVALID || node.role() != Role::Leader {
            return Err("solo cluster creation did not produce a leader".into());
        }
        drain_actions(&mut node);

        // The first post-setup proposal is mandatory. This is the
        // coverage evidence for the command commit path; setup commits
        // are deliberately not counted.
        let baseline_commit = node.commit_index();
        let position = node.propose_command();
        if position == LogPosition::INVALID
            || node.get_commit_status(position) != CommitStatus::Committed
            || node.commit_index() <= baseline_commit
        {
            return Err(format!(
                "post-setup proposal did not commit: position={position:?}, baseline={baseline_commit:?}, \
                 commit={:?}, status={:?}",
                node.commit_index(),
                node.get_commit_status(position)
            )
            .into());
        }
        drain_actions(&mut node);

        // A mandatory timeout ensures the term-transition path is not
        // left to probability.
        advance_term(&mut node)?;
        drain_actions(&mut node);

        let steps = sample_len(ctx, MAX_STEPS);
        let mut previous_term = node.current_term();
        let mut previous_commit = node.commit_index();
        let mut previous_last = node.log().last_position().index;
        for step in 0..steps {
            let op = sample_op(ctx);
            match op {
                Op::ProposeCommand => {
                    let position = node.propose_command();
                    if position == LogPosition::INVALID
                        || node.get_commit_status(position) != CommitStatus::Committed
                    {
                        return Err(format!(
                            "command proposal failed at step {step}: {position:?}, status={:?}",
                            node.get_commit_status(position)
                        )
                        .into());
                    }
                }
                Op::ProposeSameConfig => {
                    let config: ClusterConfig = node.config().clone();
                    let position = node.propose_config(config);
                    if position == LogPosition::INVALID
                        || node.get_commit_status(position) != CommitStatus::Committed
                    {
                        return Err(format!(
                            "same-config proposal failed at step {step}: {position:?}, status={:?}",
                            node.get_commit_status(position)
                        )
                        .into());
                    }
                }
                Op::AdvanceTerm => advance_term(&mut node)?,
                Op::Heartbeat => {
                    if !node.heartbeat() {
                        return Err(format!("leader heartbeat failed at step {step}").into());
                    }
                }
            }

            assert_monotonic_state(
                &node,
                previous_term,
                previous_commit,
                previous_last,
                step,
                op,
            )?;
            previous_term = node.current_term();
            previous_commit = node.commit_index();
            previous_last = node.log().last_position().index;
            drain_actions(&mut node);
        }
        Ok(())
    })
}
