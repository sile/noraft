//! Membership-change PBTs under unstable cluster links.
//!
//! Every test obtains its seed through `noprop::seed_from_env_or_time`
//! via the shared runner. CI therefore uses a fresh time-derived seed
//! unless `NORAFT_PBT_SEED` is explicitly set for reproduction.

#[path = "helpers/pbt.rs"]
pub mod pbt;
#[path = "helpers/pbt_scenario.rs"]
pub mod pbt_scenario;

use noraft::{LogPosition, NodeId};
use pbt::run_config;
use pbt_scenario::{MinMax, TestCluster, TestNode, wait_until_terminal};
use std::cell::Cell;

/// Cluster configuration changes (add / remove voters and non-voters)
/// settle without getting stuck in joint consensus, and at least one
/// command commits somewhere across the run. The run must exercise at
/// least one config change.
#[test]
fn membership_changes_settle() -> noprop::TestResult {
    let config = run_config(16)?;
    let seed = config.seed;
    let cases_with_change: Cell<usize> = Cell::new(0);
    let cases_with_commit: Cell<usize> = Cell::new(0);

    noprop::Runner::new(seed).run(config.cases, |ctx| {
        let node_ids = [NodeId::new(0), NodeId::new(1), NodeId::new(2)];
        let mut cluster = TestCluster::new(&node_ids);
        cluster.default_link_options.drop_rate = noprop::Ratio::new(3, 10);
        cluster.default_link_options.latency_ticks = MinMax::new(1, 1000);

        let position = cluster.random_node_mut(ctx).create_cluster(&node_ids);
        assert_ne!(position, LogPosition::INVALID);
        let satisfied = cluster.run_until(ctx, cluster.clock.after(100_000), |cluster| {
            cluster.leader_node().is_some()
        });
        if !satisfied {
            return Err("cluster creation timed out".into());
        }

        let rounds = noprop::sample_usize_in(ctx, 1..=4);
        let mut total_committed = 0;
        for i in 0..rounds {
            // Wait for the previous configuration change to settle:
            // proposing the next change while the cluster is still in
            // joint consensus returns `LogPosition::INVALID`.
            let settled = cluster.run_until(ctx, cluster.clock.after(1_000_000), |cluster| {
                cluster
                    .leader_node()
                    .is_some_and(|leader| !leader.config().is_joint_consensus())
            });
            if !settled {
                return Err(format!("config change {i} did not settle").into());
            }
            if noprop::sample_ratio(ctx, noprop::Ratio::new(7, 10)) {
                // Add.
                let node_id = NodeId::new(3 + i as u64);
                let voter = noprop::sample_ratio(ctx, noprop::Ratio::one_nth(2));
                let mut node = TestNode::new(node_id);
                node.voter = voter;
                cluster.nodes.push(node);
                cases_with_change.set(cases_with_change.get() + 1);

                let Some(leader) = cluster.leader_node_mut() else {
                    unreachable!();
                };
                let new_config = if voter {
                    leader.config().to_joint_consensus(&[node_id], &[])
                } else {
                    let mut new_config = leader.config().clone();
                    new_config.non_voters.insert(node_id);
                    new_config
                };
                let position = leader.propose_config(new_config);
                assert_ne!(position, LogPosition::INVALID);
            } else {
                // Remove an actual member of the leader's current
                // config. Selecting from the harness node list would
                // repeatedly generate no-op removals after a node had
                // already left the config.
                let Some(leader) = cluster.leader_node() else {
                    unreachable!();
                };
                let mut candidates: Vec<(NodeId, bool)> = leader
                    .config()
                    .non_voters
                    .iter()
                    .copied()
                    .map(|id| (id, false))
                    .collect();
                if leader.config().voters.len() > 2 {
                    candidates.extend(leader.config().voters.iter().copied().map(|id| (id, true)));
                }
                if !candidates.is_empty() {
                    let (node_id, voter) = noprop::sample_choice(ctx, &candidates);
                    cases_with_change.set(cases_with_change.get() + 1);

                    let Some(leader) = cluster.leader_node_mut() else {
                        unreachable!();
                    };
                    let new_config = if voter {
                        leader.config().to_joint_consensus(&[], &[node_id])
                    } else {
                        let mut new_config = leader.config().clone();
                        new_config.non_voters.remove(&node_id);
                        new_config
                    };
                    let position = leader.propose_config(new_config);
                    assert_ne!(position, LogPosition::INVALID);
                }
            }

            // Propose commands.
            let mut positions = Vec::new();
            let command_count = noprop::sample_usize_in(ctx, 1..=4);
            for _ in 0..command_count {
                let found = cluster.run_while_leader_absent(ctx, cluster.clock.after(1_000_000));
                if !found {
                    return Err("leader absent while proposing".into());
                }
                let Some(leader) = cluster.leader_node_mut() else {
                    unreachable!();
                };
                positions.push(leader.propose_command());
                let ticks = MinMax::new(1, 10).sample(ctx);
                cluster.run(ctx, cluster.clock.after(ticks));
            }

            for position in positions.iter() {
                // A proposal may legitimately stay in progress when a
                // config change is still converging under an unstable
                // network (e.g. a newly added voter that has not yet
                // caught up); the liveness claim is that *some*
                // command commits, not that every one does.
                if let Some(status) = wait_until_terminal(&mut cluster, ctx, *position, 20_000)
                    && status.is_committed()
                {
                    total_committed += 1;
                }
            }
        }
        if total_committed == 0 {
            return Err("no proposal committed; the cluster made no progress".into());
        }
        cases_with_commit.set(cases_with_commit.get() + 1);
        Ok(())
    })?;

    assert!(
        cases_with_change.get() > 0,
        "no case exercised a config change (seed={seed:#018x})",
    );
    assert!(
        cases_with_commit.get() > 0,
        "no case committed a command under membership changes \
         (seed={seed:#018x})",
    );
    Ok(())
}
