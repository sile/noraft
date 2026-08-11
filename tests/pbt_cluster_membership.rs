//! Membership-change PBTs under unstable cluster links.
//!
//! Every test obtains its seed through `noprop::seed_from_env_or_time`
//! via the shared runner. CI therefore uses a fresh time-derived seed
//! unless `NORAFT_PBT_SEED` is explicitly set for reproduction.

pub mod pbt_harness;

use noraft::{ClusterConfig, LogEntry, LogPosition, NodeId};
use pbt_harness::{MinMax, TestCluster, TestNode, run_config, wait_until_terminal};
use std::cell::Cell;

fn config_is_committed(cluster: &TestCluster, expected: &ClusterConfig) -> bool {
    if expected.is_joint_consensus() {
        return false;
    }

    expected.unique_nodes().all(|id| {
        let Some(node) = cluster.nodes.iter().find(|node| node.inner.id() == id) else {
            return false;
        };
        if node.inner.config() != expected {
            return false;
        }

        let mut position = (node.inner.log().snapshot_config() == expected)
            .then_some(node.inner.log().snapshot_position());
        for (candidate, entry) in node.inner.log().entries().iter_with_positions() {
            if matches!(entry, LogEntry::ClusterConfig(config) if config == *expected) {
                position = Some(candidate);
            }
        }
        position.is_some_and(|position| node.inner.get_commit_status(position).is_committed())
    })
}

fn observed_config_status(
    cluster: &TestCluster,
    position: LogPosition,
    expected: &ClusterConfig,
) -> Option<noraft::CommitStatus> {
    cluster
        .nodes
        .iter()
        .filter(|node| expected.voters.contains(&node.inner.id()))
        .map(|node| node.inner.get_commit_status(position))
        .find(|status| !status.is_in_progress())
}

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
            let mut proposed_config = None;
            if noprop::sample_ratio(ctx, noprop::Ratio::new(7, 10)) {
                // Add.
                let node_id = NodeId::new(3 + i as u64);
                let voter = noprop::sample_ratio(ctx, noprop::Ratio::one_nth(2));
                let mut node = TestNode::new(node_id);
                node.voter = voter;
                cluster.nodes.push(node);

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
                let mut expected = new_config.clone();
                if voter {
                    expected.voters = std::mem::take(&mut expected.new_voters);
                }
                let position = leader.propose_config(new_config);
                assert_ne!(position, LogPosition::INVALID);
                proposed_config = Some((position, expected, Some(node_id)));
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
                    let mut expected = new_config.clone();
                    if voter {
                        expected.voters = std::mem::take(&mut expected.new_voters);
                    }
                    let position = leader.propose_config(new_config);
                    assert_ne!(position, LogPosition::INVALID);
                    proposed_config = Some((position, expected, None));
                }
            }

            if let Some((position, expected, added_node)) = proposed_config {
                // Exercise the proposal while links are unstable, then
                // heal the network so the oracle checks safety and
                // eventual convergence without depending on perpetual
                // packet loss eventually producing a favorable run.
                let unstable_ticks = noprop::sample_usize_in(ctx, 1..=10_000);
                cluster.run(ctx, cluster.clock.after(unstable_ticks));
                let unstable_links = cluster.default_link_options.clone();
                cluster.default_link_options.drop_rate = noprop::Ratio::new(0, 1);
                cluster.default_link_options.latency_ticks = MinMax::new(1, 10);

                let terminal = cluster.run_until(ctx, cluster.clock.after(1_000_000), |cluster| {
                    observed_config_status(cluster, position, &expected).is_some()
                });
                let status = if terminal {
                    observed_config_status(&cluster, position, &expected)
                        .expect("the terminal condition was satisfied")
                } else {
                    let states: Vec<_> = cluster
                        .nodes
                        .iter()
                        .map(|node| {
                            format!(
                                "id={:?} role={:?} term={:?} commit={:?} last={:?} status={:?} \
                                 config={:?}",
                                node.inner.id(),
                                node.inner.role(),
                                node.inner.current_term(),
                                node.inner.commit_index(),
                                node.inner.log().last_position(),
                                node.inner.get_commit_status(position),
                                node.inner.config(),
                            )
                        })
                        .collect();
                    return Err(format!(
                        "config change {i} at {position:?} did not reach a terminal status; \
                         nodes={states:?}"
                    )
                    .into());
                };
                if cluster
                    .nodes
                    .iter()
                    .filter(|node| expected.voters.contains(&node.inner.id()))
                    .map(|node| node.inner.get_commit_status(position))
                    .any(|candidate| !candidate.is_in_progress() && candidate != status)
                {
                    return Err(format!(
                        "config change {i} has conflicting terminal statuses at {position:?}"
                    )
                    .into());
                }
                if status.is_committed() {
                    let finalized =
                        cluster.run_until(ctx, cluster.clock.after(1_000_000), |cluster| {
                            config_is_committed(cluster, &expected)
                        });
                    if !finalized {
                        return Err(format!(
                            "committed config change {i} did not produce the expected final config \
                             {expected:?}"
                        )
                        .into());
                    }
                    // A server removed from the committed configuration
                    // must be decommissioned. Leaving it active with a
                    // stale local config lets it keep starting disruptive
                    // elections even though it is no longer a member.
                    cluster
                        .nodes
                        .retain(|node| expected.contains(node.inner.id()));
                    cases_with_change.set(cases_with_change.get() + 1);
                } else if !status.is_rejected() {
                    return Err(format!(
                        "config change {i} ended with unexpected status {status:?}"
                    )
                    .into());
                } else if let Some(added_node) = added_node {
                    // A node introduced solely for a rejected proposal
                    // is not a cluster member and must not keep running
                    // with a stale copy of the proposed configuration.
                    cluster.nodes.retain(|node| node.inner.id() != added_node);
                }
                cluster.default_link_options = unstable_links;
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
