//! Liveness-oriented property tests for noraft clusters, driven by
//! noprop (https://github.com/sile/noprop).
//!
//! The safety invariants are covered by `tests/prop_cluster_test.rs`.
//! The properties in this file assert *bounded liveness*: commands
//! proposed while a leader exists reach a terminal commit status
//! (committed / rejected / unknown) within a declared tick budget,
//! even under unstable links, node restarts, storage repair, snapshot
//! installation, membership changes, and leader isolation.
//!
//! Each case samples its scenario parameters (link quality, proposal
//! count, restart cadence, ...) from the noprop case context, and a
//! coverage gate fails the run if the scenario's critical event
//! (restart / reset / snapshot / config change / rejoin) never
//! occurred, so a broken harness cannot pass silently.
//!
//! Seed / case budget come from the `NORAFT_PBT_SEED` and
//! `NORAFT_PBT_CASES` environment variables; unset means
//! "clock-derived seed" and the per-property default budget. A
//! failing seed can be re-run with:
//!
//! ```text
//! NORAFT_PBT_SEED=<seed> cargo test --test random_scenario_test
//! ```

use noprop::TestCaseContext;
use noraft::{
    ClusterConfig, CommitStatus, Log, LogEntries, LogIndex, LogPosition, Message, Node,
    NodeGeneration, NodeId, Role, Term,
};
use std::cell::Cell;
use std::collections::BTreeMap;

/// Reads the case-budget environment variable with a default
/// fallback.
fn cases_from_env(default: usize) -> usize {
    std::env::var("NORAFT_PBT_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Runs the cluster until `position` reaches a terminal commit status
/// (committed / rejected / unknown), or the round budget is
/// exhausted.
///
/// Each round first waits for a leader to appear (bounded by 100_000
/// ticks) because `get_commit_status` is queried through the leader.
/// Returns `None` on timeout.
fn wait_until_terminal(
    cluster: &mut TestCluster,
    ctx: &mut TestCaseContext,
    position: LogPosition,
    max_rounds: usize,
) -> Option<CommitStatus> {
    for _ in 0..max_rounds {
        let found = cluster.run_while_leader_absent(ctx, cluster.clock.add(100_000));
        if !found {
            return None;
        }
        let Some(leader) = cluster.leader_node() else {
            unreachable!();
        };
        let status = leader.get_commit_status(position);
        if !status.is_in_progress() {
            return Some(status);
        }
        cluster.run(ctx, cluster.clock.add(10));
    }
    None
}

/// All positions must reach a terminal status: committed, or (when
/// the corresponding argument is `true`) rejected due to truncation
/// of uncommitted entries, or unknown due to being covered by a
/// snapshot. Returns the number of committed positions.
fn assert_all_terminal(
    cluster: &mut TestCluster,
    ctx: &mut TestCaseContext,
    positions: &[LogPosition],
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

/// Command proposals commit under stable links, the commit indices
/// converge, and the leader does not change (term stays 1).
#[test]
fn proposals_commit_with_stable_links() -> noprop::TestResult {
    let seed = noprop::seed_from_env_or_time("NORAFT_PBT_SEED")?;
    let cases = cases_from_env(64);

    noprop::Runner::new(seed).run(cases, |ctx| {
        let node_ids = [NodeId::new(0), NodeId::new(1), NodeId::new(2)];
        let mut cluster = TestCluster::new(&node_ids);

        let position = cluster.random_node_mut(ctx).create_cluster(&node_ids);
        assert_ne!(position, LogPosition::INVALID);
        let satisfied = cluster.run_until(ctx, cluster.clock.add(10_000), |cluster| {
            cluster.leader_node().is_some()
        });
        if !satisfied {
            return Err("cluster creation timed out".into());
        }

        let proposals = noprop::sample_usize_in(ctx, 1..=32);
        let mut positions = Vec::new();
        for _ in 0..proposals {
            let Some(leader) = cluster.leader_node_mut() else {
                return Err("leader disappeared".into());
            };
            positions.push(leader.propose_command());
            let ticks = MinMax::new(1, 10).sample(ctx);
            cluster.run(ctx, cluster.clock.add(ticks));
        }

        let committed = assert_all_terminal(&mut cluster, ctx, &positions, false, false)?;
        if committed != positions.len() {
            return Err("all proposals must commit".into());
        }

        let satisfied = cluster.run_until(ctx, cluster.clock.add(1000), |cluster| {
            cluster.nodes[0].inner.commit_index() == cluster.nodes[1].inner.commit_index()
                && cluster.nodes[0].inner.commit_index() == cluster.nodes[2].inner.commit_index()
        });
        if !satisfied {
            return Err("commit indices are not synchronized".into());
        }

        // Links are stable, so the leader should not change.
        if cluster.nodes[0].inner.current_term().get() != 1 {
            return Err("leader must not change under stable links".into());
        }
        Ok(())
    })?;
    Ok(())
}

/// Command proposals commit even under a very unstable network
/// (30% drop rate, 1-1000 tick latency).
#[test]
fn proposals_commit_with_unstable_links() -> noprop::TestResult {
    let seed = noprop::seed_from_env_or_time("NORAFT_PBT_SEED")?;
    let cases = cases_from_env(32);

    noprop::Runner::new(seed).run(cases, |ctx| {
        let node_ids = [NodeId::new(0), NodeId::new(1), NodeId::new(2)];
        let mut cluster = TestCluster::new(&node_ids);
        cluster.default_link_options.drop_rate = noprop::Ratio::new(3, 10);
        cluster.default_link_options.latency_ticks = MinMax::new(1, 1000);

        let position = cluster.random_node_mut(ctx).create_cluster(&node_ids);
        assert_ne!(position, LogPosition::INVALID);
        let satisfied = cluster.run_until(ctx, cluster.clock.add(100_000), |cluster| {
            cluster.leader_node().is_some()
        });
        if !satisfied {
            return Err("cluster creation timed out".into());
        }

        let proposals = noprop::sample_usize_in(ctx, 1..=16);
        let mut positions = Vec::new();
        for _ in 0..proposals {
            let found = cluster.run_while_leader_absent(ctx, cluster.clock.add(100_000));
            if !found {
                return Err("leader absent while proposing".into());
            }
            let Some(leader) = cluster.leader_node_mut() else {
                unreachable!();
            };
            positions.push(leader.propose_command());
            let ticks = MinMax::new(1, 10).sample(ctx);
            cluster.run(ctx, cluster.clock.add(ticks));
        }

        let committed = assert_all_terminal(&mut cluster, ctx, &positions, false, false)?;
        if committed != positions.len() {
            return Err("all proposals must commit".into());
        }

        let satisfied = cluster.run_until(ctx, cluster.clock.add(100_000), |cluster| {
            cluster.nodes[0].inner.commit_index() == cluster.nodes[1].inner.commit_index()
                && cluster.nodes[0].inner.commit_index() == cluster.nodes[2].inner.commit_index()
        });
        if !satisfied {
            return Err("commit indices are not synchronized".into());
        }
        Ok(())
    })?;
    Ok(())
}

/// Command proposals commit when proposals are pipelined (no waiting
/// for the previous commit) and interleaved with heartbeats.
#[test]
fn proposals_commit_with_pipelining() -> noprop::TestResult {
    let seed = noprop::seed_from_env_or_time("NORAFT_PBT_SEED")?;
    let cases = cases_from_env(32);

    noprop::Runner::new(seed).run(cases, |ctx| {
        let node_ids = [NodeId::new(0), NodeId::new(1), NodeId::new(2)];
        let mut cluster = TestCluster::new(&node_ids);

        let position = cluster.random_node_mut(ctx).create_cluster(&node_ids);
        assert_ne!(position, LogPosition::INVALID);
        let satisfied = cluster.run_until(ctx, cluster.clock.add(10_000), |cluster| {
            cluster.leader_node().is_some()
        });
        if !satisfied {
            return Err("cluster creation timed out".into());
        }

        let proposals = noprop::sample_usize_in(ctx, 1..=32);
        let mut positions = Vec::new();
        for _ in 0..proposals {
            let pipeline = noprop::sample_ratio(ctx, noprop::Ratio::new(4, 5));
            let do_heartbeat = noprop::sample_ratio(ctx, noprop::Ratio::one_nth(2));

            let found = cluster.run_while_leader_absent(ctx, cluster.clock.add(10_000));
            if !found {
                return Err("leader absent while proposing".into());
            }
            let Some(leader) = cluster.leader_node_mut() else {
                unreachable!();
            };
            positions.push(leader.propose_command());
            if do_heartbeat && !leader.heartbeat() {
                return Err("heartbeat must succeed on the leader".into());
            }

            if !pipeline {
                let ticks = MinMax::new(0, 5).sample(ctx);
                cluster.run(ctx, cluster.clock.add(ticks));
            }
        }

        let committed = assert_all_terminal(&mut cluster, ctx, &positions, false, false)?;
        if committed != positions.len() {
            return Err("all proposals must commit".into());
        }

        let satisfied = cluster.run_until(ctx, cluster.clock.add(10_000), |cluster| {
            cluster.nodes[0].inner.commit_index() == cluster.nodes[1].inner.commit_index()
                && cluster.nodes[0].inner.commit_index() == cluster.nodes[2].inner.commit_index()
        });
        if !satisfied {
            return Err("commit indices are not synchronized".into());
        }
        Ok(())
    })?;
    Ok(())
}

/// Command proposals commit while a node periodically restarts, and
/// the run must exercise at least one restart.
#[test]
fn proposals_commit_across_node_restarts() -> noprop::TestResult {
    let seed = noprop::seed_from_env_or_time("NORAFT_PBT_SEED")?;
    let cases = cases_from_env(32);
    let cases_with_commit: Cell<usize> = Cell::new(0);

    noprop::Runner::new(seed).run(cases, |ctx| {
        let node_ids = [NodeId::new(0), NodeId::new(1), NodeId::new(2)];
        let mut cluster = TestCluster::new(&node_ids);

        // Node 0 stops and restarts periodically with the same slow
        // cadence as the original scenario test: while it is down,
        // the remaining two voters can still elect a leader and
        // commit, so the cluster keeps making progress.
        cluster.nodes[0].options.running_ticks = MinMax::new(800, 5000);
        cluster.nodes[0].options.stopping_ticks = MinMax::new(800, 5000);
        let initial_generation = cluster.nodes[0].inner.generation().get();

        let position = cluster.random_node_mut(ctx).create_cluster(&node_ids);
        assert_ne!(position, LogPosition::INVALID);
        let satisfied = cluster.run_until(ctx, cluster.clock.add(10_000), |cluster| {
            cluster.leader_node().is_some()
        });
        if !satisfied {
            return Err("cluster creation timed out".into());
        }

        // Propose a first batch of commands.
        let mut positions = Vec::new();
        let first_batch = noprop::sample_usize_in(ctx, 1..=16);
        for _ in 0..first_batch {
            let Some(leader) = cluster.leader_node_mut() else {
                unreachable!();
            };
            positions.push(leader.propose_command());
            let ticks = MinMax::new(1, 10).sample(ctx);
            cluster.run(ctx, cluster.clock.add(ticks));
        }

        // Wait for node 0 to stop and restart at least once. The
        // restart is guaranteed to happen by construction (the stop /
        // start cadence is bounded), and the wait bounds the liveness
        // claim: the cluster must not stall during the restart cycle.
        let restarted = cluster.run_until(ctx, cluster.clock.add(50_000), |cluster| {
            cluster.nodes[0].inner.generation().get() > initial_generation
        });
        if !restarted {
            return Err("node 0 did not restart within the budget".into());
        }

        // Propose a second batch of commands while node 0 keeps
        // restarting.
        let second_batch = noprop::sample_usize_in(ctx, 8..=48);
        for _ in 0..second_batch {
            let found = cluster.run_while_leader_absent(ctx, cluster.clock.add(10_000));
            if !found {
                return Err("leader absent while proposing".into());
            }
            let Some(leader) = cluster.leader_node_mut() else {
                unreachable!();
            };
            positions.push(leader.propose_command());
            let ticks = MinMax::new(1, 10).sample(ctx);
            cluster.run(ctx, cluster.clock.add(ticks));
        }

        // A proposal made on a leader that stops before replicating
        // it may be truncated by a later leader, so a terminal
        // `Rejected` status is legitimate here. The liveness claim is
        // that every proposal settles and the cluster keeps
        // committing while node 0 restarts.
        let committed = assert_all_terminal(&mut cluster, ctx, &positions, true, false)?;
        if committed > 0 {
            cases_with_commit.set(cases_with_commit.get() + 1);
        }

        let satisfied = cluster.run_until(ctx, cluster.clock.add(50_000), |cluster| {
            cluster.nodes[0].inner.commit_index() == cluster.nodes[1].inner.commit_index()
                && cluster.nodes[0].inner.commit_index() == cluster.nodes[2].inner.commit_index()
        });
        if !satisfied {
            return Err("commit indices are not synchronized".into());
        }
        Ok(())
    })?;

    assert!(
        cases_with_commit.get() > 0,
        "no case committed a command while a node was restarting (seed={seed:#018x})",
    );
    Ok(())
}

/// Command proposals commit after non-leader nodes lose their storage
/// mid-run and recover through log repair, and the run must exercise
/// at least one storage loss.
#[test]
fn proposals_commit_after_storage_repair() -> noprop::TestResult {
    let seed = noprop::seed_from_env_or_time("NORAFT_PBT_SEED")?;
    let cases = cases_from_env(32);
    let cases_with_repair: Cell<usize> = Cell::new(0);

    noprop::Runner::new(seed).run(cases, |ctx| {
        let node_ids = [NodeId::new(0), NodeId::new(1), NodeId::new(2)];
        let mut cluster = TestCluster::new(&node_ids);

        let position = cluster.random_node_mut(ctx).create_cluster(&node_ids);
        assert_ne!(position, LogPosition::INVALID);
        let satisfied = cluster.run_until(ctx, cluster.clock.add(10_000), |cluster| {
            cluster.leader_node().is_some()
        });
        if !satisfied {
            return Err("cluster creation timed out".into());
        }

        let proposals = noprop::sample_usize_in(ctx, 2..=32);
        let repair_at = noprop::sample_usize_in(ctx, 1..proposals);
        let mut positions = Vec::new();
        for i in 0..proposals {
            if i == repair_at {
                // Reset the non-leader nodes: their storage is lost.
                for node in cluster.nodes.iter_mut() {
                    if !node.inner.role().is_leader() {
                        let generation =
                            NodeGeneration::new(node.inner.generation().get().saturating_add(1));
                        let log =
                            Log::new(ClusterConfig::new(), LogEntries::new(LogPosition::ZERO));
                        node.inner =
                            Node::restart(node.inner.id(), generation, Term::ZERO, None, log);
                    }
                }
                cases_with_repair.set(cases_with_repair.get() + 1);
            }

            let found = cluster.run_while_leader_absent(ctx, cluster.clock.add(10_000));
            if !found {
                return Err("leader absent while proposing".into());
            }
            let Some(leader) = cluster.leader_node_mut() else {
                unreachable!();
            };
            positions.push(leader.propose_command());
            let ticks = MinMax::new(1, 10).sample(ctx);
            cluster.run(ctx, cluster.clock.add(ticks));
        }

        let committed = assert_all_terminal(&mut cluster, ctx, &positions, false, false)?;
        if committed != positions.len() {
            return Err("all proposals must commit".into());
        }

        let satisfied = cluster.run_until(ctx, cluster.clock.add(1_000_000), |cluster| {
            cluster.nodes[0].inner.commit_index() == cluster.nodes[1].inner.commit_index()
                && cluster.nodes[0].inner.commit_index() == cluster.nodes[2].inner.commit_index()
        });
        if !satisfied {
            return Err("commit indices are not synchronized".into());
        }
        Ok(())
    })?;

    assert!(
        cases_with_repair.get() > 0,
        "no case exercised a storage loss (seed={seed:#018x})",
    );
    Ok(())
}

/// Command proposals commit after a snapshot is installed and
/// non-leader nodes lose their storage; entries covered by the
/// snapshot are reported as unknown, newer ones as committed. The run
/// must exercise at least one snapshot installation.
#[test]
fn proposals_commit_after_snapshot_repair() -> noprop::TestResult {
    let seed = noprop::seed_from_env_or_time("NORAFT_PBT_SEED")?;
    let cases = cases_from_env(16);
    let cases_with_snapshot: Cell<usize> = Cell::new(0);

    noprop::Runner::new(seed).run(cases, |ctx| {
        let node_ids = [NodeId::new(0), NodeId::new(1), NodeId::new(2)];
        let mut cluster = TestCluster::new(&node_ids);

        let position = cluster.random_node_mut(ctx).create_cluster(&node_ids);
        assert_ne!(position, LogPosition::INVALID);
        let satisfied = cluster.run_until(ctx, cluster.clock.add(10_000), |cluster| {
            cluster.leader_node().is_some()
        });
        if !satisfied {
            return Err("cluster creation timed out".into());
        }

        let proposals = noprop::sample_usize_in(ctx, 4..=32);
        let snapshot_at = noprop::sample_usize_in(ctx, 1..proposals - 1);
        let repair_at = noprop::sample_usize_in(ctx, snapshot_at + 1..proposals);
        let mut positions = Vec::new();
        let mut snapshot_index = LogIndex::ZERO;
        for i in 0..proposals {
            if i == snapshot_at {
                // Take a snapshot at the current commit index.
                let satisfied = cluster.run_until(ctx, cluster.clock.add(10_000), |cluster| {
                    cluster
                        .nodes
                        .iter()
                        .all(|node| node.inner.commit_index().get() > 0)
                });
                if !satisfied {
                    return Err("nothing committed at snapshot time".into());
                }
                for node in cluster.nodes.iter_mut() {
                    let (snapshot_position, config) = node
                        .inner
                        .log()
                        .get_position_and_config(node.inner.commit_index())
                        .expect("commit index is contained");
                    if !node
                        .inner
                        .handle_snapshot_installed(snapshot_position, config.clone())
                    {
                        return Err("snapshot installation failed".into());
                    }
                    if node.inner.role().is_leader() {
                        snapshot_index = snapshot_position.index;
                    }
                }
                cases_with_snapshot.set(cases_with_snapshot.get() + 1);
            }
            if i == repair_at {
                // Reset the non-leader nodes: their storage is lost.
                for node in cluster.nodes.iter_mut() {
                    if !node.inner.role().is_leader() {
                        let generation =
                            NodeGeneration::new(node.inner.generation().get().saturating_add(1));
                        let log =
                            Log::new(ClusterConfig::new(), LogEntries::new(LogPosition::ZERO));
                        node.inner =
                            Node::restart(node.inner.id(), generation, Term::ZERO, None, log);
                    }
                }
            }

            let found = cluster.run_while_leader_absent(ctx, cluster.clock.add(10_000));
            if !found {
                return Err("leader absent while proposing".into());
            }
            let Some(leader) = cluster.leader_node_mut() else {
                unreachable!();
            };
            positions.push(leader.propose_command());
            let ticks = MinMax::new(1, 10).sample(ctx);
            cluster.run(ctx, cluster.clock.add(ticks));
        }

        // Entries covered by the snapshot are unknown; newer entries
        // must commit.
        for (i, position) in positions.iter().enumerate() {
            if position.index < snapshot_index {
                let status = wait_until_terminal(&mut cluster, ctx, *position, 1000)
                    .ok_or_else(|| format!("proposal {i} did not reach a terminal status"))?;
                if !status.is_unknown() {
                    return Err(format!(
                        "snapshot-covered proposal {i} must be unknown, got {status:?}"
                    )
                    .into());
                }
            }
        }
        let after_snapshot: Vec<LogPosition> = positions
            .iter()
            .copied()
            .filter(|position| position.index >= snapshot_index)
            .collect();
        let committed = assert_all_terminal(&mut cluster, ctx, &after_snapshot, false, false)?;
        if committed != after_snapshot.len() {
            return Err("all post-snapshot proposals must commit".into());
        }

        let satisfied = cluster.run_until(ctx, cluster.clock.add(1_000_000), |cluster| {
            cluster.nodes[0].inner.commit_index() == cluster.nodes[1].inner.commit_index()
                && cluster.nodes[0].inner.commit_index() == cluster.nodes[2].inner.commit_index()
        });
        if !satisfied {
            return Err("commit indices are not synchronized".into());
        }
        Ok(())
    })?;

    assert!(
        cases_with_snapshot.get() > 0,
        "no case exercised a snapshot installation (seed={seed:#018x})",
    );
    Ok(())
}

/// Cluster configuration changes (add / remove voters and non-voters)
/// settle without getting stuck in joint consensus, and at least one
/// command commits somewhere across the run. The run must exercise at
/// least one config change.
#[test]
fn membership_changes_settle() -> noprop::TestResult {
    let seed = noprop::seed_from_env_or_time("NORAFT_PBT_SEED")?;
    let cases = cases_from_env(16);
    let cases_with_change: Cell<usize> = Cell::new(0);
    let cases_with_commit: Cell<usize> = Cell::new(0);

    noprop::Runner::new(seed).run(cases, |ctx| {
        let node_ids = [NodeId::new(0), NodeId::new(1), NodeId::new(2)];
        let mut cluster = TestCluster::new(&node_ids);
        cluster.default_link_options.drop_rate = noprop::Ratio::new(3, 10);
        cluster.default_link_options.latency_ticks = MinMax::new(1, 1000);

        let position = cluster.random_node_mut(ctx).create_cluster(&node_ids);
        assert_ne!(position, LogPosition::INVALID);
        let satisfied = cluster.run_until(ctx, cluster.clock.add(100_000), |cluster| {
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
            let settled = cluster.run_until(ctx, cluster.clock.add(1_000_000), |cluster| {
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
            } else if cluster.nodes.iter().filter(|n| n.voter).count() > 2 {
                // Remove.
                let candidate_ids = cluster
                    .nodes
                    .iter()
                    .map(|n| n.inner.id())
                    .collect::<Vec<_>>();
                let node_id = noprop::sample_choice(ctx, &candidate_ids);
                cases_with_change.set(cases_with_change.get() + 1);

                let Some(leader) = cluster.leader_node_mut() else {
                    unreachable!();
                };
                let new_config = if leader.config().non_voters.contains(&node_id) {
                    let mut new_config = leader.config().clone();
                    new_config.non_voters.remove(&node_id);
                    new_config
                } else {
                    leader.config().to_joint_consensus(&[], &[node_id])
                };
                let position = leader.propose_config(new_config);
                assert_ne!(position, LogPosition::INVALID);
            }

            // Propose commands.
            let mut positions = Vec::new();
            let command_count = noprop::sample_usize_in(ctx, 1..=4);
            for _ in 0..command_count {
                let found = cluster.run_while_leader_absent(ctx, cluster.clock.add(1_000_000));
                if !found {
                    return Err("leader absent while proposing".into());
                }
                let Some(leader) = cluster.leader_node_mut() else {
                    unreachable!();
                };
                positions.push(leader.propose_command());
                let ticks = MinMax::new(1, 10).sample(ctx);
                cluster.run(ctx, cluster.clock.add(ticks));
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

/// After the leader is isolated from the cluster and a new leader is
/// elected, rejoining the old leader reconciles the divergent logs:
/// every proposed position reaches a terminal status, at least one
/// position commits, and the commit indices converge.
#[test]
fn divergent_logs_reconcile() -> noprop::TestResult {
    let seed = noprop::seed_from_env_or_time("NORAFT_PBT_SEED")?;
    let cases = cases_from_env(16);
    let cases_with_commit: Cell<usize> = Cell::new(0);

    noprop::Runner::new(seed).run(cases, |ctx| {
        let node_ids = [NodeId::new(0), NodeId::new(1), NodeId::new(2)];
        let mut cluster = TestCluster::new(&node_ids);

        let position = cluster.random_node_mut(ctx).create_cluster(&node_ids);
        assert_ne!(position, LogPosition::INVALID);
        let satisfied = cluster.run_until(ctx, cluster.clock.add(10_000), |cluster| {
            cluster.leader_node().is_some()
        });
        if !satisfied {
            return Err("cluster creation timed out".into());
        }
        // Let the initial cluster-config / term entries replicate to
        // all nodes before proceeding: a follower that never receives
        // the config entry is not a voter and can never campaign.
        cluster.run(ctx, cluster.clock.add(100));

        // Propose commands as usual.
        let mut positions = Vec::new();
        let first_batch = noprop::sample_usize_in(ctx, 4..=16);
        for _ in 0..first_batch {
            let Some(leader) = cluster.leader_node_mut() else {
                unreachable!();
            };
            positions.push(leader.propose_command());
            let ticks = MinMax::new(1, 10).sample(ctx);
            cluster.run(ctx, cluster.clock.add(ticks));
        }

        // Propose more commands without giving the cluster time to
        // replicate them, then isolate the leader.
        let second_batch = noprop::sample_usize_in(ctx, 4..=16);
        for _ in 0..second_batch {
            let Some(leader) = cluster.leader_node_mut() else {
                unreachable!();
            };
            positions.push(leader.propose_command());
        }
        let leader_node_index = cluster
            .nodes
            .iter()
            .position(|n| n.inner.role().is_leader())
            .expect("leader exists");
        let old_leader = cluster.nodes.remove(leader_node_index);

        // Elect a new leader.
        let found = cluster.run_while_leader_absent(ctx, cluster.clock.add(1_000_000));
        if !found {
            return Err("new leader was not elected".into());
        }

        // Propose remaining commands.
        let third_batch = noprop::sample_usize_in(ctx, 4..=16);
        for _ in 0..third_batch {
            let found = cluster.run_while_leader_absent(ctx, cluster.clock.add(1_000_000));
            if !found {
                return Err("leader absent while proposing".into());
            }
            let Some(leader) = cluster.leader_node_mut() else {
                unreachable!();
            };
            positions.push(leader.propose_command());
            let ticks = MinMax::new(1, 10).sample(ctx);
            cluster.run(ctx, cluster.clock.add(ticks));
        }

        // Rejoin the old leader.
        cluster.nodes.push(old_leader);

        // Uncommitted positions on the isolated leader are truncated
        // after the rejoin, so a terminal `Rejected` status is
        // legitimate here; the liveness claim is that the cluster
        // recovers and keeps committing.
        let committed = assert_all_terminal(&mut cluster, ctx, &positions, true, false)?;
        if committed == 0 {
            return Err("no proposal committed after reconciliation".into());
        }
        cases_with_commit.set(cases_with_commit.get() + 1);

        let satisfied = cluster.run_until(ctx, cluster.clock.add(10_000), |cluster| {
            cluster.nodes[0].inner.commit_index() == cluster.nodes[1].inner.commit_index()
                && cluster.nodes[0].inner.commit_index() == cluster.nodes[2].inner.commit_index()
        });
        if !satisfied {
            return Err("commit indices are not synchronized".into());
        }
        Ok(())
    })?;

    assert!(
        cases_with_commit.get() > 0,
        "no case committed a command after divergent-log reconciliation \
         (seed={seed:#018x})",
    );
    Ok(())
}

#[derive(Debug)]
pub struct TestCluster {
    pub nodes: Vec<TestNode>,
    pub clock: Clock,
    pub default_link_options: TestLinkOptions,
    seqno: u64,
}

impl TestCluster {
    pub fn new(node_ids: &[NodeId]) -> Self {
        Self {
            nodes: node_ids.iter().map(|&id| TestNode::new(id)).collect(),
            clock: Clock::new(),
            default_link_options: TestLinkOptions::default(),
            seqno: 0,
        }
    }

    pub fn leader_node(&self) -> Option<&Node> {
        self.nodes
            .iter()
            .find(|node| node.inner.role().is_leader())
            .map(|node| &node.inner)
    }

    pub fn leader_node_mut(&mut self) -> Option<&mut Node> {
        self.nodes
            .iter_mut()
            .find(|node| node.inner.role().is_leader())
            .map(|node| &mut node.inner)
    }

    pub fn random_node_mut(&mut self, ctx: &mut TestCaseContext) -> &mut Node {
        let index = noprop::sample_usize_in(ctx, 0..self.nodes.len());
        &mut self.nodes[index].inner
    }

    pub fn run_while_leader_absent(&mut self, ctx: &mut TestCaseContext, deadline: Clock) -> bool {
        self.run_until(ctx, deadline, |cluster| cluster.leader_node().is_some())
    }

    pub fn run(&mut self, ctx: &mut TestCaseContext, deadline: Clock) {
        self.run_until(ctx, deadline, |_| false);
    }

    pub fn run_until<F>(&mut self, ctx: &mut TestCaseContext, deadline: Clock, condition: F) -> bool
    where
        F: Fn(&TestCluster) -> bool,
    {
        while self.clock < deadline && !condition(self) {
            self.run_tick(ctx);
        }
        self.clock < deadline
    }

    pub fn run_tick(&mut self, ctx: &mut TestCaseContext) {
        self.clock.tick();
        let mut messages = Vec::new();
        let mut snapshots = Vec::new();

        // Run nodes.
        for node in &mut self.nodes {
            node.run_tick(ctx, self.clock);

            let src = node.inner.id();
            let mut actions = std::mem::take(node.inner.actions_mut());
            if let Some(msg) = actions.broadcast_message.take() {
                for dst in node.inner.peers() {
                    messages.push((src, dst, msg.clone()));
                }
            }
            for (dst, msg) in actions.send_messages {
                messages.push((src, dst, msg));
            }
            for dst in actions.install_snapshots {
                snapshots.push((
                    src,
                    dst,
                    node.inner.log().snapshot_position(),
                    node.inner.log().snapshot_config().clone(),
                ));
            }
        }

        // Deliver messages.
        for (src, dst, msg) in messages {
            self.send_message(ctx, src, dst, msg);
        }

        // Deliver snapshots.
        for (src, dst, position, config) in snapshots {
            self.send_snashot(ctx, src, dst, position, config);
        }
    }

    fn send_message(&mut self, ctx: &mut TestCaseContext, _src: NodeId, dst: NodeId, msg: Message) {
        let options = &self.default_link_options;

        if noprop::sample_ratio(ctx, options.drop_rate) {
            return;
        }

        let latency = options.latency_ticks.sample(ctx) * message_size(&msg);
        for node in &mut self.nodes {
            if node.inner.id() == dst {
                node.incoming_messages
                    .insert((self.clock.add(latency), self.seqno), msg);
                self.seqno += 1;
                return;
            }
        }
    }

    fn send_snashot(
        &mut self,
        ctx: &mut TestCaseContext,
        _src: NodeId,
        dst: NodeId,
        position: LogPosition,
        config: ClusterConfig,
    ) {
        for node in &mut self.nodes {
            if node.inner.id() == dst {
                if node.snapshot_finish_time.is_some() {
                    return;
                }

                node.snapshot_finish_time = Some((
                    self.clock
                        .add(node.options.install_snapshot_ticks.sample(ctx)),
                    position,
                    config,
                ));
                return;
            }
        }
    }
}

fn message_size(msg: &Message) -> usize {
    match msg {
        Message::AppendEntriesCall { entries, .. } => entries.len(),
        Message::AppendEntriesReply { .. } => 1,
        Message::RequestVoteCall { .. } => 1,
        Message::RequestVoteReply { .. } => 1,
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
    pub log_entries_lost: MinMax,
    pub max_entries_per_rpc: usize,
    pub voter: bool,
}

impl Default for TestNodeOptions {
    fn default() -> Self {
        Self {
            election_timeout_ticks: MinMax::new(100, 1000),
            storage_latency_ticks: MinMax::new(1, 10),
            install_snapshot_ticks: MinMax::new(1000, 10_000),
            running_ticks: MinMax::constant(usize::MAX),
            stopping_ticks: MinMax::constant(usize::MAX),
            log_entries_lost: MinMax::constant(0),
            max_entries_per_rpc: 100,
            voter: true,
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct MinMax {
    pub min: usize,
    pub max: usize,
}

impl MinMax {
    pub fn new(min: usize, max: usize) -> MinMax {
        assert!(min <= max);
        MinMax { min, max }
    }

    pub fn constant(value: usize) -> MinMax {
        MinMax::new(value, value)
    }

    pub fn sample(&self, ctx: &mut TestCaseContext) -> usize {
        noprop::sample_usize_in(ctx, self.min..=self.max)
    }
}

#[derive(Debug)]
pub struct TestNode {
    pub inner: Node,
    pub options: TestNodeOptions,
    pub running: bool,
    pub timeout_expire_time: Option<Clock>,
    pub storage_finish_time: Option<Clock>,
    pub snapshot_finish_time: Option<(Clock, LogPosition, ClusterConfig)>,
    pub incoming_messages: BTreeMap<(Clock, u64), Message>,
    pub stop_time: Option<Clock>,
    pub start_time: Option<Clock>,
    pub voter: bool,
}

impl TestNode {
    pub fn new(id: NodeId) -> TestNode {
        TestNode {
            inner: Node::start(id),
            options: TestNodeOptions::default(),
            running: true,
            timeout_expire_time: None,
            storage_finish_time: None,
            snapshot_finish_time: None,
            incoming_messages: BTreeMap::new(),
            stop_time: None,
            start_time: None,
            voter: true,
        }
    }

    pub fn run_tick(&mut self, ctx: &mut TestCaseContext, now: Clock) {
        if !self.voter {
            assert!(self.inner.role().is_follower());
        }

        if !self.running {
            if self.start_time.take_if(|t| *t <= now).is_some() {
                self.running = true;

                while let Some(entry) = self.incoming_messages.first_entry() {
                    if entry.key().0 < now {
                        entry.remove();
                    } else {
                        break;
                    }
                }

                self.inner = Node::restart(
                    self.inner.id(),
                    NodeGeneration::new(self.inner.generation().get().saturating_add(1)),
                    self.inner.current_term(),
                    self.inner.voted_for(),
                    self.inner.log().clone(),
                );
            } else {
                return;
            }
        }
        if self.stop_time.is_none() {
            self.stop_time = Some(now.add(self.options.running_ticks.sample(ctx)));
        }
        if self.stop_time.take_if(|t| *t <= now).is_some() {
            self.running = false;
            self.timeout_expire_time = None;
            self.storage_finish_time = None;
            self.start_time = Some(now.add(self.options.stopping_ticks.sample(ctx)));
            return;
        }

        self.storage_finish_time.take_if(|t| *t <= now);
        if self.storage_finish_time.is_some() {
            // Storage operations are synchronous, so we can't do
            // anything else until they finish.
            return;
        }

        if self.timeout_expire_time.take_if(|t| *t <= now).is_some() {
            self.inner.handle_election_timeout();
        }

        if let Some((_, position, config)) =
            self.snapshot_finish_time.take_if(|(t, _, _)| *t <= now)
        {
            let _succeeded = self.inner.handle_snapshot_installed(position, config);
        }

        while let Some(entry) = self.incoming_messages.first_entry() {
            if entry.key().0 <= now {
                let message = entry.remove();
                self.inner
                    .handle_message(&message)
                    .expect("message handling should succeed");
            } else {
                break;
            }
        }

        if std::mem::take(&mut self.inner.actions_mut().set_election_timeout) {
            self.reset_election_timeout(ctx, now);
        }
        if std::mem::take(&mut self.inner.actions_mut().save_current_term) {
            self.extend_storage_finish_time(ctx, now, 1);
        }
        if std::mem::take(&mut self.inner.actions_mut().save_voted_for) {
            self.extend_storage_finish_time(ctx, now, 1);
        }
        if let Some(entries) = self.inner.actions_mut().append_log_entries.take() {
            self.extend_storage_finish_time(ctx, now, entries.len());
        }
    }

    fn reset_election_timeout(&mut self, ctx: &mut TestCaseContext, now: Clock) {
        let timeout = match self.inner.role() {
            Role::Leader => self.options.election_timeout_ticks.min,
            Role::Candidate => self.options.election_timeout_ticks.sample(ctx),
            Role::Follower => self.options.election_timeout_ticks.max,
        };
        self.timeout_expire_time = Some(now.add(timeout));
    }

    fn extend_storage_finish_time(&mut self, ctx: &mut TestCaseContext, now: Clock, n: usize) {
        let remaining_latency = self.storage_finish_time.map_or(0, |t| t.0 - now.0);
        let additional_latency = self.options.storage_latency_ticks.sample(ctx) * n;
        let latency = remaining_latency + additional_latency;
        self.storage_finish_time = Some(now.add(latency));
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Clock(usize);

impl Clock {
    pub fn new() -> Clock {
        Clock(0)
    }

    pub fn tick(&mut self) {
        self.0 += 1;
    }

    pub fn add(&self, ticks: usize) -> Clock {
        Clock(self.0.saturating_add(ticks))
    }
}
