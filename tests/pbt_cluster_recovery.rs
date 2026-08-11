//! Recovery PBTs for restarts, storage loss, snapshots, and divergent logs.
//!
//! Every test obtains its seed through `noprop::seed_from_env_or_time`
//! via the shared runner. CI therefore uses a fresh time-derived seed
//! unless `NORAFT_PBT_SEED` is explicitly set for reproduction.

pub mod helpers;

use helpers::pbt::{run, run_config};
use helpers::pbt_scenario::{MinMax, TestCluster, assert_all_terminal, wait_until_terminal};
use noraft::{LogIndex, LogPosition, NodeId};
use std::cell::Cell;

fn check_snapshot_compatible_logs(cluster: &TestCluster) -> Result<(), String> {
    for left_index in 0..cluster.nodes.len() {
        for right_index in left_index + 1..cluster.nodes.len() {
            let left = cluster.nodes[left_index].inner.log();
            let right = cluster.nodes[right_index].inner.log();
            let shared_anchor = left
                .snapshot_position()
                .index
                .max(right.snapshot_position().index);
            if left.entries().get_term(shared_anchor) != right.entries().get_term(shared_anchor) {
                return Err(format!(
                    "snapshot anchors disagree for nodes {left_index} and {right_index} at \
                     {shared_anchor:?}: left={left:?}, right={right:?}"
                ));
            }

            let shared_last = left.last_position().index.min(right.last_position().index);
            let mut index = shared_anchor;
            while index < shared_last {
                index = index.next();
                let left_entry = (
                    left.entries().get_term(index),
                    left.entries().get_entry(index),
                );
                let right_entry = (
                    right.entries().get_term(index),
                    right.entries().get_entry(index),
                );
                if left_entry != right_entry {
                    return Err(format!(
                        "log suffixes disagree for nodes {left_index} and {right_index} at \
                         {index:?}: left={left_entry:?}, right={right_entry:?}"
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Command proposals commit while a node periodically restarts, and
/// the run must exercise at least one restart.
#[test]
fn proposals_commit_across_node_restarts() -> noprop::TestResult {
    run(32, |ctx| {
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
        let satisfied = cluster.run_until(ctx, cluster.clock.after(10_000), |cluster| {
            cluster.leader_node().is_some()
        });
        if !satisfied {
            return Err("cluster creation timed out".into());
        }

        // Propose a first batch of commands.
        let mut before_restart_positions = Vec::new();
        let first_batch = noprop::sample_usize_in(ctx, 1..=16);
        for _ in 0..first_batch {
            let Some(leader) = cluster.leader_node_mut() else {
                unreachable!();
            };
            before_restart_positions.push(leader.propose_command());
            let ticks = MinMax::new(1, 10).sample(ctx);
            cluster.run(ctx, cluster.clock.after(ticks));
        }

        // Wait for node 0 to stop and restart at least once. The
        // restart is guaranteed to happen by construction (the stop /
        // start cadence is bounded), and the wait bounds the liveness
        // claim: the cluster must not stall during the restart cycle.
        let restarted = cluster.run_until(ctx, cluster.clock.after(50_000), |cluster| {
            cluster.nodes[0].inner.generation().get() > initial_generation
        });
        if !restarted {
            return Err("node 0 did not restart within the budget".into());
        }

        // Propose a second batch of commands while node 0 keeps
        // restarting.
        let second_batch = noprop::sample_usize_in(ctx, 8..=48);
        let mut after_restart_positions = Vec::new();
        for _ in 0..second_batch {
            let found = cluster.run_while_leader_absent(ctx, cluster.clock.after(10_000));
            if !found {
                return Err("leader absent while proposing".into());
            }
            let Some(leader) = cluster.leader_node_mut() else {
                unreachable!();
            };
            after_restart_positions.push(leader.propose_command());
            let ticks = MinMax::new(1, 10).sample(ctx);
            cluster.run(ctx, cluster.clock.after(ticks));
        }

        // A proposal made on a leader that stops before replicating
        // it may be truncated by a later leader, so a terminal
        // `Rejected` status is legitimate here. The liveness claim is
        // that every proposal settles and the cluster keeps
        // committing while node 0 restarts.
        assert_all_terminal(&mut cluster, ctx, &before_restart_positions, true, false)?;
        let committed_after_restart =
            assert_all_terminal(&mut cluster, ctx, &after_restart_positions, true, false)?;
        if committed_after_restart == 0 {
            return Err("no post-restart proposal committed".into());
        }

        let satisfied = cluster.run_until(ctx, cluster.clock.after(50_000), |cluster| {
            cluster.nodes[0].inner.commit_index() == cluster.nodes[1].inner.commit_index()
                && cluster.nodes[0].inner.commit_index() == cluster.nodes[2].inner.commit_index()
        });
        if !satisfied {
            return Err("commit indices are not synchronized".into());
        }
        Ok(())
    })
}

/// Command proposals commit after non-leader nodes lose their storage
/// mid-run and recover through log repair, and the run must exercise
/// at least one storage loss.
#[test]
fn proposals_commit_after_storage_repair() -> noprop::TestResult {
    let config = run_config(32)?;
    let seed = config.seed;
    let cases_with_repair: Cell<usize> = Cell::new(0);

    noprop::Runner::new(seed).run(config.cases, |ctx| {
        let node_ids = [NodeId::new(0), NodeId::new(1), NodeId::new(2)];
        let mut cluster = TestCluster::new(&node_ids);

        let position = cluster.random_node_mut(ctx).create_cluster(&node_ids);
        assert_ne!(position, LogPosition::INVALID);
        let satisfied = cluster.run_until(ctx, cluster.clock.after(10_000), |cluster| {
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
                        node.lose_storage();
                    }
                }
                cases_with_repair.set(cases_with_repair.get() + 1);
            }

            let found = cluster.run_while_leader_absent(ctx, cluster.clock.after(10_000));
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

        let committed = assert_all_terminal(&mut cluster, ctx, &positions, false, false)?;
        if committed != positions.len() {
            return Err("all proposals must commit".into());
        }

        let satisfied = cluster.run_until(ctx, cluster.clock.after(1_000_000), |cluster| {
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
    let config = run_config(16)?;
    let seed = config.seed;
    let cases_with_snapshot: Cell<usize> = Cell::new(0);

    noprop::Runner::new(seed).run(config.cases, |ctx| {
        let node_ids = [NodeId::new(0), NodeId::new(1), NodeId::new(2)];
        let mut cluster = TestCluster::new(&node_ids);

        let position = cluster.random_node_mut(ctx).create_cluster(&node_ids);
        assert_ne!(position, LogPosition::INVALID);
        let satisfied = cluster.run_until(ctx, cluster.clock.after(10_000), |cluster| {
            cluster.leader_node().is_some()
        });
        if !satisfied {
            return Err("cluster creation timed out".into());
        }

        let proposals = noprop::sample_usize_in(ctx, 4..=32);
        let snapshot_at = noprop::sample_usize_in(ctx, 1..proposals - 1);
        let repair_at = noprop::sample_usize_in(ctx, snapshot_at + 1..proposals);
        // Both followers will need a snapshot after losing storage.
        // Dropping their first transfers makes a later successful
        // installation evidence that the leader retried.
        let forced_snapshot_drops = 2;
        let mut positions = Vec::new();
        let mut snapshot_index = LogIndex::ZERO;
        for i in 0..proposals {
            if i == snapshot_at {
                // Take a snapshot at the current commit index.
                let satisfied = cluster.run_until(ctx, cluster.clock.after(10_000), |cluster| {
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
                        node.lose_storage();
                    }
                }
                cluster.drop_next_snapshots(forced_snapshot_drops);
            }

            let found = cluster.run_while_leader_absent(ctx, cluster.clock.after(10_000));
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

        let satisfied = cluster.run_until(ctx, cluster.clock.after(1_000_000), |cluster| {
            let reference = &cluster.nodes[0].inner;
            cluster.nodes.iter().skip(1).all(|node| {
                node.inner.commit_index() == reference.commit_index()
                    && node.inner.log().last_position() == reference.log().last_position()
            })
        });
        if !satisfied {
            return Err("log tails and commit indices are not synchronized".into());
        }
        check_snapshot_compatible_logs(&cluster)?;
        if cluster.snapshot_requests() <= forced_snapshot_drops {
            return Err(format!(
                "snapshot was not retried after {forced_snapshot_drops} forced drops: requests={}",
                cluster.snapshot_requests()
            )
            .into());
        }
        if cluster.snapshot_drops() < forced_snapshot_drops {
            return Err(format!(
                "only {} of {forced_snapshot_drops} forced snapshot drops were observed",
                cluster.snapshot_drops()
            )
            .into());
        }
        if cluster.snapshot_installations_succeeded() == 0 {
            return Err("no remote snapshot installation completed".into());
        }
        if cluster.snapshot_installations_rejected() != 0 {
            return Err(format!(
                "{} remote snapshot installations were rejected",
                cluster.snapshot_installations_rejected()
            )
            .into());
        }
        Ok(())
    })?;

    assert!(
        cases_with_snapshot.get() > 0,
        "no case exercised a snapshot installation (seed={seed:#018x})",
    );
    Ok(())
}

/// After the leader is isolated from the cluster and a new leader is
/// elected, rejoining the old leader reconciles the divergent logs:
/// every proposed position reaches a terminal status, at least one
/// new-leader position commits, and the commit indices converge.
#[test]
fn divergent_logs_reconcile() -> noprop::TestResult {
    run(16, |ctx| {
        let node_ids = [NodeId::new(0), NodeId::new(1), NodeId::new(2)];
        let mut cluster = TestCluster::new(&node_ids);

        let position = cluster.random_node_mut(ctx).create_cluster(&node_ids);
        assert_ne!(position, LogPosition::INVALID);
        let satisfied = cluster.run_until(ctx, cluster.clock.after(10_000), |cluster| {
            cluster.leader_node().is_some()
        });
        if !satisfied {
            return Err("cluster creation timed out".into());
        }
        // Let the initial cluster-config / term entries replicate to
        // all nodes before proceeding: a follower that never receives
        // the config entry is not a voter and can never campaign.
        cluster.run(ctx, cluster.clock.after(100));

        // Propose commands as usual.
        let mut before_isolation_positions = Vec::new();
        let first_batch = noprop::sample_usize_in(ctx, 4..=16);
        for _ in 0..first_batch {
            let Some(leader) = cluster.leader_node_mut() else {
                unreachable!();
            };
            before_isolation_positions.push(leader.propose_command());
            let ticks = MinMax::new(1, 10).sample(ctx);
            cluster.run(ctx, cluster.clock.after(ticks));
        }

        // Propose more commands without giving the cluster time to
        // replicate them, then isolate the leader.
        let second_batch = noprop::sample_usize_in(ctx, 4..=16);
        let mut isolated_leader_positions = Vec::new();
        for _ in 0..second_batch {
            let Some(leader) = cluster.leader_node_mut() else {
                unreachable!();
            };
            isolated_leader_positions.push(leader.propose_command());
        }
        let leader_node_index = cluster
            .nodes
            .iter()
            .position(|n| n.inner.role().is_leader())
            .expect("leader exists");
        let old_leader = cluster.nodes.remove(leader_node_index);

        // Elect a new leader.
        let found = cluster.run_while_leader_absent(ctx, cluster.clock.after(1_000_000));
        if !found {
            return Err("new leader was not elected".into());
        }

        // Propose remaining commands.
        let third_batch = noprop::sample_usize_in(ctx, 4..=16);
        let mut new_leader_positions = Vec::new();
        for _ in 0..third_batch {
            let found = cluster.run_while_leader_absent(ctx, cluster.clock.after(1_000_000));
            if !found {
                return Err("leader absent while proposing".into());
            }
            let Some(leader) = cluster.leader_node_mut() else {
                unreachable!();
            };
            new_leader_positions.push(leader.propose_command());
            let ticks = MinMax::new(1, 10).sample(ctx);
            cluster.run(ctx, cluster.clock.after(ticks));
        }

        // Rejoin the old leader.
        cluster.nodes.push(old_leader);

        // Uncommitted positions on the isolated leader are truncated
        // after the rejoin, so a terminal `Rejected` status is
        // legitimate here; the liveness claim is that the cluster
        // recovers and keeps committing.
        assert_all_terminal(&mut cluster, ctx, &before_isolation_positions, true, false)?;
        assert_all_terminal(&mut cluster, ctx, &isolated_leader_positions, true, false)?;
        let committed_by_new_leader =
            assert_all_terminal(&mut cluster, ctx, &new_leader_positions, true, false)?;
        if committed_by_new_leader == 0 {
            return Err("the newly elected leader committed no proposal".into());
        }

        let satisfied = cluster.run_until(ctx, cluster.clock.after(10_000), |cluster| {
            cluster.nodes[0].inner.commit_index() == cluster.nodes[1].inner.commit_index()
                && cluster.nodes[0].inner.commit_index() == cluster.nodes[2].inner.commit_index()
        });
        if !satisfied {
            return Err("commit indices are not synchronized".into());
        }
        Ok(())
    })
}
