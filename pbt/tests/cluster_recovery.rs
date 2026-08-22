//! Recovery PBTs for restarts and divergent logs.
//!
//! Every test obtains its seed through `noprop::seed_from_env_or_time`
//! via the shared runner. CI therefore uses a fresh time-derived seed
//! unless `NORAFT_PBT_SEED` is explicitly set for reproduction.

use noraft::{LogPosition, NodeId};
use pbt::{MinMax, TestCluster, assert_all_terminal, run};

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
            cluster.nodes[0].restarts() > 0
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
