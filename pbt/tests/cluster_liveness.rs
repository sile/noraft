//! Bounded-liveness PBTs for stable, unstable, and pipelined traffic.
//!
//! Every test obtains its seed through `noprop::seed_from_env_or_time`
//! via the shared runner. CI therefore uses a fresh time-derived seed
//! unless `NORAFT_PBT_SEED` is explicitly set for reproduction.

use noraft::{LogPosition, NodeId};
use pbt::{MinMax, TestCluster, assert_all_terminal, run};

/// Command proposals commit under stable links, the commit indices
/// converge, and the leader does not change (term stays 1).
#[test]
fn proposals_commit_with_stable_links() -> noprop::TestResult {
    run(64, |ctx| {
        let node_ids = [NodeId::new(0), NodeId::new(1), NodeId::new(2)];
        let mut cluster = TestCluster::new(&node_ids);
        // The harness default includes a 1% drop rate, which can delay
        // RequestVote and bump the term during bootstrap. This property
        // is specifically about leadership stability, so drops are off.
        cluster.default_link_options.drop_rate = noprop::Ratio::new(0, 1);

        let position = cluster.random_node_mut(ctx).create_cluster(&node_ids);
        assert_ne!(position, LogPosition::INVALID);
        let satisfied = cluster.run_until(ctx, cluster.clock.after(10_000), |cluster| {
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
            cluster.run(ctx, cluster.clock.after(ticks));
        }

        let committed = assert_all_terminal(&mut cluster, ctx, &positions, false, false)?;
        if committed != positions.len() {
            return Err("all proposals must commit".into());
        }

        let satisfied = cluster.run_until(ctx, cluster.clock.after(1000), |cluster| {
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
    })
}

/// Command proposals commit even under a very unstable network
/// (30% drop rate, 1-1000 tick latency).
#[test]
fn proposals_commit_with_unstable_links() -> noprop::TestResult {
    run(32, |ctx| {
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

        let proposals = noprop::sample_usize_in(ctx, 1..=16);
        let mut positions = Vec::new();
        for _ in 0..proposals {
            let found = cluster.run_while_leader_absent(ctx, cluster.clock.after(100_000));
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

        let satisfied = cluster.run_until(ctx, cluster.clock.after(100_000), |cluster| {
            cluster.nodes[0].inner.commit_index() == cluster.nodes[1].inner.commit_index()
                && cluster.nodes[0].inner.commit_index() == cluster.nodes[2].inner.commit_index()
        });
        if !satisfied {
            return Err("commit indices are not synchronized".into());
        }
        Ok(())
    })
}

/// Command proposals commit when proposals are pipelined (no waiting
/// for the previous commit) and interleaved with heartbeats.
#[test]
fn proposals_commit_with_pipelining() -> noprop::TestResult {
    run(32, |ctx| {
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
        let mut positions = Vec::new();
        // Always issue at least one back-to-back pair before the
        // harness gets a chance to run. This is the trigger the
        // property is named after, so it must not depend on a draw.
        let Some(leader) = cluster.leader_node_mut() else {
            unreachable!();
        };
        positions.push(leader.propose_command());
        positions.push(leader.propose_command());

        for _ in 2..proposals {
            let pipeline = noprop::sample_ratio(ctx, noprop::Ratio::new(4, 5));
            let do_heartbeat = noprop::sample_ratio(ctx, noprop::Ratio::one_nth(2));

            let found = cluster.run_while_leader_absent(ctx, cluster.clock.after(10_000));
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
                cluster.run(ctx, cluster.clock.after(ticks));
            }
        }

        let committed = assert_all_terminal(&mut cluster, ctx, &positions, false, false)?;
        if committed != positions.len() {
            return Err("all proposals must commit".into());
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
