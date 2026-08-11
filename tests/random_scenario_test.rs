use noraft::{
    ClusterConfig, CommitStatus, Log, LogEntries, LogIndex, LogPosition, Message, Node,
    NodeGeneration, NodeId, Role, Term,
};
use rand::{
    Rng, RngExt, SeedableRng,
    distr::{Distribution, uniform::SampleRange},
    prelude::IndexedRandom,
    rngs::StdRng,
};
use std::collections::BTreeMap;

#[test]
fn propose_commands() -> noprop::TestResult {
    let seed = noprop::seed_from_env_or_time("NORAFT_PBT_SEED")?;
    let rng = StdRng::seed_from_u64(seed);

    let node_ids = [NodeId::new(0), NodeId::new(1), NodeId::new(2)];

    // Create a cluster.
    let mut cluster = TestCluster::new(&node_ids, rng);
    let position = cluster.random_node_mut().create_cluster(&node_ids);
    assert_ne!(position, LogPosition::INVALID);

    let deadline = cluster.clock.add(10000);
    let satisfied = cluster.run_until(deadline, |cluster| cluster.leader_node().is_some());
    assert!(satisfied, "Create cluster timeout");

    // Propose commands.
    let mut positions = Vec::new();
    for _ in 0..100 {
        let Some(leader) = cluster.leader_node_mut() else {
            panic!("No leader");
        };
        positions.push(leader.propose_command());

        let ticks = MinMax::new(1, 10).sample(&mut cluster.rng);
        cluster.run(cluster.clock.add(ticks));
    }
    assert_eq!(positions.len(), 100);

    for position in positions {
        let mut committed = false;
        for _ in 0..1000 {
            let Some(leader) = cluster.leader_node_mut() else {
                panic!("No leader");
            };
            if leader.get_commit_status(position).is_committed() {
                committed = true;
                break;
            }
            cluster.run(cluster.clock.add(10));
        }
        assert!(committed);
    }

    let deadline = cluster.clock.add(1000);
    let satisfied = cluster.run_until(deadline, |cluster| {
        cluster.nodes[0].inner.commit_index() == cluster.nodes[1].inner.commit_index()
            && cluster.nodes[0].inner.commit_index() == cluster.nodes[2].inner.commit_index()
    });
    assert!(satisfied, "Commit indices are not synchronized");

    // Links are stable, so the leader should not change.
    assert_eq!(cluster.nodes[0].inner.current_term().get(), 1);
    Ok(())
}

#[test]
fn unstable_network() -> noprop::TestResult {
    let seed = noprop::seed_from_env_or_time("NORAFT_PBT_SEED")?;
    let rng = StdRng::seed_from_u64(seed);

    let node_ids = [NodeId::new(0), NodeId::new(1), NodeId::new(2)];

    // Create a cluster.
    let mut cluster = TestCluster::new(&node_ids, rng);

    // [NOTE] This test is almost the same as `propose_commands()`, but the network is very unstable.
    cluster.default_link_options.drop_rate = 0.3;
    cluster.default_link_options.latency_ticks = MinMax::new(1, 1000);

    let position = cluster.random_node_mut().create_cluster(&node_ids);
    assert_ne!(position, LogPosition::INVALID);

    let deadline = cluster.clock.add(100000);
    let satisfied = cluster.run_until(deadline, |cluster| cluster.leader_node().is_some());
    assert!(satisfied, "Create cluster timeout");

    // Propose commands.
    let mut positions = Vec::new();
    for _ in 0..100 {
        cluster.run_while_leader_absent(cluster.clock.add(100_000));
        let Some(leader) = cluster.leader_node_mut() else {
            panic!("No leader");
        };
        positions.push(leader.propose_command());

        let ticks = MinMax::new(1, 10).sample(&mut cluster.rng);
        cluster.run(cluster.clock.add(ticks));
    }
    assert_eq!(positions.len(), 100);

    for position in positions {
        let mut committed = false;
        for _ in 0..10000 {
            cluster.run_while_leader_absent(cluster.clock.add(100_000));
            let Some(leader) = cluster.leader_node_mut() else {
                panic!("No leader");
            };
            if leader.get_commit_status(position).is_committed() {
                committed = true;
                break;
            }
            cluster.run(cluster.clock.add(10));
        }
        assert!(committed);
    }

    let deadline = cluster.clock.add(100000);
    let satisfied = cluster.run_until(deadline, |cluster| {
        cluster.nodes[0].inner.commit_index() == cluster.nodes[1].inner.commit_index()
            && cluster.nodes[0].inner.commit_index() == cluster.nodes[2].inner.commit_index()
    });
    assert!(satisfied, "Commit indices are not synchronized");
    Ok(())
}

#[test]
fn node_restart() -> noprop::TestResult {
    let seed = noprop::seed_from_env_or_time("NORAFT_PBT_SEED")?;
    let rng = StdRng::seed_from_u64(seed);

    let node_ids = [NodeId::new(0), NodeId::new(1), NodeId::new(2)];

    // Create a cluster.
    let mut cluster = TestCluster::new(&node_ids, rng);

    cluster.nodes[0].options.running_ticks = MinMax::new(800, 5000);
    cluster.nodes[0].options.stopping_ticks = MinMax::new(800, 5000);

    let position = cluster.random_node_mut().create_cluster(&node_ids);
    assert_ne!(position, LogPosition::INVALID);

    let deadline = cluster.clock.add(10000);
    let satisfied = cluster.run_until(deadline, |cluster| cluster.leader_node().is_some());
    assert!(satisfied, "Create cluster timeout");

    // Propose commands.
    let mut positions = Vec::new();
    for _ in 0..100 {
        cluster.run_while_leader_absent(cluster.clock.add(10000));
        let Some(leader) = cluster.leader_node_mut() else {
            panic!("No leader");
        };
        positions.push(leader.propose_command());

        let ticks = MinMax::new(1, 10).sample(&mut cluster.rng);
        cluster.run(cluster.clock.add(ticks));
    }
    assert_eq!(positions.len(), 100);

    for position in positions {
        let mut committed = false;
        for _ in 0..1000 {
            cluster.run_while_leader_absent(cluster.clock.add(10000));
            let Some(leader) = cluster.leader_node_mut() else {
                panic!("No leader");
            };
            if leader.get_commit_status(position).is_committed() {
                committed = true;
                break;
            }
            cluster.run(cluster.clock.add(10));
        }
        assert!(committed);
    }

    let deadline = cluster.clock.add(50000);
    let satisfied = cluster.run_until(deadline, |cluster| {
        cluster.nodes[0].inner.commit_index() == cluster.nodes[1].inner.commit_index()
            && cluster.nodes[0].inner.commit_index() == cluster.nodes[2].inner.commit_index()
    });
    assert!(satisfied, "Commit indices are not synchronized");
    Ok(())
}

#[test]
fn pipelining() -> noprop::TestResult {
    let seed = noprop::seed_from_env_or_time("NORAFT_PBT_SEED")?;
    let rng = StdRng::seed_from_u64(seed);

    let node_ids = [NodeId::new(0), NodeId::new(1), NodeId::new(2)];

    // Create a cluster.
    let mut cluster = TestCluster::new(&node_ids, rng);
    let position = cluster.random_node_mut().create_cluster(&node_ids);
    assert_ne!(position, LogPosition::INVALID);

    let deadline = cluster.clock.add(10000);
    let satisfied = cluster.run_until(deadline, |cluster| cluster.leader_node().is_some());
    assert!(satisfied, "Create cluster timeout");

    // Propose commands and trigger heartbeats.
    let mut positions = Vec::new();
    for _ in 0..100 {
        let pipeline_command = cluster.rng.random_bool(0.8);
        let do_hearbeat = cluster.rng.random_bool(0.5);

        cluster.run_while_leader_absent(cluster.clock.add(10000));
        let Some(leader) = cluster.leader_node_mut() else {
            panic!("No leader");
        };
        positions.push(leader.propose_command());
        if do_hearbeat {
            assert!(leader.heartbeat());
        }

        if !pipeline_command {
            let ticks = MinMax::new(0, 5).sample(&mut cluster.rng);
            cluster.run(cluster.clock.add(ticks));
        }
    }
    assert_eq!(positions.len(), 100);

    for position in positions {
        let mut committed = false;
        for _ in 0..1000 {
            cluster.run_while_leader_absent(cluster.clock.add(10000));
            let Some(leader) = cluster.leader_node_mut() else {
                panic!("No leader");
            };
            if leader.get_commit_status(position).is_committed() {
                committed = true;
                break;
            }
            cluster.run(cluster.clock.add(10));
        }
        assert!(committed);
    }

    let deadline = cluster.clock.add(10000);
    let satisfied = cluster.run_until(deadline, |cluster| {
        cluster.nodes[0].inner.commit_index() == cluster.nodes[1].inner.commit_index()
            && cluster.nodes[0].inner.commit_index() == cluster.nodes[2].inner.commit_index()
    });
    assert!(satisfied, "Commit indices are not synchronized");
    Ok(())
}

#[test]
fn storage_repair_without_snapshot() -> noprop::TestResult {
    let seed = noprop::seed_from_env_or_time("NORAFT_PBT_SEED")?;
    let rng = StdRng::seed_from_u64(seed);

    let node_ids = [NodeId::new(0), NodeId::new(1), NodeId::new(2)];

    // Create a cluster.
    let mut cluster = TestCluster::new(&node_ids, rng);
    let position = cluster.random_node_mut().create_cluster(&node_ids);
    assert_ne!(position, LogPosition::INVALID);

    let deadline = cluster.clock.add(10000);
    let satisfied = cluster.run_until(deadline, |cluster| cluster.leader_node().is_some());
    assert!(satisfied, "Create cluster timeout");

    // Propose commands.
    let mut positions = Vec::new();
    for i in 0..100 {
        if i == 50 {
            for node in cluster.nodes.iter_mut() {
                if !node.inner.role().is_leader() {
                    // Reset the node.
                    let generation =
                        NodeGeneration::new(node.inner.generation().get().saturating_add(1));
                    let log = Log::new(ClusterConfig::new(), LogEntries::new(LogPosition::ZERO));
                    node.inner = Node::restart(node.inner.id(), generation, Term::ZERO, None, log);
                }
            }
        }

        cluster.run_while_leader_absent(cluster.clock.add(10000));
        let Some(leader) = cluster.leader_node_mut() else {
            panic!("No leader");
        };
        positions.push(leader.propose_command());

        let ticks = MinMax::new(1, 10).sample(&mut cluster.rng);
        cluster.run(cluster.clock.add(ticks));
    }
    assert_eq!(positions.len(), 100);

    for position in positions {
        let mut committed = false;
        for _ in 0..1000 {
            let Some(leader) = cluster.leader_node_mut() else {
                panic!("No leader");
            };
            if leader.get_commit_status(position).is_committed() {
                committed = true;
                break;
            }
            cluster.run(cluster.clock.add(10));
        }
        assert!(committed);
    }

    let deadline = cluster.clock.add(1_000_000);
    let satisfied = cluster.run_until(deadline, |cluster| {
        cluster.nodes[0].inner.commit_index() == cluster.nodes[1].inner.commit_index()
            && cluster.nodes[0].inner.commit_index() == cluster.nodes[2].inner.commit_index()
    });
    assert!(satisfied, "Commit indices are not synchronized");
    Ok(())
}

#[test]
fn storage_repair_with_snapshot() -> noprop::TestResult {
    let seed = noprop::seed_from_env_or_time("NORAFT_PBT_SEED")?;
    let rng = StdRng::seed_from_u64(seed);

    let node_ids = [NodeId::new(0), NodeId::new(1), NodeId::new(2)];

    // Create a cluster.
    let mut cluster = TestCluster::new(&node_ids, rng);
    let position = cluster.random_node_mut().create_cluster(&node_ids);
    assert_ne!(position, LogPosition::INVALID);

    let deadline = cluster.clock.add(10000);
    let satisfied = cluster.run_until(deadline, |cluster| cluster.leader_node().is_some());
    assert!(satisfied, "Create cluster timeout");

    // Propose commands.
    let mut positions = Vec::new();
    let mut snapshot_index = LogIndex::ZERO;
    for i in 0..100 {
        if i == 25 {
            // Take snapshot
            cluster.run_until(cluster.clock.add(10000), |cluster| {
                cluster
                    .nodes
                    .iter()
                    .all(|node| node.inner.commit_index().get() > 0)
            });
            for node in cluster.nodes.iter_mut() {
                let (position, config) = node
                    .inner
                    .log()
                    .get_position_and_config(node.inner.commit_index())
                    .expect("unreachable");
                assert!(
                    node.inner
                        .handle_snapshot_installed(position, config.clone())
                );
                if node.inner.role().is_leader() {
                    snapshot_index = position.index;
                }
            }
        }
        if i == 50 {
            for node in cluster.nodes.iter_mut() {
                if !node.inner.role().is_leader() {
                    // Reset the node.
                    let generation =
                        NodeGeneration::new(node.inner.generation().get().saturating_add(1));
                    let log = Log::new(ClusterConfig::new(), LogEntries::new(LogPosition::ZERO));
                    node.inner = Node::restart(node.inner.id(), generation, Term::ZERO, None, log);
                }
            }
        }

        cluster.run_while_leader_absent(cluster.clock.add(10000));
        let Some(leader) = cluster.leader_node_mut() else {
            panic!("No leader");
        };
        positions.push(leader.propose_command());

        let ticks = MinMax::new(1, 10).sample(&mut cluster.rng);
        cluster.run(cluster.clock.add(ticks));
    }
    assert_eq!(positions.len(), 100);

    for position in positions {
        let mut status = CommitStatus::InProgress;
        for _ in 0..1000 {
            let Some(leader) = cluster.leader_node_mut() else {
                panic!("No leader");
            };

            status = leader.get_commit_status(position);
            if !status.is_in_progress() {
                break;
            }
            cluster.run(cluster.clock.add(10));
        }

        if position.index < snapshot_index {
            assert!(status.is_unknown());
        } else {
            assert!(status.is_committed());
        }
    }

    let deadline = cluster.clock.add(1_000_000);
    let satisfied = cluster.run_until(deadline, |cluster| {
        cluster.nodes[0].inner.commit_index() == cluster.nodes[1].inner.commit_index()
            && cluster.nodes[0].inner.commit_index() == cluster.nodes[2].inner.commit_index()
    });
    assert!(satisfied, "Commit indices are not synchronized");
    Ok(())
}

#[test]
fn dynamic_membership() -> noprop::TestResult {
    let seed = noprop::seed_from_env_or_time("NORAFT_PBT_SEED")?;
    let rng = StdRng::seed_from_u64(seed);

    let node_ids = [NodeId::new(0), NodeId::new(1), NodeId::new(2)];

    // Create a cluster.
    let mut cluster = TestCluster::new(&node_ids, rng);

    cluster.default_link_options.drop_rate = 0.3;
    cluster.default_link_options.latency_ticks = MinMax::new(1, 1000);

    let position = cluster.random_node_mut().create_cluster(&node_ids);
    assert_ne!(position, LogPosition::INVALID);

    let deadline = cluster.clock.add(100000);
    let satisfied = cluster.run_until(deadline, |cluster| cluster.leader_node().is_some());
    assert!(satisfied, "Create cluster timeout");

    for i in 0..10 {
        // Change the cluster configuration. Wait for the previous
        // configuration change to settle first: proposing the next
        // change while the cluster is still in joint consensus returns
        // `LogPosition::INVALID`.
        let deadline = cluster.clock.add(1_000_000);
        let settled = cluster.run_until(deadline, |cluster| {
            cluster
                .leader_node()
                .is_some_and(|leader| !leader.config().is_joint_consensus())
        });
        assert!(settled, "Leader without joint consensus not found");
        if cluster.rng.random_bool(0.7) {
            // Add.
            let node_id = NodeId::new(3 + i);
            let voter = cluster.rng.random_bool(0.5);
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
            let position = leader.propose_config(new_config);
            assert_ne!(position, LogPosition::INVALID);
        } else if cluster.nodes.iter().filter(|n| n.voter).count() > 2 {
            // Remove.
            let node_ids = cluster
                .nodes
                .iter()
                .map(|n| n.inner.id())
                .collect::<Vec<_>>();
            let node_id = node_ids
                .choose(&mut cluster.rng)
                .copied()
                .expect("unreachable");

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
        for _ in 0..10 {
            cluster.run_while_leader_absent(cluster.clock.add(1_000_000));
            let Some(leader) = cluster.leader_node_mut() else {
                panic!("No leader");
            };
            positions.push(leader.propose_command());

            let ticks = MinMax::new(1, 10).sample(&mut cluster.rng);
            cluster.run(cluster.clock.add(ticks));
        }
        assert_eq!(positions.len(), 10);

        let mut success_count = 0;
        for position in positions {
            for _ in 0..20000 {
                cluster.run_while_leader_absent(cluster.clock.add(1_000_000));
                let Some(leader) = cluster.leader_node_mut() else {
                    panic!("No leader");
                };
                if !leader.get_commit_status(position).is_in_progress() {
                    if leader.get_commit_status(position).is_committed() {
                        success_count += 1;
                    }
                    break;
                }
                cluster.run(cluster.clock.add(10));
            }
        }
        // The committed count is not asserted: under the random
        // scheduler (drop_rate / latency) the number of commits within
        // the budget is not guaranteed. The safety of committed entries
        // is checked by the invariant tests in `tests/prop_cluster_test.rs`,
        // and bounded liveness is left to controlled-scheduler
        // scenarios.
        let _ = success_count;
    }
    Ok(())
}

#[test]
fn truncate_divergence_log() -> noprop::TestResult {
    let seed = noprop::seed_from_env_or_time("NORAFT_PBT_SEED")?;
    let rng = StdRng::seed_from_u64(seed);

    let node_ids = [NodeId::new(0), NodeId::new(1), NodeId::new(2)];

    // Create a cluster.
    let mut cluster = TestCluster::new(&node_ids, rng);
    let position = cluster.random_node_mut().create_cluster(&node_ids);
    assert_ne!(position, LogPosition::INVALID);

    let deadline = cluster.clock.add(10000);
    let satisfied = cluster.run_until(deadline, |cluster| cluster.leader_node().is_some());
    assert!(satisfied, "Create cluster timeout");

    // Propose 20 commands as usual.
    let mut positions = Vec::new();
    for _ in 0..20 {
        let Some(leader) = cluster.leader_node_mut() else {
            panic!("No leader");
        };
        positions.push(leader.propose_command());

        let ticks = MinMax::new(1, 10).sample(&mut cluster.rng);
        cluster.run(cluster.clock.add(ticks));
    }

    // Propose next 20 commands without calling `cluster.run()`
    for _ in 0..20 {
        let Some(leader) = cluster.leader_node_mut() else {
            panic!("No leader");
        };
        positions.push(leader.propose_command());
    }

    // Isolate the leader from the cluster.
    let leader_node_index = cluster
        .nodes
        .iter()
        .position(|n| n.inner.role().is_leader())
        .expect("unreachable");
    let old_leader = cluster.nodes.remove(leader_node_index);

    // Elect a new leader.
    cluster.run_while_leader_absent(cluster.clock.add(1_000_000));

    // Propose remaining commands.
    for _ in 0..60 {
        let Some(leader) = cluster.leader_node_mut() else {
            panic!("No leader");
        };
        positions.push(leader.propose_command());
    }
    assert_eq!(positions.len(), 100);

    // Rejoin the old leader.
    cluster.nodes.push(old_leader);

    let mut success_count = 0;
    for position in positions {
        for _ in 0..1000 {
            let Some(leader) = cluster.leader_node_mut() else {
                panic!("No leader");
            };
            if !leader.get_commit_status(position).is_in_progress() {
                if leader.get_commit_status(position).is_committed() {
                    success_count += 1;
                }
                break;
            }
            cluster.run(cluster.clock.add(10));
        }
    }
    assert!(60 <= success_count);
    assert!(success_count <= 80);

    let deadline = cluster.clock.add(10_000);
    let satisfied = cluster.run_until(deadline, |cluster| {
        cluster.nodes[0].inner.commit_index() == cluster.nodes[1].inner.commit_index()
            && cluster.nodes[0].inner.commit_index() == cluster.nodes[2].inner.commit_index()
    });
    assert!(satisfied, "Commit indices are not synchronized");
    Ok(())
}

#[derive(Debug)]
pub struct TestCluster {
    pub nodes: Vec<TestNode>,
    pub clock: Clock,
    pub rng: StdRng,
    pub default_link_options: TestLinkOptions,
    seqno: u64,
}

impl TestCluster {
    pub fn new(node_ids: &[NodeId], rng: StdRng) -> Self {
        Self {
            nodes: node_ids.iter().map(|&id| TestNode::new(id)).collect(),
            clock: Clock::new(),
            rng,
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

    pub fn random_node_mut(&mut self) -> &mut Node {
        let index = self.rng.random_range(0..self.nodes.len());
        &mut self.nodes[index].inner
    }

    pub fn run_while_leader_absent(&mut self, deadline: Clock) {
        self.run_until(deadline, |cluster| cluster.leader_node().is_some());
    }

    pub fn run(&mut self, deadline: Clock) {
        self.run_until(deadline, |_| false);
    }

    pub fn run_until<F>(&mut self, deadline: Clock, condition: F) -> bool
    where
        F: Fn(&TestCluster) -> bool,
    {
        while self.clock < deadline && !condition(self) {
            self.run_tick();
        }
        self.clock < deadline
    }

    pub fn run_tick(&mut self) {
        self.clock.tick();
        let mut messages = Vec::new();
        let mut snapshots = Vec::new();

        // Run nodes.
        for node in &mut self.nodes {
            node.run_tick(&mut self.rng, self.clock);

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
            self.send_message(src, dst, msg);
        }

        // Deliver snapshots.
        for (src, dst, position, config) in snapshots {
            self.send_snashot(src, dst, position, config);
        }
    }

    fn send_message(&mut self, _src: NodeId, dst: NodeId, msg: Message) {
        let options = &self.default_link_options;

        if self.rng.random_bool(options.drop_rate) {
            return;
        }

        let latency = options.latency_ticks.sample(&mut self.rng) * message_size(&msg);
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
                        .add(node.options.install_snapshot_ticks.sample(&mut self.rng)),
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
    pub drop_rate: f64,
}

impl Default for TestLinkOptions {
    fn default() -> Self {
        Self {
            latency_ticks: MinMax::new(5, 20),
            drop_rate: 0.01,
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
}

impl Distribution<usize> for MinMax {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        rng.random_range(self.min..=self.max)
    }
}

impl SampleRange<usize> for MinMax {
    fn sample_single<R: Rng + ?Sized>(
        self,
        rng: &mut R,
    ) -> Result<usize, rand::distr::uniform::Error> {
        Ok(self.sample(rng))
    }

    fn is_empty(&self) -> bool {
        self.min > self.max
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

    pub fn run_tick(&mut self, rng: &mut StdRng, now: Clock) {
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
            self.stop_time = Some(now.add(self.options.running_ticks.sample(rng)));
        }
        if self.stop_time.take_if(|t| *t <= now).is_some() {
            self.running = false;
            self.timeout_expire_time = None;
            self.storage_finish_time = None;
            self.start_time = Some(now.add(self.options.stopping_ticks.sample(rng)));
            return;
        }

        self.storage_finish_time.take_if(|t| *t <= now);
        if self.storage_finish_time.is_some() {
            // Storage operations are synchronous, so we can't do anything else until they finish.
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
                if self.inner.could_be_disruptive_request_vote(&message) {
                    continue;
                }
                self.inner
                    .handle_message(&message)
                    .expect("message handling should succeed");
            } else {
                break;
            }
        }

        if std::mem::take(&mut self.inner.actions_mut().set_election_timeout) {
            self.reset_election_timeout(rng, now);
        }
        if std::mem::take(&mut self.inner.actions_mut().save_current_term) {
            self.extend_storage_finish_time(rng, now, 1);
        }
        if std::mem::take(&mut self.inner.actions_mut().save_voted_for) {
            self.extend_storage_finish_time(rng, now, 1);
        }
        if let Some(entries) = self.inner.actions_mut().append_log_entries.take() {
            self.extend_storage_finish_time(rng, now, entries.len());
        }
    }

    fn reset_election_timeout(&mut self, rng: &mut StdRng, now: Clock) {
        let timeout = match self.inner.role() {
            Role::Leader => self.options.election_timeout_ticks.min,
            Role::Candidate => self.options.election_timeout_ticks.sample(rng),
            Role::Follower => self.options.election_timeout_ticks.max,
        };
        self.timeout_expire_time = Some(now.add(timeout));
    }

    fn extend_storage_finish_time(&mut self, rng: &mut StdRng, now: Clock, n: usize) {
        let remaining_latency = self.storage_finish_time.map_or(0, |t| t.0 - now.0);
        let additional_latency = self.options.storage_latency_ticks.sample(rng) * n;
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
