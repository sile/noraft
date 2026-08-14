//! Model-based PBTs for noraft cluster configurations.

pub mod pbt_harness;

use noraft::{ClusterConfig, NodeId};
use pbt_harness::{run, sample_config, sample_normal_config};
use std::collections::BTreeSet;

/// `unique_nodes` is the sorted set union, including configurations
/// where voter sets overlap during joint consensus.
#[test]
fn cluster_config_unique_nodes_matches_set_union() -> noprop::TestResult {
    run(1024, |ctx| {
        let empty = ClusterConfig::new();
        if empty.unique_nodes().next().is_some() {
            return Err("an empty config yielded a node".into());
        }

        let mut config = sample_config(ctx);
        let joint_overlap = NodeId::new(100);
        config.voters.insert(joint_overlap);
        config.new_voters.insert(joint_overlap);
        let three_way_overlap = NodeId::new(101);
        config.voters.insert(three_way_overlap);
        config.new_voters.insert(three_way_overlap);
        config.non_voters.insert(three_way_overlap);

        let expected: BTreeSet<NodeId> = config
            .voters
            .iter()
            .chain(&config.new_voters)
            .chain(&config.non_voters)
            .copied()
            .collect();
        let actual: Vec<NodeId> = config.unique_nodes().collect();
        if actual != expected.iter().copied().collect::<Vec<_>>() {
            return Err(format!("unique_nodes returned {actual:?}, expected {expected:?}").into());
        }
        for id in &expected {
            if !config.contains(*id) {
                return Err(format!("contains({id:?}) rejected a member of the union").into());
            }
        }
        if config.is_joint_consensus() != !config.new_voters.is_empty() {
            return Err("is_joint_consensus disagrees with new_voters".into());
        }
        Ok(())
    })
}

/// `to_joint_consensus` keeps the old voters, computes the new voters
/// as `(old + additions) - removals`, and strips any promoted node
/// from `non_voters` so the result satisfies the disjointness
/// precondition of `Node::propose_config`.
#[test]
fn cluster_config_to_joint_consensus_matches_set_model() -> noprop::TestResult {
    run(1024, |ctx| {
        let base = sample_normal_config(ctx);
        let before = base.clone();
        // Mix fresh ids with existing non-voters so the sampler also
        // exercises the promote-existing-non-voter branch.
        let mut addition_pool: Vec<NodeId> = (100..104).map(NodeId::new).collect();
        addition_pool.extend(base.non_voters.iter().copied());
        let additions: Vec<NodeId> = addition_pool
            .into_iter()
            .filter(|_| noprop::sample_bool(ctx))
            .collect();
        let removals: Vec<NodeId> = base
            .voters
            .iter()
            .copied()
            .filter(|_| noprop::sample_bool(ctx))
            .collect();
        let mut expected_new_voters = base.voters.clone();
        expected_new_voters.extend(additions.iter().copied());
        expected_new_voters.retain(|id| !removals.contains(id));
        let mut expected_non_voters = base.non_voters.clone();
        expected_non_voters.retain(|id| !expected_new_voters.contains(id));

        let joint = base.to_joint_consensus(&additions, &removals);
        if base != before {
            return Err("to_joint_consensus mutated its source config".into());
        }
        if joint.voters != base.voters
            || joint.non_voters != expected_non_voters
            || joint.new_voters != expected_new_voters
        {
            return Err(format!(
                "joint config mismatch: got {joint:?}, \
                 expected new_voters {expected_new_voters:?}, \
                 expected non_voters {expected_non_voters:?}"
            )
            .into());
        }
        Ok(())
    })
}
