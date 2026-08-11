//! Model-based PBTs for noraft cluster configurations.

#[path = "helpers/pbt.rs"]
pub mod pbt;

use noraft::{ClusterConfig, NodeId};
use pbt::{run, sample_config, sample_normal_config};
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

/// `to_joint_consensus` keeps the old voters and computes the new
/// voters as `(old + additions) - removals`.
#[test]
fn cluster_config_to_joint_consensus_matches_set_model() -> noprop::TestResult {
    run(1024, |ctx| {
        let base = sample_normal_config(ctx);
        let before = base.clone();
        let additions: Vec<NodeId> = (100..104)
            .filter(|_| noprop::sample_bool(ctx))
            .map(NodeId::new)
            .collect();
        let removals: Vec<NodeId> = base
            .voters
            .iter()
            .copied()
            .filter(|_| noprop::sample_bool(ctx))
            .collect();
        let mut expected = base.voters.clone();
        expected.extend(additions.iter().copied());
        expected.retain(|id| !removals.contains(id));

        let joint = base.to_joint_consensus(&additions, &removals);
        if base != before {
            return Err("to_joint_consensus mutated its source config".into());
        }
        if joint.voters != base.voters
            || joint.non_voters != base.non_voters
            || joint.new_voters != expected
        {
            return Err(format!(
                "joint config mismatch: got {joint:?}, expected new voters {expected:?}"
            )
            .into());
        }
        Ok(())
    })
}
