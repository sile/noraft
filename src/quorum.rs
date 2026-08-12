use crate::{config::ClusterConfig, log::LogIndex, node::NodeId};
use alloc::collections::BTreeSet;

#[derive(Debug, Clone)]
pub struct Quorum {
    majority_indices: BTreeSet<(LogIndex, NodeId)>,
    new_majority_indices: BTreeSet<(LogIndex, NodeId)>,
}

impl Quorum {
    pub fn new(config: &ClusterConfig) -> Self {
        let majority_indices = config
            .voters
            .iter()
            .take(config.voters.len() / 2 + 1)
            .copied()
            .map(|id| (LogIndex::new(0), id))
            .collect::<BTreeSet<_>>();
        let new_majority_indices = config
            .new_voters
            .iter()
            .take(config.new_voters.len() / 2 + 1)
            .copied()
            .map(|id| (LogIndex::new(0), id))
            .collect::<BTreeSet<_>>();
        Self {
            majority_indices,
            new_majority_indices,
        }
    }

    pub fn update_match_index(
        &mut self,
        config: &ClusterConfig,
        node_id: NodeId,
        old_index: LogIndex,
        index: LogIndex,
    ) {
        debug_assert!(old_index <= index);

        let old_entry = (old_index, node_id);
        let new_entry = (index, node_id);

        if config.voters.contains(&node_id) {
            update_majority(&mut self.majority_indices, old_entry, new_entry);
        }
        if config.new_voters.contains(&node_id) {
            update_majority(&mut self.new_majority_indices, old_entry, new_entry);
        }
    }

    pub fn smallest_majority_index(&self) -> LogIndex {
        let Some(i0) = self.majority_indices.first().map(|(i, _)| *i) else {
            unreachable!();
        };
        if let Some(i1) = self.new_majority_indices.first().map(|(i, _)| *i) {
            i0.min(i1)
        } else {
            i0
        }
    }
}

fn update_majority<T: Ord>(
    set: &mut BTreeSet<(T, NodeId)>,
    old_entry: (T, NodeId),
    new_entry: (T, NodeId),
) {
    debug_assert!(old_entry.0 <= new_entry.0);

    // This set keeps only the current majority-sized top entries, not every
    // node. The incremental update is valid only while each node's index moves
    // monotonically forward; if an index can decrease, rebuild the whole quorum.
    if set.first().is_none_or(|min| new_entry.0 <= min.0) {
        return;
    }

    set.insert(new_entry);
    if !set.remove(&old_entry) {
        set.pop_first();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeMap;
    use alloc::vec::Vec;

    fn id(n: u64) -> NodeId {
        NodeId::new(n)
    }

    fn idx(n: u64) -> LogIndex {
        LogIndex::new(n)
    }

    fn cfg(voters: &[u64], new_voters: &[u64], non_voters: &[u64]) -> ClusterConfig {
        let mut c = ClusterConfig::new();
        for &v in voters {
            c.voters.insert(id(v));
        }
        for &v in new_voters {
            c.new_voters.insert(id(v));
        }
        for &v in non_voters {
            c.non_voters.insert(id(v));
        }
        c
    }

    // Naive top-k min over the current index of every voter, used as an
    // oracle for `Quorum::smallest_majority_index`.
    fn naive_smallest_majority(voters: &BTreeMap<u64, u64>) -> u64 {
        let mut sorted: Vec<u64> = voters.values().copied().collect();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        let majority = voters.len() / 2 + 1;
        sorted[majority - 1]
    }

    #[test]
    fn majority_size_matches_voter_count() {
        let q = Quorum::new(&cfg(&[1, 2, 3, 4, 5], &[], &[]));
        assert_eq!(q.majority_indices.len(), 3);
        assert_eq!(q.new_majority_indices.len(), 0);

        let q = Quorum::new(&cfg(&[1, 2, 3], &[], &[]));
        assert_eq!(q.majority_indices.len(), 2);

        let q = Quorum::new(&cfg(&[1], &[], &[]));
        assert_eq!(q.majority_indices.len(), 1);

        // In joint consensus each voter set has its own majority quota.
        let q = Quorum::new(&cfg(&[1, 2, 3], &[1, 2, 3, 4, 5], &[]));
        assert_eq!(q.majority_indices.len(), 2);
        assert_eq!(q.new_majority_indices.len(), 3);
    }

    #[test]
    fn advancing_a_voter_outside_top_k_evicts_the_min() {
        let c = cfg(&[1, 2, 3, 4, 5], &[], &[]);
        let mut q = Quorum::new(&c);
        // BTreeSet<NodeId> yields ids in ascending order, so the initial
        // top-k picks the three smallest ids.
        assert_eq!(
            q.majority_indices.iter().copied().collect::<Vec<_>>(),
            [(idx(0), id(1)), (idx(0), id(2)), (idx(0), id(3))]
        );
        assert_eq!(q.smallest_majority_index(), idx(0));

        // Advancing a voter outside the top-k inserts it and evicts the
        // current minimum (index-then-id order).
        q.update_match_index(&c, id(4), idx(0), idx(10));
        assert_eq!(
            q.majority_indices.iter().copied().collect::<Vec<_>>(),
            [(idx(0), id(2)), (idx(0), id(3)), (idx(10), id(4))]
        );
        assert_eq!(q.smallest_majority_index(), idx(0));

        q.update_match_index(&c, id(5), idx(0), idx(20));
        assert_eq!(
            q.majority_indices.iter().copied().collect::<Vec<_>>(),
            [(idx(0), id(3)), (idx(10), id(4)), (idx(20), id(5))]
        );
        assert_eq!(q.smallest_majority_index(), idx(0));

        // Advancing a voter already inside the top-k updates its entry in
        // place: node 3 moves from (0, 3) to (15, 3), and the set's min
        // shifts from (0, 3) to (10, 4).
        q.update_match_index(&c, id(3), idx(0), idx(15));
        assert_eq!(
            q.majority_indices.iter().copied().collect::<Vec<_>>(),
            [(idx(10), id(4)), (idx(15), id(3)), (idx(20), id(5))]
        );
        assert_eq!(q.smallest_majority_index(), idx(10));
    }

    #[test]
    fn same_index_tie_break_favors_larger_node_id() {
        let c = cfg(&[1, 2, 3, 4, 5], &[], &[]);
        let mut q = Quorum::new(&c);
        // Same-index update on a voter outside the top-k is a no-op because
        // `update_majority` returns early when `new_entry.0 <= min.0`.
        q.update_match_index(&c, id(4), idx(0), idx(0));
        assert_eq!(
            q.majority_indices.iter().copied().collect::<Vec<_>>(),
            [(idx(0), id(1)), (idx(0), id(2)), (idx(0), id(3))]
        );

        // A strictly greater index brings node 4 in and evicts (0, 1); the
        // eviction resolves the same-index tie in favor of larger node ids.
        q.update_match_index(&c, id(4), idx(0), idx(1));
        assert_eq!(
            q.majority_indices.iter().copied().collect::<Vec<_>>(),
            [(idx(0), id(2)), (idx(0), id(3)), (idx(1), id(4))]
        );
    }

    #[test]
    fn joint_smallest_majority_picks_min_of_old_and_new() {
        let c = cfg(&[1, 2, 3], &[1, 2, 4, 5, 6], &[]);
        let mut q = Quorum::new(&c);
        assert_eq!(q.smallest_majority_index(), idx(0));

        // Raise every old voter but leave the new-only voters at 0.
        for n in [1u64, 2, 3] {
            q.update_match_index(&c, id(n), idx(0), idx(10));
        }
        // Old majority min is 10; new majority still contains the fresh
        // new-only voters at index 0, so the joint min is 0.
        assert_eq!(q.smallest_majority_index(), idx(0));

        // Now raise the new-only voters; the joint min catches up to 10.
        for n in [4u64, 5, 6] {
            q.update_match_index(&c, id(n), idx(0), idx(20));
        }
        assert_eq!(q.smallest_majority_index(), idx(10));
    }

    #[test]
    fn quorum_rebuild_starts_over_from_zero() {
        let c_old = cfg(&[1, 2, 3], &[], &[]);
        let mut q = Quorum::new(&c_old);
        q.update_match_index(&c_old, id(1), idx(0), idx(10));
        q.update_match_index(&c_old, id(2), idx(0), idx(20));
        q.update_match_index(&c_old, id(3), idx(0), idx(30));
        assert_eq!(q.smallest_majority_index(), idx(20));

        // Rebuild for a wider voter set: all followers start at 0 again
        // and re-applying the last-known index rebuilds the top-k.
        let c_new = cfg(&[1, 2, 3, 4, 5], &[], &[]);
        let mut q = Quorum::new(&c_new);
        assert_eq!(q.smallest_majority_index(), idx(0));
        q.update_match_index(&c_new, id(1), idx(0), idx(10));
        q.update_match_index(&c_new, id(2), idx(0), idx(20));
        q.update_match_index(&c_new, id(3), idx(0), idx(30));
        assert_eq!(q.smallest_majority_index(), idx(10));
    }

    #[test]
    fn non_voter_updates_do_not_affect_majority() {
        let c = cfg(&[1, 2, 3], &[], &[9, 10]);
        let mut q = Quorum::new(&c);
        let before = q.majority_indices.clone();

        q.update_match_index(&c, id(9), idx(0), idx(100));
        q.update_match_index(&c, id(10), idx(0), idx(200));

        assert_eq!(q.majority_indices, before);
        assert!(q.new_majority_indices.is_empty());
        assert_eq!(q.smallest_majority_index(), idx(0));
    }

    #[test]
    fn joint_old_and_new_voters_evolve_independently() {
        // Overlapping-but-distinct voter sets: node 3 belongs to both,
        // 1 and 2 are old-only, 4 and 5 are new-only.
        let c = cfg(&[1, 2, 3], &[3, 4, 5], &[]);
        let mut q = Quorum::new(&c);

        // Old-only voter advance touches the old set but not the new one.
        q.update_match_index(&c, id(1), idx(0), idx(100));
        assert_eq!(q.smallest_majority_index(), idx(0));

        // New-only voter advance touches the new set but not the old one.
        q.update_match_index(&c, id(4), idx(0), idx(50));
        q.update_match_index(&c, id(5), idx(0), idx(60));
        // Old majority is still bounded by the voter at index 0.
        // New majority top-k excludes the min, so its floor is 50.
        assert_eq!(q.smallest_majority_index(), idx(0));

        // The shared voter (3) advances: in the old set it evicts (0, 2),
        // while in the new set (30, 3) is a no-op because 30 <= the new
        // set's current min 50 (early return in update_majority).
        q.update_match_index(&c, id(3), idx(0), idx(30));
        // Old majority min is now min of (100 for id 1, 30 for id 3) = 30.
        // The new top-k stays {(50,4), (60,5)} (min 50); the joint min 30
        // comes from the old set {(30,3), (100,1)}.
        assert_eq!(q.smallest_majority_index(), idx(30));
    }

    #[test]
    fn smallest_majority_matches_naive_top_k() {
        let voters = [1u64, 2, 3, 4, 5];
        let c = cfg(&voters, &[], &[]);
        let mut q = Quorum::new(&c);
        let mut current: BTreeMap<u64, u64> = voters.iter().map(|&v| (v, 0)).collect();

        // Interleaved advances of different voters, including one that
        // moves twice; the incremental result must always match a naive
        // top-k over every voter's latest index.
        let updates: &[(u64, u64)] = &[
            (1, 5),
            (2, 10),
            (3, 15),
            (4, 20),
            (5, 25),
            (1, 30),
            (3, 40),
            (2, 35),
        ];

        for &(node, new_index) in updates {
            let old = current[&node];
            q.update_match_index(&c, id(node), idx(old), idx(new_index));
            current.insert(node, new_index);
            assert_eq!(
                q.smallest_majority_index().get(),
                naive_smallest_majority(&current),
                "after updating node {node} to {new_index}"
            );
        }
    }
}
