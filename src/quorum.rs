use crate::{config::ClusterConfig, log::LogIndex, node::NodeId};
use alloc::collections::BTreeSet;

#[derive(Debug, Clone)]
pub struct Quorum {
    majority_indices: BTreeSet<(LogIndex, NodeId)>,
    new_majority_indices: BTreeSet<(LogIndex, NodeId)>,
}

impl Quorum {
    pub fn new(config: &ClusterConfig) -> Self {
        debug_assert!(
            !config.voters.is_empty(),
            "a quorum requires at least one voter"
        );
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
        let i0 = self
            .majority_indices
            .first()
            .map(|(i, _)| *i)
            .expect("Quorum was constructed without any voter");
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

    if old_entry == new_entry {
        // A no-op update must not touch the set: otherwise the removal
        // below would drop an in-set entry and shrink the set below the
        // majority quota when that entry is not the current minimum.
        //
        // No current caller in this crate reaches this branch (every
        // callsite either supplies `old_index = LogIndex::ZERO` with a
        // fresh follower or requires `old < new` via a preceding strict
        // check). It is kept as a defensive guard for future callers and
        // is exercised by the model-based PBT.
        return;
    }

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
    use alloc::format;
    use alloc::string::String;
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

    // Naive oracle: the k-th largest index, where k = n/2 + 1.
    fn naive_majority_min(indices: &[u64]) -> u64 {
        debug_assert!(
            !indices.is_empty(),
            "naive_majority_min requires a non-empty voter set"
        );
        let mut sorted = indices.to_vec();
        sorted.sort_unstable();
        let majority = sorted.len() / 2 + 1;
        sorted[sorted.len() - majority]
    }

    // Samples the next index of a node: a no-op (same index), a small
    // advance, or the u64::MAX boundary. The fallback samples a `u64`
    // offset first and adds it with `saturating_add`, so the result
    // stays in `u64` space regardless of the target's `usize` width.
    fn sample_advance(ctx: &mut noprop::TestCaseContext, old: u64) -> u64 {
        noprop::sample_with_boundaries(
            ctx,
            &[old, old.saturating_add(1), u64::MAX],
            noprop::Ratio::one_nth(5),
            |ctx| {
                let offset = noprop::sample_usize_in(ctx, 0..=16) as u64;
                old.saturating_add(offset)
            },
        )
    }

    // Samples a count that gets extra weight on even sizes, the
    // singleton, and the maximum.
    fn sample_voter_count(ctx: &mut noprop::TestCaseContext, max: usize) -> usize {
        noprop::sample_with_boundaries(ctx, &[1, 2, 4, max], noprop::Ratio::one_nth(5), |ctx| {
            noprop::sample_usize_in(ctx, 1..=max)
        })
    }

    // Same shape as `sample_voter_count` but allows 0 (non-joint runs)
    // and shifts the extra weight to the non-joint boundary, the
    // singleton, and the maximum. Kept separate so that changing the
    // voter sampler cannot silently distort the new-voter distribution.
    fn sample_new_voter_count(ctx: &mut noprop::TestCaseContext, max: usize) -> usize {
        noprop::sample_with_boundaries(ctx, &[0, 1, 2, max], noprop::Ratio::one_nth(5), |ctx| {
            noprop::sample_usize_in(ctx, 0..=max)
        })
    }

    fn sample_non_voter_count(ctx: &mut noprop::TestCaseContext) -> usize {
        noprop::sample_with_boundaries(ctx, &[0, 1, 4], noprop::Ratio::one_nth(5), |ctx| {
            noprop::sample_usize_in(ctx, 0..=4)
        })
    }

    // Samples `count` distinct ids in `0..domain`. Callers keep `count`
    // at most half of `domain`, so the retry loop is short.
    fn sample_distinct_ids(
        ctx: &mut noprop::TestCaseContext,
        count: usize,
        domain: usize,
    ) -> Vec<u64> {
        debug_assert!(
            count <= domain,
            "sample_distinct_ids: count ({count}) must be <= domain ({domain})"
        );
        let mut ids = Vec::with_capacity(count);
        while ids.len() < count {
            let candidate = noprop::sample_usize_in(ctx, 0..domain) as u64;
            if !ids.contains(&candidate) {
                ids.push(candidate);
            }
        }
        ids
    }

    // Samples a bounded update count while giving empty, singleton, and
    // maximum lengths explicit probability.
    fn sample_steps(ctx: &mut noprop::TestCaseContext, max: usize) -> usize {
        noprop::sample_with_boundaries(ctx, &[0, 1, max], noprop::Ratio::one_nth(5), |ctx| {
            noprop::sample_usize_in(ctx, 2..max)
        })
    }

    // Compares `Quorum` against the naive model:
    // `smallest_majority_index` must equal the k-th largest index of
    // each voter set, and each set must hold exactly its majority
    // quota. The quota is asserted independently of the oracle so that
    // a change to the majority formula cannot silently break the
    // comparison. The Err payload carries the full state so a failing
    // seed can be triaged without re-running.
    //
    // The internal identity of `majority_indices` (which ids happen to
    // be inside the top-k when several voters share the same index) is
    // *not* checked: `Quorum::new` initialises the set from a
    // `BTreeSet<NodeId>` iterator (ascending id order), while
    // `update_majority` prefers larger ids on tie eviction. Both
    // representations yield the same `smallest_majority_index` under
    // the current semantics, and only that value is a public contract.
    fn check_against_naive(
        q: &Quorum,
        voters: &[u64],
        new_voters: &[u64],
        current: &BTreeMap<u64, u64>,
        step: usize,
    ) -> Result<(), String> {
        let voter_indices: Vec<u64> = voters.iter().map(|id| current[id]).collect();
        let mut expected = naive_majority_min(&voter_indices);
        if !new_voters.is_empty() {
            let new_indices: Vec<u64> = new_voters.iter().map(|id| current[id]).collect();
            expected = expected.min(naive_majority_min(&new_indices));
        }
        let actual = q.smallest_majority_index().get();
        if actual != expected {
            return Err(format!(
                "step {step}: smallest_majority_index = {actual}, expected {expected}\n  \
                 voters = {voters:?}, new_voters = {new_voters:?}, current = {current:?}\n  \
                 majority_indices = {:?}, new_majority_indices = {:?}",
                q.majority_indices, q.new_majority_indices
            ));
        }
        if q.majority_indices.len() != voters.len() / 2 + 1 {
            return Err(format!(
                "step {step}: majority set size = {}, expected {}\n  \
                 voters = {voters:?}, current = {current:?}\n  \
                 majority_indices = {:?}",
                q.majority_indices.len(),
                voters.len() / 2 + 1,
                q.majority_indices,
            ));
        }
        if new_voters.is_empty() {
            if !q.new_majority_indices.is_empty() {
                return Err(format!(
                    "step {step}: new majority set must be empty\n  \
                     new_majority_indices = {:?}",
                    q.new_majority_indices
                ));
            }
        } else if q.new_majority_indices.len() != new_voters.len() / 2 + 1 {
            return Err(format!(
                "step {step}: new majority set size = {}, expected {}\n  \
                 new_voters = {new_voters:?}, current = {current:?}\n  \
                 new_majority_indices = {:?}",
                q.new_majority_indices.len(),
                new_voters.len() / 2 + 1,
                q.new_majority_indices,
            ));
        }
        Ok(())
    }

    // Guards against silently removing the `debug_assert!` in
    // `Quorum::new`: without it a caller passing an empty voter set
    // would build a broken `Quorum` and blow up later inside
    // `smallest_majority_index`'s `unreachable!()`.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "at least one voter")]
    fn empty_voters_config_is_rejected() {
        let _ = Quorum::new(&cfg(&[], &[], &[]));
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

        // Even voter counts: the majority is n/2 + 1, not n/2.
        let q = Quorum::new(&cfg(&[1, 2], &[], &[]));
        assert_eq!(q.majority_indices.len(), 2);

        let q = Quorum::new(&cfg(&[1, 2, 3, 4], &[], &[]));
        assert_eq!(q.majority_indices.len(), 3);

        // In joint consensus each voter set has its own majority quota.
        let q = Quorum::new(&cfg(&[1, 2, 3], &[1, 2, 3, 4, 5], &[]));
        assert_eq!(q.majority_indices.len(), 2);
        assert_eq!(q.new_majority_indices.len(), 3);

        let q = Quorum::new(&cfg(&[1, 2, 3, 4], &[1, 2, 3, 4, 5, 6], &[]));
        assert_eq!(q.majority_indices.len(), 3);
        assert_eq!(q.new_majority_indices.len(), 4);
    }

    #[test]
    fn even_voter_majority_requires_half_plus_one() {
        // Two voters: both must advance before the majority moves.
        let c = cfg(&[1, 2], &[], &[]);
        let mut q = Quorum::new(&c);
        q.update_match_index(&c, id(1), idx(0), idx(10));
        assert_eq!(q.smallest_majority_index(), idx(0));
        q.update_match_index(&c, id(2), idx(0), idx(20));
        assert_eq!(q.smallest_majority_index(), idx(10));

        // Four voters: three must advance, two are not enough.
        let c = cfg(&[1, 2, 3, 4], &[], &[]);
        let mut q = Quorum::new(&c);
        q.update_match_index(&c, id(1), idx(0), idx(10));
        q.update_match_index(&c, id(2), idx(0), idx(20));
        assert_eq!(q.smallest_majority_index(), idx(0));
        q.update_match_index(&c, id(3), idx(0), idx(30));
        assert_eq!(q.smallest_majority_index(), idx(10));
    }

    #[test]
    fn in_set_voter_same_index_update_is_a_no_op() {
        // The scenario must exercise the path guarded *only* by the no-op
        // check in `update_majority` (i.e. an in-set, non-min voter is
        // re-applied at the same index that is strictly greater than the
        // set's current min). If the guard is removed, `insert` becomes a
        // no-op while `remove(&old_entry)` succeeds, shrinking the set
        // below the majority quota.
        let c = cfg(&[1, 2, 3, 4, 5], &[], &[]);
        let mut q = Quorum::new(&c);
        q.update_match_index(&c, id(4), idx(0), idx(10));
        assert_eq!(
            q.majority_indices.iter().copied().collect::<Vec<_>>(),
            [(idx(0), id(2)), (idx(0), id(3)), (idx(10), id(4))]
        );

        // `(10, 4)` is in the set and is not the min. Re-applying the same
        // index must leave the set intact.
        q.update_match_index(&c, id(4), idx(10), idx(10));
        assert_eq!(
            q.majority_indices.iter().copied().collect::<Vec<_>>(),
            [(idx(0), id(2)), (idx(0), id(3)), (idx(10), id(4))]
        );
        assert_eq!(q.smallest_majority_index(), idx(0));
    }

    // This test is independent of the no-op guard added in
    // `update_majority`: it pins the success path of
    // `set.remove(&old_entry)` when the removed entry is in the set but
    // is not the current minimum. If the removal branch is broken and
    // `pop_first` is always taken, (0, 2) would be evicted and the
    // result would still have size 3 and min 0; only the exact set
    // contents distinguish the two branches.
    #[test]
    fn advancing_in_set_non_min_voter_removes_its_old_entry() {
        let c = cfg(&[1, 2, 3, 4, 5], &[], &[]);
        let mut q = Quorum::new(&c);
        q.update_match_index(&c, id(4), idx(0), idx(10));
        assert_eq!(
            q.majority_indices.iter().copied().collect::<Vec<_>>(),
            [(idx(0), id(2)), (idx(0), id(3)), (idx(10), id(4))]
        );

        q.update_match_index(&c, id(3), idx(0), idx(5));
        assert_eq!(
            q.majority_indices.iter().copied().collect::<Vec<_>>(),
            [(idx(0), id(2)), (idx(5), id(3)), (idx(10), id(4))]
        );
        assert_eq!(q.smallest_majority_index(), idx(0));
    }

    #[test]
    fn advancing_a_voter_outside_top_k_evicts_the_min() {
        let c = cfg(&[1, 2, 3, 4, 5], &[], &[]);
        let mut q = Quorum::new(&c);
        // `Quorum::new` takes the first majority-sized entries of
        // `config.voters` (a BTreeSet<NodeId>, iterated in ascending id
        // order), so the initial top-k holds the three smallest ids.
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
    fn same_index_tie_break_evicts_smaller_node_id_first() {
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
    fn constructing_a_wider_quorum_starts_from_zero() {
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

    // Model-based property: `Quorum::smallest_majority_index` always
    // equals the k-th largest index among the voters (k = n/2 + 1) of
    // each voter set, for random configurations (1..=8 voters with even
    // sizes biased, optional joint consensus with overlapping voter
    // sets, and disjoint non-voters) and random monotonically increasing
    // update sequences, including no-op and u64::MAX updates.
    //
    // `Quorum` is crate-private, so unlike the other PBTs this property
    // lives here instead of tests/pbt_*.rs and follows the crate's
    // NORAFT_PBT_SEED reproduction convention directly.
    #[test]
    fn quorum_matches_naive_model() -> noprop::TestResult {
        let seed = noprop::seed_from_env_or_time("NORAFT_PBT_SEED")?;
        let advanced_min_cases = core::cell::Cell::new(0usize);
        let joint_cases = core::cell::Cell::new(0usize);
        let non_joint_cases = core::cell::Cell::new(0usize);
        let non_voter_update_cases = core::cell::Cell::new(0usize);
        let mut runner = noprop::Runner::new(seed);

        runner.run(1024, |ctx| {
            let voter_count = sample_voter_count(ctx, 8);
            let voters: Vec<u64> = sample_distinct_ids(ctx, voter_count, 16);
            let new_voter_count = sample_new_voter_count(ctx, 8);
            let new_voters: Vec<u64> = sample_distinct_ids(ctx, new_voter_count, 16);
            let non_voter_count = sample_non_voter_count(ctx);
            let non_voters: Vec<u64> = sample_distinct_ids(ctx, non_voter_count, 8)
                .into_iter()
                .map(|id| id + 16)
                .collect();
            let config = cfg(&voters, &new_voters, &non_voters);
            let pool: Vec<u64> = voters.iter().chain(&new_voters).copied().collect();
            let mut current: BTreeMap<u64, u64> = pool.iter().map(|&id| (id, 0)).collect();

            let mut q = Quorum::new(&config);
            check_against_naive(&q, &voters, &new_voters, &current, 0)?;

            let steps = sample_steps(ctx, 64);
            for step in 1..=steps {
                if !non_voters.is_empty() && noprop::sample_ratio(ctx, noprop::Ratio::one_nth(5)) {
                    let node_id = noprop::sample_choice(ctx, &non_voters);
                    let index = sample_advance(ctx, 0);
                    q.update_match_index(&config, id(node_id), idx(0), idx(index));
                    non_voter_update_cases.set(non_voter_update_cases.get() + 1);
                } else {
                    let node_id = noprop::sample_choice(ctx, &pool);
                    let old = current[&node_id];
                    let index = sample_advance(ctx, old);
                    q.update_match_index(&config, id(node_id), idx(old), idx(index));
                    current.insert(node_id, index);
                }
                check_against_naive(&q, &voters, &new_voters, &current, step)?;
            }

            if q.smallest_majority_index().get() > 0 {
                advanced_min_cases.set(advanced_min_cases.get() + 1);
            }
            if new_voters.is_empty() {
                non_joint_cases.set(non_joint_cases.get() + 1);
            } else {
                joint_cases.set(joint_cases.get() + 1);
            }
            Ok(())
        })?;

        assert!(
            advanced_min_cases.get() > 0,
            "no case advanced the smallest majority index\n{runner}"
        );
        assert!(
            joint_cases.get() > 0,
            "no case exercised a joint consensus configuration\n{runner}"
        );
        assert!(
            non_joint_cases.get() > 0,
            "no case exercised a non-joint configuration\n{runner}"
        );
        assert!(
            non_voter_update_cases.get() > 0,
            "no case exercised a non-voter update\n{runner}"
        );
        Ok(())
    }
}
