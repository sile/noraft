//! Stateless property-based tests for noraft's pure APIs, driven by
//! noprop (https://github.com/sile/noprop).
//!
//! Seed / case budget come from the `NORAFT_PBT_SEED` and
//! `NORAFT_PBT_CASES` environment variables; unset means
//! "clock-derived seed" and "1024 cases". A failing seed can be
//! re-run with:
//!
//! ```text
//! NORAFT_PBT_SEED=<seed> cargo test --test prop_stateless_test
//! ```

use noraft::{ClusterConfig, Log, LogEntries, LogEntry, LogIndex, LogPosition, NodeId, Term};
use std::cell::Cell;

/// Runs a property with the standard seed / case-budget handling.
fn run_prop(
    cases: usize,
    property: impl Fn(&mut noprop::TestCaseContext) -> Result<(), Box<dyn std::error::Error>>,
) -> noprop::TestResult {
    let seed = noprop::seed_from_env_or_time("NORAFT_PBT_SEED")?;
    noprop::Runner::new(seed).run(cases, property)?;
    Ok(())
}

/// Reads the case-budget environment variable with a default fallback.
fn cases_from_env(default: usize) -> usize {
    std::env::var("NORAFT_PBT_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Samples a log entry, weighted toward `Command` (the most common
/// entry in real clusters).
fn sample_entry(ctx: &mut noprop::TestCaseContext) -> LogEntry {
    match noprop::sample_weighted_index(ctx, &[3, 1, 1]) {
        0 => LogEntry::Command,
        1 => LogEntry::Term(Term::new(noprop::sample_usize_in(ctx, 0..16) as u64)),
        _ => LogEntry::ClusterConfig(sample_config(ctx)),
    }
}

/// Samples a cluster config whose three node sets are pairwise disjoint.
///
/// Valid-by-construction: each node of the pool is inserted into at
/// most one of the three sets, so no node can be a member of two sets.
fn sample_config(ctx: &mut noprop::TestCaseContext) -> ClusterConfig {
    let pool: [NodeId; 6] = [
        NodeId::new(0),
        NodeId::new(1),
        NodeId::new(2),
        NodeId::new(3),
        NodeId::new(4),
        NodeId::new(5),
    ];
    let mut config = ClusterConfig::new();
    for id in pool {
        match noprop::sample_usize_in(ctx, 0..4) {
            0 => {
                config.voters.insert(id);
            }
            1 => {
                config.new_voters.insert(id);
            }
            2 => {
                config.non_voters.insert(id);
            }
            _ => {}
        }
    }
    config
}

/// Samples `LogEntries` whose previous position is the last position
/// of `base` (i.e., the result can be appended to `base`).
///
/// The entry count is drawn with boundary emphasis so that empty and
/// small suffix cases occur with meaningful probability.
fn sample_entries_from(ctx: &mut noprop::TestCaseContext, base: &LogEntries) -> LogEntries {
    let count = noprop::sample_usize_in(ctx, 0..=16);
    LogEntries::from_iter(base.last_position(), (0..count).map(|_| sample_entry(ctx)))
}

/// Samples a random position within the range of `entries`
/// (inclusive of the previous position and the last position).
fn sample_contained_position(
    ctx: &mut noprop::TestCaseContext,
    entries: &LogEntries,
) -> LogPosition {
    let prev = entries.prev_position();
    let last = entries.last_position();
    let index = LogIndex::new(noprop::sample_usize_in(
        ctx,
        prev.index.get() as usize..=last.index.get() as usize,
    ) as u64);
    LogPosition {
        term: entries.get_term(index).expect("index is in range"),
        index,
    }
}

/// `iter_with_positions` must yield monotonically increasing indices
/// and entries that are readable back via `get_entry`.
///
/// The position term of an entry is the term of the most recent
/// `LogEntry::Term` at or before its index.
#[test]
fn log_entries_iteration_consistency() -> noprop::TestResult {
    let cases_with_entries: Cell<usize> = Cell::new(0);
    run_prop(cases_from_env(1024), |ctx| {
        let mut entries = LogEntries::new(LogPosition::ZERO);
        let count = noprop::sample_usize_in(ctx, 0..=64);
        for _ in 0..count {
            entries.push(sample_entry(ctx));
        }

        let mut prev = LogPosition::ZERO;
        for (position, entry) in entries.iter_with_positions() {
            if position.index != prev.index.next() {
                return Err(
                    format!("indices must advance by one: {position:?} after {prev:?}").into(),
                );
            }
            // A `Term` entry sets the term of its position; other
            // entries keep the term of the previous entry.
            let expected_term = match entry {
                LogEntry::Term(term) => term,
                _ => prev.term,
            };
            if position.term != expected_term {
                return Err(format!(
                    "position term mismatch: {position:?}, expected {expected_term:?}"
                )
                .into());
            }
            let fetched = entries
                .get_entry(position.index)
                .ok_or_else(|| format!("entry at {position:?} must be readable back"))?;
            if fetched != entry {
                return Err(format!(
                    "get_entry({:?}) returned {fetched:?}, expected {entry:?}",
                    position.index
                )
                .into());
            }
            prev = position;
        }
        if count > 0 {
            cases_with_entries.set(cases_with_entries.get() + 1);
            if entries.last_position() != prev {
                return Err(format!(
                    "last_position {:?} must equal the last iterated position {prev:?}",
                    entries.last_position()
                )
                .into());
            }
        }
        Ok(())
    })?;
    assert!(
        cases_with_entries.get() > 0,
        "no case exercised a non-empty LogEntries"
    );
    Ok(())
}

/// `LogEntries::truncate` must keep the first `len` entries unchanged
/// and must be a no-op when `len` is at least the current length.
#[test]
fn log_entries_truncate_keeps_prefix() -> noprop::TestResult {
    let cases_truncating: Cell<usize> = Cell::new(0);
    run_prop(cases_from_env(1024), |ctx| {
        let mut entries = LogEntries::new(LogPosition::ZERO);
        let count = noprop::sample_usize_in(ctx, 0..=32);
        for _ in 0..count {
            entries.push(sample_entry(ctx));
        }
        let len = noprop::sample_usize_in(ctx, 0..=count);

        let before: Vec<(LogPosition, LogEntry)> = entries.iter_with_positions().collect();
        let last_position = entries.last_position();
        entries.truncate(len);

        if len >= count {
            if entries.len() != count {
                return Err(format!(
                    "truncate({len}) must be a no-op, but len became {}",
                    entries.len()
                )
                .into());
            }
        } else {
            cases_truncating.set(cases_truncating.get() + 1);
            if entries.len() != len {
                return Err(format!(
                    "truncate({len}) must leave {len} entries, but len became {}",
                    entries.len()
                )
                .into());
            }
            // The surviving prefix must be unchanged, and the dropped
            // tail must be the last `count - len` entries.
            let after: Vec<(LogPosition, LogEntry)> = entries.iter_with_positions().collect();
            if after != before[..len] {
                return Err(format!(
                    "truncate({len}) changed the prefix: before={before:?}, after={after:?}"
                )
                .into());
            }
            let _ = last_position;
        }
        Ok(())
    })?;
    assert!(
        cases_truncating.get() > 0,
        "no case exercised an actual truncation"
    );
    Ok(())
}

/// `LogEntries::since` must return the suffix after a contained
/// position, with the correct previous / last positions, and must
/// return `None` for a position whose index is in range but whose
/// term does not match.
#[test]
fn log_entries_since_returns_tail_suffix() -> noprop::TestResult {
    let cases_with_suffix: Cell<usize> = Cell::new(0);
    run_prop(cases_from_env(1024), |ctx| {
        let mut entries = LogEntries::new(LogPosition::ZERO);
        let count = noprop::sample_usize_in(ctx, 0..=32);
        for _ in 0..count {
            entries.push(sample_entry(ctx));
        }
        let position = sample_contained_position(ctx, &entries);

        let Some(suffix) = entries.since(position) else {
            return Err(format!("since({position:?}) must succeed on a contained position").into());
        };
        if suffix.prev_position() != position {
            return Err(format!(
                "suffix prev_position must be {position:?}, got {:?}",
                suffix.prev_position()
            )
            .into());
        }
        if suffix.last_position() != entries.last_position() {
            return Err(format!(
                "suffix last_position must be {:?}, got {:?}",
                entries.last_position(),
                suffix.last_position()
            )
            .into());
        }
        let expected: Vec<(LogPosition, LogEntry)> = entries
            .iter_with_positions()
            .skip_while(|(pos, _)| pos.index.get() <= position.index.get())
            .collect();
        let actual: Vec<(LogPosition, LogEntry)> = suffix.iter_with_positions().collect();
        if actual != expected {
            return Err(format!(
                "suffix must equal the tail after {position:?}: expected {expected:?}, got {actual:?}"
            )
            .into());
        }
        // The suffix must be re-appendable to its own previous
        // position, rebuilding the tail.
        if !suffix.is_empty() {
            cases_with_suffix.set(cases_with_suffix.get() + 1);
            let mut rebuilt = LogEntries::new(position);
            rebuilt.append_suffix(&suffix);
            if rebuilt.iter_with_positions().collect::<Vec<_>>() != actual {
                return Err("suffix must round-trip through append_suffix".into());
            }
        }
        // A term-mismatched position within the index range must be
        // rejected.
        if !entries.is_empty() {
            let mismatched = LogPosition {
                term: Term::new(position.term.get().wrapping_add(1)),
                index: position.index,
            };
            if entries.since(mismatched).is_some() {
                return Err(
                    format!("since({mismatched:?}) must return None on a term mismatch").into(),
                );
            }
        }
        Ok(())
    })?;
    assert!(
        cases_with_suffix.get() > 0,
        "no case exercised a non-empty suffix"
    );
    Ok(())
}

/// `Log::append_suffix` must advance `last_position` to the suffix's
/// last position when the suffix fits, and leave the log unchanged
/// otherwise.
#[test]
fn log_append_suffix_updates_last_position() -> noprop::TestResult {
    let cases_appended: Cell<usize> = Cell::new(0);
    let cases_rejected: Cell<usize> = Cell::new(0);
    run_prop(cases_from_env(1024), |ctx| {
        let mut entries = LogEntries::new(LogPosition::ZERO);
        let count = noprop::sample_usize_in(ctx, 0..=32);
        for _ in 0..count {
            entries.push(sample_entry(ctx));
        }
        let mut log = Log::new(ClusterConfig::new(), entries);

        // Build a suffix whose `prev_position` may or may not be
        // contained in the log. The anchor is in-range most of the
        // time and out of range otherwise, so both the append and the
        // rejection path are exercised.
        let mut suffix = sample_entries_from(ctx, &LogEntries::new(LogPosition::ZERO));
        let anchor_index = match noprop::sample_weighted_index(ctx, &[4, 1]) {
            0 => LogIndex::new(noprop::sample_usize_in(
                ctx,
                log.entries().prev_position().index.get() as usize
                    ..=log.last_position().index.get() as usize,
            ) as u64),
            _ => LogIndex::new(
                log.last_position().index.get() + 1 + noprop::sample_usize_in(ctx, 0..4) as u64,
            ),
        };
        let anchor_term = log
            .entries()
            .get_term(anchor_index)
            .unwrap_or(log.entries().prev_position().term);
        suffix = LogEntries::from_iter(
            LogPosition {
                term: anchor_term,
                index: anchor_index,
            },
            suffix.iter(),
        );

        let before = log.last_position();
        let appended = log.append_suffix(&suffix);
        if log.entries().contains(suffix.prev_position()) {
            if !appended {
                return Err(format!(
                    "append_suffix must succeed when prev_position {:?} is contained",
                    suffix.prev_position()
                )
                .into());
            }
            cases_appended.set(cases_appended.get() + 1);
            if log.last_position() != suffix.last_position() {
                return Err(format!(
                    "last_position {:?} must become the suffix's last_position {:?}",
                    log.last_position(),
                    suffix.last_position()
                )
                .into());
            }
        } else {
            if appended || log.last_position() != before {
                return Err(format!(
                    "append_suffix must fail without changing the log when prev_position {:?} \
                     is not contained",
                    suffix.prev_position()
                )
                .into());
            }
            cases_rejected.set(cases_rejected.get() + 1);
        }
        Ok(())
    })?;
    assert!(
        cases_appended.get() > 0 && cases_rejected.get() > 0,
        "no case exercised both the append and the rejection path"
    );
    Ok(())
}

/// `Log::latest_config` must return the most recent `ClusterConfig`
/// entry in the log, falling back to the snapshot config when there
/// is none.
///
/// `Log::get_position_and_config` must return the position and
/// config at a contained index, and `None` out of range.
#[test]
fn log_latest_config_and_get_position_and_config() -> noprop::TestResult {
    let cases_with_config: Cell<usize> = Cell::new(0);
    run_prop(cases_from_env(1024), |ctx| {
        let snapshot_config = sample_config(ctx);
        let mut entries = LogEntries::new(LogPosition::ZERO);
        let count = noprop::sample_usize_in(ctx, 0..=32);
        let mut last_config = snapshot_config.clone();
        for _ in 0..count {
            let entry = sample_entry(ctx);
            if let LogEntry::ClusterConfig(config) = &entry {
                last_config = config.clone();
                cases_with_config.set(cases_with_config.get() + 1);
            }
            entries.push(entry);
        }
        let log = Log::new(snapshot_config.clone(), entries);

        if log.latest_config() != &last_config {
            return Err(format!(
                "latest_config must be the most recent config entry: got {:?}, expected \
                 {last_config:?}",
                log.latest_config()
            )
            .into());
        }

        // A contained index must return its position and the config
        // in effect at that index.
        let index = sample_contained_position(ctx, log.entries()).index;
        let (position, config) = log
            .get_position_and_config(index)
            .expect("contained index must resolve");
        if position.index != index {
            return Err(format!(
                "get_position_and_config({index:?}) must return index {index:?}, got {:?}",
                position.index
            )
            .into());
        }
        if position.term != log.entries().get_term(index).expect("contained index") {
            return Err(format!(
                "get_position_and_config({index:?}) must return the entry term, got {:?}",
                position.term
            )
            .into());
        }
        let mut expected_config = snapshot_config.clone();
        for (pos, entry) in log.entries().iter_with_positions() {
            if pos.index > index {
                break;
            }
            if let LogEntry::ClusterConfig(config) = entry {
                expected_config = config;
            }
        }
        if config != &expected_config {
            return Err(format!(
                "get_position_and_config({index:?}) must return the config in effect at that \
                 index: got {config:?}, expected {expected_config:?}"
            )
            .into());
        }

        // An out-of-range index must return None.
        let beyond = LogIndex::new(log.last_position().index.get().saturating_add(1));
        if log.get_position_and_config(beyond).is_some() {
            return Err(format!(
                "get_position_and_config({beyond:?}) must return None for an out-of-range index"
            )
            .into());
        }
        Ok(())
    })?;
    assert!(
        cases_with_config.get() > 0,
        "no case exercised a log with a ClusterConfig entry"
    );
    Ok(())
}

/// The three node sets of a `ClusterConfig` must be pairwise disjoint
/// and `unique_nodes` must be exactly their union.
#[test]
fn cluster_config_sets_are_disjoint_and_complete() -> noprop::TestResult {
    run_prop(cases_from_env(1024), |ctx| {
        let config = sample_config(ctx);
        let unique: Vec<NodeId> = config.unique_nodes().collect();

        // Pairwise disjoint.
        for id in &unique {
            let in_voters = config.voters.contains(id);
            let in_new = config.new_voters.contains(id);
            let in_non = config.non_voters.contains(id);
            let membership = [in_voters, in_new, in_non].iter().filter(|b| **b).count();
            if membership != 1 {
                return Err(format!(
                    "node {id:?} must belong to exactly one set (voters / new_voters / \
                     non_voters)"
                )
                .into());
            }
        }

        // Complete: every node of the three sets appears exactly once
        // in `unique_nodes`.
        let expected_len = config.voters.len() + config.new_voters.len() + config.non_voters.len();
        if unique.len() != expected_len {
            return Err(format!(
                "unique_nodes yields {} nodes, expected {expected_len}",
                unique.len()
            )
            .into());
        }
        if config.is_joint_consensus() != !config.new_voters.is_empty() {
            return Err("is_joint_consensus must reflect a non-empty new_voters".into());
        }
        Ok(())
    })
}

/// `to_joint_consensus` must set `new_voters` to the old voters plus
/// the additions minus the removals, and must keep `voters` as the old
/// configuration during the joint consensus.
#[test]
fn cluster_config_to_joint_consensus() -> noprop::TestResult {
    run_prop(cases_from_env(1024), |ctx| {
        let base = sample_config(ctx);
        // Sample additions / removals from the pool of candidate node
        // ids.
        let mut adding = Vec::new();
        let mut removing = Vec::new();
        for i in 0..4u64 {
            match noprop::sample_usize_in(ctx, 0..3) {
                0 => adding.push(NodeId::new(100 + i)),
                1 => removing.push(NodeId::new(i)),
                _ => {}
            }
        }
        let joint = base.to_joint_consensus(&adding, &removing);
        let mut expected: Vec<NodeId> = base.voters.iter().copied().collect();
        expected.extend(adding.iter().copied());
        expected.retain(|id| !removing.contains(id));
        expected.sort_unstable();
        expected.dedup();
        let actual: Vec<NodeId> = joint.new_voters.iter().copied().collect();
        if actual != expected {
            return Err(format!(
                "new_voters must be voters + additions - removals: got {actual:?}, \
                 expected {expected:?}"
            )
            .into());
        }
        // `voters` must stay as the old configuration during joint
        // consensus.
        if joint.voters != base.voters {
            return Err("voters must stay unchanged during joint consensus".into());
        }
        Ok(())
    })
}

/// `LogPosition::next` advances the index by one and keeps the term.
#[test]
fn log_position_next_advances_index_only() -> noprop::TestResult {
    run_prop(cases_from_env(1024), |ctx| {
        let term = Term::new(noprop::sample_usize_in(ctx, 0..1 << 20) as u64);
        let index = LogIndex::new(noprop::sample_usize_in(ctx, 0..1 << 20) as u64);
        let position = LogPosition::new(term, index);
        let next = position.next();
        if next.index != index.next() {
            return Err(format!("next must advance the index: {next:?}").into());
        }
        if next.term != term {
            return Err(format!("next must keep the term: {next:?} vs {term:?}").into());
        }
        Ok(())
    })
}

/// `LogIndex::next` advances the index by one.
#[test]
fn log_index_next_advances_by_one() -> noprop::TestResult {
    run_prop(cases_from_env(1024), |ctx| {
        let index = LogIndex::new(noprop::sample_usize_in(ctx, 0..1 << 20) as u64);
        let next = index.next();
        if next.get() != index.get() + 1 {
            return Err(
                format!("next must advance the index by one: {next:?} after {index:?}").into(),
            );
        }
        Ok(())
    })
}

/// `Ratio` must validate its numerator / denominator at construction
/// time: a numerator exceeding the denominator or a zero denominator
/// panics. This guards against accidental misuse in test generators.
#[test]
#[should_panic]
fn ratio_rejects_invalid_construction() {
    let _ = noprop::Ratio::new(3, 2);
}
