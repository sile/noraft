//! Model-based PBTs for noraft log APIs.

pub mod helpers;

use helpers::pbt::{run, sample_config, sample_len, sample_log_entry, sample_u64_before_max};
use noraft::{Log, LogEntries, LogEntry, LogIndex, LogPosition, Term};

fn sample_entries(ctx: &mut noprop::TestCaseContext, max_len: usize) -> LogEntries {
    let count = sample_len(ctx, max_len);
    LogEntries::from_iter(LogPosition::ZERO, (0..count).map(|_| sample_log_entry(ctx)))
}

fn sample_entries_from(
    ctx: &mut noprop::TestCaseContext,
    prev_position: LogPosition,
    max_len: usize,
) -> LogEntries {
    let count = sample_len(ctx, max_len);
    LogEntries::from_iter(prev_position, (0..count).map(|_| sample_log_entry(ctx)))
}

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
    LogPosition::new(
        entries
            .get_term(index)
            .expect("a sampled contained index must have a term"),
        index,
    )
}

fn different_term(term: Term) -> Term {
    if term.get() == u64::MAX {
        Term::ZERO
    } else {
        Term::new(term.get() + 1)
    }
}

/// Iteration, indexed lookup, and the reported last position must
/// describe the same entry sequence.
#[test]
fn log_entries_iteration_matches_indexed_lookup() -> noprop::TestResult {
    run(1024, |ctx| {
        let entries = sample_entries(ctx, 64);
        let iterated: Vec<(LogPosition, LogEntry)> = entries
            .iter_with_positions()
            .map(|(position, entry)| (position, entry.clone()))
            .collect();

        if iterated.len() != entries.len() {
            return Err(format!(
                "iteration returned {} entries, expected {}",
                iterated.len(),
                entries.len()
            )
            .into());
        }

        let mut previous = entries.prev_position();
        for (position, entry) in &iterated {
            if position.index != previous.index.next() {
                return Err(format!(
                    "indices must advance by one: {position:?} after {previous:?}"
                )
                .into());
            }
            let expected_term = match entry {
                LogEntry::Term(term) => *term,
                _ => previous.term,
            };
            if position.term != expected_term {
                return Err(format!(
                    "position term mismatch: {position:?}, expected {expected_term:?}"
                )
                .into());
            }
            if entries.get_entry(position.index) != Some(entry.clone()) {
                return Err(
                    format!("get_entry({:?}) did not return {entry:?}", position.index).into(),
                );
            }
            previous = *position;
        }

        if entries.last_position() != previous {
            return Err(format!(
                "last_position {:?} does not match the iterated tail {previous:?}",
                entries.last_position()
            )
            .into());
        }
        Ok(())
    })
}

/// Truncation must preserve exactly the requested prefix, including
/// its entry values and final position.
#[test]
fn log_entries_truncate_matches_prefix_model() -> noprop::TestResult {
    run(1024, |ctx| {
        let mut entries = sample_entries(ctx, 32);
        let before: Vec<(LogPosition, LogEntry)> = entries
            .iter_with_positions()
            .map(|(position, entry)| (position, entry.clone()))
            .collect();
        let max_requested = before.len() + 4;
        let len = noprop::sample_with_boundaries(
            ctx,
            &[0, before.len(), max_requested],
            noprop::Ratio::one_nth(5),
            |ctx| noprop::sample_usize_in(ctx, 0..=max_requested),
        );
        let expected = &before[..len.min(before.len())];
        let expected_last = expected
            .last()
            .map_or(entries.prev_position(), |(position, _)| *position);

        entries.truncate(len);
        let actual: Vec<(LogPosition, LogEntry)> = entries
            .iter_with_positions()
            .map(|(position, entry)| (position, entry.clone()))
            .collect();
        if actual != expected {
            return Err(
                format!("truncate({len}) produced {actual:?}, expected {expected:?}").into(),
            );
        }
        if entries.last_position() != expected_last {
            return Err(format!(
                "truncate({len}) ended at {:?}, expected {expected_last:?}",
                entries.last_position()
            )
            .into());
        }
        Ok(())
    })
}

/// `since` must return the exact suffix after a contained position and
/// reject the same index paired with a different term.
#[test]
fn log_entries_since_matches_suffix_model() -> noprop::TestResult {
    run(1024, |ctx| {
        let entries = sample_entries(ctx, 32);
        let position = sample_contained_position(ctx, &entries);
        let suffix = entries
            .since(position)
            .ok_or_else(|| format!("since({position:?}) rejected a contained position"))?;
        let expected: Vec<(LogPosition, LogEntry)> = entries
            .iter_with_positions()
            .filter(|(candidate, _)| candidate.index > position.index)
            .map(|(candidate, entry)| (candidate, entry.clone()))
            .collect();
        let actual: Vec<(LogPosition, LogEntry)> = suffix
            .iter_with_positions()
            .map(|(candidate, entry)| (candidate, entry.clone()))
            .collect();

        if suffix.prev_position() != position || actual != expected {
            return Err(format!(
                "since({position:?}) produced prev={:?}, entries={actual:?}; expected {expected:?}",
                suffix.prev_position()
            )
            .into());
        }
        if suffix.last_position() != entries.last_position() {
            return Err(format!(
                "suffix ended at {:?}, expected {:?}",
                suffix.last_position(),
                entries.last_position()
            )
            .into());
        }

        let mismatched = LogPosition::new(different_term(position.term), position.index);
        if entries.since(mismatched).is_some() {
            return Err(format!("since({mismatched:?}) accepted a mismatched term").into());
        }
        Ok(())
    })
}

/// A valid suffix replaces the local tail after its anchor; an invalid
/// anchor leaves the complete log unchanged.
#[test]
fn log_append_suffix_matches_replacement_model() -> noprop::TestResult {
    run(1024, |ctx| {
        let snapshot_config = sample_config(ctx);
        let original = Log::new(snapshot_config.clone(), sample_entries(ctx, 32));
        let anchor = sample_contained_position(ctx, original.entries());
        let suffix = sample_entries_from(ctx, anchor, 16);

        let expected_entries = original
            .entries()
            .iter_with_positions()
            .take_while(|(position, _)| position.index <= anchor.index)
            .map(|(_, entry)| entry.clone())
            .chain(suffix.iter());
        let expected = Log::new(
            snapshot_config,
            LogEntries::from_iter(original.snapshot_position(), expected_entries),
        );
        let mut actual = original.clone();
        if !actual.append_suffix(&suffix) {
            return Err(format!("append_suffix rejected valid anchor {anchor:?}").into());
        }
        if actual != expected {
            return Err(format!(
                "append_suffix result mismatch: actual={actual:?}, expected={expected:?}"
            )
            .into());
        }

        let invalid_anchor = LogPosition::new(Term::ZERO, original.last_position().index.next());
        let invalid_suffix = sample_entries_from(ctx, invalid_anchor, 16);
        let mut unchanged = original.clone();
        if unchanged.append_suffix(&invalid_suffix) || unchanged != original {
            return Err("append_suffix changed a log for an invalid anchor".into());
        }
        Ok(())
    })
}

/// `latest_config` and `get_position_and_config` must agree with a
/// straightforward scan of the log.
#[test]
fn log_config_queries_match_scan_model() -> noprop::TestResult {
    run(1024, |ctx| {
        let snapshot_config = sample_config(ctx);
        let entries = sample_entries(ctx, 32);
        let log = Log::new(snapshot_config.clone(), entries);
        let mut latest = snapshot_config.clone();
        for (_, entry) in log.entries().iter_with_positions() {
            if let LogEntry::ClusterConfig(config) = entry {
                latest = config.clone();
            }
        }
        if log.latest_config() != &latest {
            return Err(format!(
                "latest_config returned {:?}, expected {latest:?}",
                log.latest_config()
            )
            .into());
        }

        let target = sample_contained_position(ctx, log.entries());
        let (actual_position, actual_config) = log
            .get_position_and_config(target.index)
            .expect("a contained index must resolve");
        let mut expected_config = snapshot_config;
        for (position, entry) in log.entries().iter_with_positions() {
            if position.index > target.index {
                break;
            }
            if let LogEntry::ClusterConfig(config) = entry {
                expected_config = config.clone();
            }
        }
        if actual_position != target || actual_config != &expected_config {
            return Err(format!(
                "config query mismatch at {:?}: got ({actual_position:?}, {actual_config:?}), \
                 expected ({target:?}, {expected_config:?})",
                target.index
            )
            .into());
        }

        let beyond = log.last_position().index.next();
        if log.get_position_and_config(beyond).is_some() {
            return Err(format!("out-of-range index {beyond:?} resolved unexpectedly").into());
        }
        Ok(())
    })
}

#[test]
fn log_position_next_advances_index_only() -> noprop::TestResult {
    run(1024, |ctx| {
        let term = Term::new(noprop::sample_u64(ctx));
        let index = LogIndex::new(sample_u64_before_max(ctx));
        let position = LogPosition::new(term, index);
        let next = position.next();
        if next.index.get() != index.get() + 1 || next.term != term {
            return Err(format!("next({position:?}) returned {next:?}").into());
        }
        Ok(())
    })
}

#[test]
fn log_index_next_advances_by_one() -> noprop::TestResult {
    run(1024, |ctx| {
        let index = LogIndex::new(sample_u64_before_max(ctx));
        let next = index.next();
        if next.get() != index.get() + 1 {
            return Err(format!("next({index:?}) returned {next:?}").into());
        }
        Ok(())
    })
}
