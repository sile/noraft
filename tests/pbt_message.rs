//! Model-based PBTs for public message transport helpers.

pub mod pbt_harness;

use noraft::{LogEntries, LogIndex, LogPosition, Message, NodeId, Term};
use pbt_harness::{run, sample_len, sample_log_entry};

/// Stripping an AppendEntries prefix must preserve message metadata and
/// produce the exact unsent suffix for every boundary-sized request.
#[test]
fn append_entries_prefix_stripping_matches_model() -> noprop::TestResult {
    run(1024, |ctx| {
        let count = sample_len(ctx, 64);
        let prev_position = LogPosition {
            term: Term::new(noprop::sample_u64(ctx)),
            index: LogIndex::new(noprop::sample_usize_in(ctx, 0..=1024) as u64),
        };
        let entries =
            LogEntries::from_iter(prev_position, (0..count).map(|_| sample_log_entry(ctx)));
        let before: Vec<_> = entries.iter_with_positions().collect();
        let sent_count = noprop::sample_with_boundaries(
            ctx,
            &[0, count, usize::MAX],
            noprop::Ratio::one_nth(5),
            |ctx| noprop::sample_usize_in(ctx, 0..=count + 4),
        );
        let dropped = sent_count.min(count);
        let expected_prev = if dropped == 0 {
            entries.prev_position()
        } else {
            before[dropped - 1].0
        };
        let expected_entries = LogEntries::from_iter(
            expected_prev,
            before[dropped..].iter().map(|(_, entry)| entry.clone()),
        );

        let from = NodeId::new(noprop::sample_u64(ctx));
        let term = Term::new(noprop::sample_u64(ctx));
        let commit_index = LogIndex::new(noprop::sample_u64(ctx));
        let mut actual = Message::AppendEntriesCall {
            from,
            term,
            commit_index,
            entries,
        };
        if !actual.strip_append_entries_prefix(sent_count) {
            return Err("AppendEntriesCall was not recognized".into());
        }
        let expected = Message::AppendEntriesCall {
            from,
            term,
            commit_index,
            entries: expected_entries,
        };
        if actual != expected {
            return Err(format!(
                "strip_append_entries_prefix({sent_count}) mismatch: \
                 actual={actual:?}, expected={expected:?}"
            )
            .into());
        }

        let mut request_vote = Message::RequestVoteCall {
            from,
            term,
            last_position: expected_prev,
        };
        let unchanged = request_vote.clone();
        if request_vote.strip_append_entries_prefix(sent_count) || request_vote != unchanged {
            return Err("prefix stripping changed a non-AppendEntries message".into());
        }
        Ok(())
    })
}
