//! Model-based PBTs for public message transport helpers.

/// Stripping an AppendEntries prefix must preserve message metadata and
/// produce the exact unsent suffix for every boundary-sized request.
#[test]
fn append_entries_prefix_stripping_matches_model() -> noprop::TestResult {
    pbt::run(1024, |ctx| {
        let count = pbt::sample_len(ctx, 64);
        let prev_position = noraft::LogPosition {
            term: noraft::Term::new(noprop::sample_u64(ctx)),
            index: noraft::LogIndex::new(noprop::sample_usize_in(ctx, 0..=1024) as u64),
        };
        let entries = noraft::LogEntries::from_iter(
            prev_position,
            (0..count).map(|_| pbt::sample_log_entry(ctx)),
        );
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
        let expected_entries = noraft::LogEntries::from_iter(
            expected_prev,
            before[dropped..].iter().map(|(_, entry)| entry.clone()),
        );

        let from = noraft::NodeId::new(noprop::sample_u64(ctx));
        let term = noraft::Term::new(noprop::sample_u64(ctx));
        let commit_index = noraft::LogIndex::new(noprop::sample_u64(ctx));
        let mut actual = noraft::Message::AppendEntriesCall {
            from,
            term,
            commit_index,
            entries,
        };
        if !actual.strip_append_entries_prefix(sent_count) {
            return Err("AppendEntriesCall was not recognized".into());
        }
        let expected = noraft::Message::AppendEntriesCall {
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

        let mut request_vote = noraft::Message::RequestVoteCall {
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
