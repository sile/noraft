use crate::{
    Term,
    log::{LogEntries, LogIndex, LogPosition},
    node::{NodeGeneration, NodeId},
};

/// Message for RPC.
///
/// Note that this enum does not include the InstallSnapshot RPC,
/// as the specifics of snapshot installation depend heavily on
/// each individual application and are not managed by this crate.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Message {
    /// RequestVote RPC call.
    RequestVoteCall {
        /// Sender of the message.
        from: NodeId,

        /// Term of the sender.
        term: Term,

        /// Last log position of the candidate.
        last_position: LogPosition,
    },

    /// RequestVote RPC reply.
    RequestVoteReply {
        /// Sender of the message.
        from: NodeId,

        /// Term of the sender.
        term: Term,

        /// Whether the vote is granted or not.
        vote_granted: bool,
    },

    /// AppendEntries RPC call.
    AppendEntriesCall {
        /// Sender of the message.
        from: NodeId,

        /// Term of the sender.
        term: Term,

        /// Leader's commit index.
        commit_index: LogIndex,

        /// Entries to append.
        ///
        /// Note that if the entries are too large to fit in a single message,
        /// they can be shrunk by calling [`LogEntries::truncate()`] before sending.
        /// Also, after sending a part of entries, the sent prefix can be dropped by
        /// calling [`Message::strip_append_entries_prefix()`].
        entries: LogEntries,
    },

    /// AppendEntries RPC reply.
    AppendEntriesReply {
        /// Sender of the message.
        from: NodeId,

        /// Term of the sender.
        term: Term,

        /// Generation of the sender.
        generation: NodeGeneration,

        /// Last log position of the follower.
        ///
        /// Instead of replying a boolean `success` as defined in the Raft paper,
        /// this crate replies the last log position of the follower.
        /// With this adjustment, the leader can quickly determine the appropriate match index of the follower,
        /// even if the follower is significantly behind the leader.
        last_position: LogPosition,
    },
}

impl Message {
    /// Returns the sender node ID of the message.
    pub fn from(&self) -> NodeId {
        match self {
            Self::RequestVoteCall { from, .. } => *from,
            Self::RequestVoteReply { from, .. } => *from,
            Self::AppendEntriesCall { from, .. } => *from,
            Self::AppendEntriesReply { from, .. } => *from,
        }
    }

    /// Returns the term of the message.
    pub fn term(&self) -> Term {
        match self {
            Self::RequestVoteCall { term, .. } => *term,
            Self::RequestVoteReply { term, .. } => *term,
            Self::AppendEntriesCall { term, .. } => *term,
            Self::AppendEntriesReply { term, .. } => *term,
        }
    }

    /// Drops the first `sent_count` entries if this is an [`Message::AppendEntriesCall`].
    ///
    /// Returns [`true`] when this is an [`Message::AppendEntriesCall`], otherwise returns [`false`]
    /// and does not modify the message.
    ///
    /// If `sent_count` is `0`, this method has no effect.
    /// If `sent_count` is greater than or equal to the number of entries, all entries are dropped.
    pub fn strip_append_entries_prefix(&mut self, sent_count: usize) -> bool {
        let Self::AppendEntriesCall { entries, .. } = self else {
            return false;
        };

        if sent_count == 0 {
            return true;
        }
        if sent_count >= entries.len() {
            *entries = LogEntries::new(entries.last_position());
            return true;
        }

        let new_prev_index = LogIndex::new(entries.prev_position().index.get() + sent_count as u64);
        let new_prev_term = entries.get_term(new_prev_index).expect(
            "sent_count is less than the number of entries, so the new previous index is always valid",
        );
        let new_prev_position = LogPosition {
            term: new_prev_term,
            index: new_prev_index,
        };
        *entries = entries.since(new_prev_position).expect(
            "new previous position is derived from the current entries, so it is always valid",
        );
        true
    }

    pub(crate) fn request_vote_call(term: Term, from: NodeId, last_position: LogPosition) -> Self {
        Self::RequestVoteCall {
            from,
            term,
            last_position,
        }
    }

    pub(crate) fn request_vote_reply(term: Term, from: NodeId, vote_granted: bool) -> Self {
        Self::RequestVoteReply {
            from,
            term,
            vote_granted,
        }
    }

    pub(crate) fn append_entries_call(
        term: Term,
        from: NodeId,
        commit_index: LogIndex,
        entries: LogEntries,
    ) -> Self {
        Self::AppendEntriesCall {
            from,
            term,
            commit_index,
            entries,
        }
    }

    pub(crate) fn append_entries_reply(
        term: Term,
        from: NodeId,
        generation: NodeGeneration,
        last_position: LogPosition,
    ) -> Self {
        Self::AppendEntriesReply {
            from,
            term,
            generation,
            last_position,
        }
    }

    pub(crate) fn merge(&mut self, other: Self) {
        debug_assert_eq!(self.from(), other.from());
        debug_assert!(self.term() <= other.term());

        let Self::AppendEntriesCall {
            from: from0,
            term: term0,
            commit_index: leader_commit0,
            entries: entries0,
        } = self
        else {
            *self = other;
            return;
        };
        let Self::AppendEntriesCall {
            from: from1,
            term: term1,
            commit_index: leader_commit1,
            entries: entries1,
        } = other
        else {
            *self = other;
            return;
        };

        *from0 = from1;
        *term0 = term1;
        *leader_commit0 = leader_commit1;
        if entries0.contains(entries1.prev_position()) {
            entries0.append(&entries1);
        } else {
            *entries0 = entries1;
        }
    }

    pub(crate) fn handle_snapshot_installed(&mut self, last_included_position: LogPosition) {
        match self {
            Message::RequestVoteCall {
                term,
                last_position,
                ..
            } => {
                *term = (*term).max(last_included_position.term);
                if last_position.index < last_included_position.index {
                    *last_position = last_included_position;
                }
            }
            Message::RequestVoteReply { term, .. } => {
                *term = (*term).max(last_included_position.term);
            }
            Message::AppendEntriesCall { term, entries, .. } => {
                *term = (*term).max(last_included_position.term);
                entries.handle_snapshot_installed(last_included_position);
            }
            Message::AppendEntriesReply {
                term,
                generation: _,
                last_position,
                ..
            } => {
                *term = (*term).max(last_included_position.term);
                if last_position.index < last_included_position.index {
                    *last_position = last_included_position;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{LogEntry, Term};

    use super::*;

    #[test]
    fn strip_append_entries_prefix_for_append_entries_call() {
        let from = NodeId::new(1);
        let term = Term::new(3);
        let commit_index = LogIndex::new(5);
        let prev = LogPosition {
            term: Term::new(2),
            index: LogIndex::new(10),
        };
        let entries = LogEntries::from_iter(
            prev,
            [
                LogEntry::Command,
                LogEntry::Term(Term::new(3)),
                LogEntry::Command,
            ],
        );
        let mut message = Message::append_entries_call(term, from, commit_index, entries);

        assert!(message.strip_append_entries_prefix(1));
        let Message::AppendEntriesCall { entries, .. } = &message else {
            panic!();
        };
        assert_eq!(entries.len(), 2);
        assert_eq!(
            entries.prev_position(),
            LogPosition {
                term: Term::new(2),
                index: LogIndex::new(11),
            }
        );
        assert_eq!(
            entries.last_position(),
            LogPosition {
                term: Term::new(3),
                index: LogIndex::new(13),
            }
        );
        assert_eq!(
            entries.iter().collect::<Vec<_>>(),
            vec![LogEntry::Term(Term::new(3)), LogEntry::Command]
        );
    }

    #[test]
    fn strip_append_entries_prefix_with_zero_len() {
        let from = NodeId::new(1);
        let term = Term::new(3);
        let commit_index = LogIndex::new(5);
        let prev = LogPosition {
            term: Term::new(2),
            index: LogIndex::new(10),
        };
        let original_entries = LogEntries::from_iter(prev, [LogEntry::Command, LogEntry::Command]);
        let mut message =
            Message::append_entries_call(term, from, commit_index, original_entries.clone());

        assert!(message.strip_append_entries_prefix(0));
        let Message::AppendEntriesCall {
            entries: current_entries,
            ..
        } = &message
        else {
            panic!();
        };
        assert_eq!(current_entries, &original_entries);
    }

    #[test]
    fn strip_append_entries_prefix_drops_all_entries() {
        let from = NodeId::new(1);
        let term = Term::new(3);
        let commit_index = LogIndex::new(5);
        let prev = LogPosition {
            term: Term::new(2),
            index: LogIndex::new(10),
        };
        let entries = LogEntries::from_iter(
            prev,
            [
                LogEntry::Command,
                LogEntry::Term(Term::new(3)),
                LogEntry::Command,
            ],
        );
        let old_last_position = entries.last_position();
        let mut message = Message::append_entries_call(term, from, commit_index, entries);

        assert!(message.strip_append_entries_prefix(3));
        let Message::AppendEntriesCall { entries, .. } = &message else {
            panic!();
        };
        assert!(entries.is_empty());
        assert_eq!(entries.prev_position(), old_last_position);
        assert_eq!(entries.last_position(), old_last_position);

        assert!(message.strip_append_entries_prefix(usize::MAX));
        let Message::AppendEntriesCall { entries, .. } = &message else {
            panic!();
        };
        assert!(entries.is_empty());
        assert_eq!(entries.prev_position(), old_last_position);
        assert_eq!(entries.last_position(), old_last_position);
    }

    #[test]
    fn strip_append_entries_prefix_non_append_entries_call_is_noop() {
        let from = NodeId::new(1);
        let term = Term::new(3);
        let mut request_vote = Message::request_vote_call(
            term,
            from,
            LogPosition {
                term,
                index: LogIndex::new(5),
            },
        );
        let original = request_vote.clone();

        assert!(!request_vote.strip_append_entries_prefix(1));
        assert_eq!(request_vote, original);
    }
}
