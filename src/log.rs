use crate::{Term, config::ClusterConfig};
use alloc::collections::BTreeMap;

/// In-memory representation of a [`Node`][crate::Node] local log.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Log {
    snapshot_config: ClusterConfig,
    entries: LogEntries,
}

impl Log {
    /// Makes a new [`Log`] instance with the given cluster configuration and entries.
    ///
    /// # Examples
    ///
    /// ```
    /// let empty_config = noraft::ClusterConfig::new();
    /// let mut single_config = noraft::ClusterConfig::new();
    /// single_config.voters.insert(noraft::NodeId::new(1));
    ///
    /// let entries = noraft::LogEntries::from_iter(
    ///     noraft::LogPosition::ZERO,
    ///     vec![
    ///         noraft::LogEntry::Term(noraft::Term::ZERO),
    ///         noraft::LogEntry::ClusterConfig(single_config.clone()),
    ///         noraft::LogEntry::Command,
    ///     ],
    /// );
    /// let log = noraft::Log::new(empty_config.clone(), entries);
    ///
    /// assert_eq!(log.snapshot_position(), noraft::LogPosition::ZERO);
    /// assert_eq!(log.snapshot_config(), &empty_config);
    /// assert_eq!(log.latest_config(), &single_config);
    /// ```
    pub const fn new(snapshot_config: ClusterConfig, entries: LogEntries) -> Self {
        Self {
            snapshot_config,
            entries,
        }
    }

    /// Returns a reference to the entries in this log.
    pub fn entries(&self) -> &LogEntries {
        &self.entries
    }

    pub(crate) fn entries_mut(&mut self) -> &mut LogEntries {
        &mut self.entries
    }

    /// Appends a suffix to this log.
    ///
    /// This method preserves the snapshot configuration and delegates to
    /// [`LogEntries::append_suffix()`].
    ///
    /// The caller is expected to pass a suffix whose `prev_position()` is
    /// contained in this log.
    /// Returns [`true`] if the suffix was appended.
    /// Returns [`false`] and leaves this log unchanged if `suffix.prev_position()`
    /// is not contained in this log.
    pub fn append_suffix(&mut self, suffix: &LogEntries) -> bool {
        self.entries.append_suffix(suffix)
    }

    /// Returns the position of the last entry in this log.
    ///
    /// This is equivalent to `self.entries().last_position()`.
    pub fn last_position(&self) -> LogPosition {
        self.entries.last_position()
    }

    /// Returns the log position where the snapshot was taken.
    ///
    /// This is equivalent to `self.entries().prev_position()`.
    pub fn snapshot_position(&self) -> LogPosition {
        self.entries.prev_position
    }

    /// Returns a reference to the cluster configuration at the time the snapshot was taken.
    pub fn snapshot_config(&self) -> &ClusterConfig {
        &self.snapshot_config
    }

    /// Returns a reference to the cluster configuration located at the highest index in this log.
    pub fn latest_config(&self) -> &ClusterConfig {
        self.entries
            .configs
            .last_key_value()
            .map(|(_, v)| v)
            .unwrap_or(&self.snapshot_config)
    }

    /// Returns the log position and a reference to the most recent cluster configuration at the given index.
    ///
    /// If the index is out of range, this method returns `None`.
    ///
    /// This method is useful when taking snapshots.
    pub fn get_position_and_config(
        &self,
        index: LogIndex,
    ) -> Option<(LogPosition, &ClusterConfig)> {
        self.entries().get_term(index).and_then(|term| {
            self.get_config(index)
                .map(|config| (LogPosition { term, index }, config))
        })
    }

    pub(crate) fn get_config(&self, index: LogIndex) -> Option<&ClusterConfig> {
        self.entries().contains_index(index).then(|| {
            self.entries
                .configs
                .range(..=index)
                .map(|x| x.1)
                .next_back()
                .unwrap_or(&self.snapshot_config)
        })
    }

    pub(crate) fn latest_config_index(&self) -> LogIndex {
        self.entries
            .configs
            .last_key_value()
            .map(|(i, _)| *i)
            .unwrap_or(self.entries.prev_position.index)
    }
}

/// Log entries.
///
/// This representation is compact and only requires `O(|terms|) + O(|configs|)` memory,
/// where `|terms|` is the number of [`LogEntry::Term`] entries and
/// `|configs|` is the number of [`LogEntry::ClusterConfig`] entries.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LogEntries {
    prev_position: LogPosition,
    last_position: LogPosition,
    terms: BTreeMap<LogIndex, Term>,
    configs: BTreeMap<LogIndex, ClusterConfig>,
}

impl LogEntries {
    /// Makes a new empty [`LogEntries`] instance at the given position.
    ///
    /// # Examples
    ///
    /// ```
    /// let entries = noraft::LogEntries::new(noraft::LogPosition::ZERO);
    /// assert!(entries.is_empty());
    /// assert_eq!(entries.len(), 0);
    /// assert_eq!(entries.iter().count(), 0);
    /// assert_eq!(entries.prev_position(), noraft::LogPosition::ZERO);
    /// assert_eq!(entries.last_position(), noraft::LogPosition::ZERO);
    /// ```
    pub const fn new(prev_position: LogPosition) -> Self {
        Self {
            prev_position,
            last_position: prev_position,
            terms: BTreeMap::new(),
            configs: BTreeMap::new(),
        }
    }

    /// Makes a new [`LogEntries`] instance with the given entries.
    ///
    /// # Examples
    ///
    /// ```
    /// let entries = noraft::LogEntries::from_iter(
    ///     noraft::LogPosition::ZERO,
    ///     vec![
    ///         noraft::LogEntry::Term(noraft::Term::ZERO),
    ///         noraft::LogEntry::Command,
    ///         noraft::LogEntry::Command,
    ///     ],
    /// );
    /// assert!(!entries.is_empty());
    /// assert_eq!(entries.len(), 3);
    /// assert_eq!(entries.iter().count(), 3);
    /// assert_eq!(entries.prev_position(), noraft::LogPosition::ZERO);
    /// assert_eq!(
    ///     entries.last_position(),
    ///     noraft::LogPosition {
    ///         term: noraft::Term::ZERO,
    ///         index: noraft::LogIndex::new(3),
    ///     }
    /// );
    /// ```
    pub fn from_iter<I>(prev_position: LogPosition, entries: I) -> Self
    where
        I: IntoIterator<Item = LogEntry>,
    {
        let mut this = Self::new(prev_position);
        this.extend(entries);
        this
    }

    /// Returns the number of entries in this [`LogEntries`] instance.
    pub fn len(&self) -> usize {
        let len = self.last_position.index.get() - self.prev_position.index.get();
        // Keeping more entries than `usize::MAX` is outside the supported
        // in-memory representation, even when the absolute log index is higher.
        usize::try_from(len).expect("log entries length exceeds usize::MAX")
    }

    /// Returns [`true`] if the log entries is empty (i.e., the previous and last positions are the same).
    pub fn is_empty(&self) -> bool {
        self.prev_position == self.last_position
    }

    /// Returns the position immediately before the first entry in this [`LogEntries`] instance.
    pub fn prev_position(&self) -> LogPosition {
        self.prev_position
    }

    /// Returns the position of the last entry in this [`LogEntries`] instance.
    pub fn last_position(&self) -> LogPosition {
        self.last_position
    }

    /// Returns an iterator over the entries in this [`LogEntries`] instance.
    ///
    /// # Overflow
    ///
    /// If `prev_position().index` equals `LogIndex::new(u64::MAX)`, computing
    /// the iteration range panics in debug builds and wraps in release
    /// builds. Guard untrusted values with [`LogIndex::checked_next`]
    /// beforehand.
    pub fn iter(&self) -> impl '_ + Iterator<Item = LogEntry> {
        (self.prev_position.index.get() + 1..=self.last_position.index.get()).map(|i| {
            let i = LogIndex::new(i);
            if let Some(term) = self.terms.get(&i).copied() {
                LogEntry::Term(term)
            } else if let Some(config) = self.configs.get(&i).cloned() {
                LogEntry::ClusterConfig(config)
            } else {
                LogEntry::Command
            }
        })
    }

    /// Returns an iterator over the entries in this [`LogEntries`] instance with their positions.
    ///
    /// # Examples
    /// ```
    /// let mut entries = noraft::LogEntries::new(noraft::LogPosition::ZERO);
    /// entries.push(noraft::LogEntry::Command);
    /// entries.push(noraft::LogEntry::Term(noraft::Term::new(1)));
    /// entries.push(noraft::LogEntry::Command);
    ///
    /// fn pos(term: u64, index: u64) -> noraft::LogPosition {
    ///     noraft::LogPosition {
    ///         term: noraft::Term::new(term),
    ///         index: noraft::LogIndex::new(index),
    ///     }
    /// }
    ///
    /// let mut iter = entries.iter_with_positions();
    /// assert_eq!(iter.next(), Some((pos(0, 1), noraft::LogEntry::Command)));
    /// assert_eq!(
    ///     iter.next(),
    ///     Some((pos(1, 2), noraft::LogEntry::Term(noraft::Term::new(1))))
    /// );
    /// assert_eq!(iter.next(), Some((pos(1, 3), noraft::LogEntry::Command)));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// # Overflow
    ///
    /// If `prev_position().index` equals `LogIndex::new(u64::MAX)`, computing
    /// the base index panics in debug builds and wraps in release builds.
    /// Guard untrusted values with [`LogIndex::checked_next`] beforehand.
    pub fn iter_with_positions(&self) -> impl '_ + Iterator<Item = (LogPosition, LogEntry)> {
        let base_index = self.prev_position.index.get() + 1;
        let mut term = self.prev_position.term;
        self.iter().enumerate().map(move |(i, entry)| {
            if let LogEntry::Term(t) = entry {
                term = t;
            }
            let index = LogIndex::new(base_index + i as u64);
            let position = LogPosition { term, index };
            (position, entry)
        })
    }

    /// Returns [`true`] if the given position is within the range of this entries.
    ///
    /// # Examples
    ///
    /// ```
    /// fn pos(term: u64, index: u64) -> noraft::LogPosition {
    ///     noraft::LogPosition {
    ///         term: noraft::Term::new(term),
    ///         index: noraft::LogIndex::new(index),
    ///     }
    /// }
    ///
    /// let entries = noraft::LogEntries::from_iter(
    ///     noraft::LogPosition::ZERO,
    ///     vec![
    ///         noraft::LogEntry::Term(noraft::Term::ZERO),
    ///         noraft::LogEntry::Command,
    ///         noraft::LogEntry::Term(noraft::Term::new(1)),
    ///         noraft::LogEntry::Command,
    ///     ],
    /// );
    /// assert!(entries.contains(pos(0, 0))); // Including the previous position
    /// assert!(entries.contains(pos(1, 4))); // Including the last position
    /// assert!(!entries.contains(pos(0, 4))); // Index is within the range but term is different
    /// assert!(!entries.contains(pos(1, 5))); // Index is out of range
    /// ```
    pub fn contains(&self, position: LogPosition) -> bool {
        Some(position.term) == self.get_term(position.index)
    }

    /// Returns [`true`] if the given index is within the range of this entries.
    ///
    /// Unlike [`LogEntries::contains()`], this method does not check the term of the given index.
    ///
    /// # Examples
    ///
    /// ```
    /// let entries = noraft::LogEntries::from_iter(
    ///     noraft::LogPosition::ZERO,
    ///     vec![
    ///         noraft::LogEntry::Term(noraft::Term::ZERO),
    ///         noraft::LogEntry::Command,
    ///         noraft::LogEntry::Term(noraft::Term::new(1)),
    ///         noraft::LogEntry::Command,
    ///     ],
    /// );
    /// assert!(entries.contains_index(noraft::LogIndex::ZERO)); // Including the previous index
    /// assert!(entries.contains_index(noraft::LogIndex::new(1)));
    /// assert!(entries.contains_index(noraft::LogIndex::new(4))); // Including the last index
    /// assert!(!entries.contains_index(noraft::LogIndex::new(5)));
    /// ```
    pub fn contains_index(&self, index: LogIndex) -> bool {
        (self.prev_position.index..=self.last_position.index).contains(&index)
    }

    /// Returns the term of the given index if it is within the range of this entries.
    pub fn get_term(&self, index: LogIndex) -> Option<Term> {
        self.contains_index(index).then(|| {
            self.terms
                .range(..=index)
                .next_back()
                .map(|(_, term)| *term)
                .unwrap_or(self.prev_position.term)
        })
    }

    /// Returns the entry at the given index if it is within the range of this entries.
    ///
    /// Note that if the index is equal to the previous index, this method returns [`None`].
    ///
    /// # Examples
    ///
    /// ```
    /// let entries = noraft::LogEntries::from_iter(
    ///     noraft::LogPosition::ZERO,
    ///     vec![
    ///         noraft::LogEntry::Term(noraft::Term::ZERO),
    ///         noraft::LogEntry::Command,
    ///         noraft::LogEntry::Term(noraft::Term::new(1)),
    ///     ],
    /// );
    /// assert_eq!(entries.get_entry(noraft::LogIndex::ZERO), None);
    /// assert_eq!(
    ///     entries.get_entry(noraft::LogIndex::new(1)),
    ///     Some(noraft::LogEntry::Term(noraft::Term::ZERO))
    /// );
    /// assert_eq!(
    ///     entries.get_entry(noraft::LogIndex::new(2)),
    ///     Some(noraft::LogEntry::Command)
    /// );
    /// assert_eq!(
    ///     entries.get_entry(noraft::LogIndex::new(3)),
    ///     Some(noraft::LogEntry::Term(noraft::Term::new(1)))
    /// );
    /// assert_eq!(entries.get_entry(noraft::LogIndex::new(4)), None);
    /// ```
    pub fn get_entry(&self, index: LogIndex) -> Option<LogEntry> {
        if !self.contains_index(index) || self.prev_position.index == index {
            None
        } else if let Some(term) = self.terms.get(&index).copied() {
            Some(LogEntry::Term(term))
        } else if let Some(config) = self.configs.get(&index).cloned() {
            Some(LogEntry::ClusterConfig(config))
        } else {
            Some(LogEntry::Command)
        }
    }

    /// Appends an entry to the back of this entries.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut entries = noraft::LogEntries::new(noraft::LogPosition::ZERO);
    /// entries.push(noraft::LogEntry::Term(noraft::Term::ZERO));
    /// entries.push(noraft::LogEntry::Command);
    ///
    /// assert_eq!(
    ///     entries.get_entry(noraft::LogIndex::new(1)),
    ///     Some(noraft::LogEntry::Term(noraft::Term::ZERO))
    /// );
    /// assert_eq!(
    ///     entries.last_position(),
    ///     noraft::LogPosition {
    ///         term: noraft::Term::ZERO,
    ///         index: noraft::LogIndex::new(2),
    ///     }
    /// );
    /// ```
    ///
    /// # Overflow
    ///
    /// If `last_position().index` equals `LogIndex::new(u64::MAX)`, advancing
    /// the position panics in debug builds and wraps in release builds.
    /// Guard untrusted values with [`LogIndex::checked_next`] beforehand.
    pub fn push(&mut self, entry: LogEntry) {
        self.last_position = self.last_position.next();
        match entry {
            LogEntry::Term(term) => {
                self.terms.insert(self.last_position.index, term);
                self.last_position.term = term;
            }
            LogEntry::ClusterConfig(config) => {
                self.configs
                    .insert(self.last_position.index, config.clone());
            }
            LogEntry::Command => {}
        }
    }

    /// Shortens the entries, keeping the first `len` entries and dropping the rest.
    /// If `len` is greater or equal to `LogEntries::len()`, this has no effect.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut entries = noraft::LogEntries::new(noraft::LogPosition::ZERO);
    /// entries.push(noraft::LogEntry::Term(noraft::Term::ZERO));
    /// entries.push(noraft::LogEntry::Command);
    /// entries.push(noraft::LogEntry::Term(noraft::Term::new(1)));
    /// assert_eq!(entries.len(), 3);
    ///
    /// // No effect.
    /// entries.truncate(3);
    /// assert_eq!(entries.len(), 3);
    ///
    /// // Drop the last two entries.
    /// entries.truncate(1);
    /// assert_eq!(entries.len(), 1);
    /// assert_eq!(
    ///     entries.get_entry(noraft::LogIndex::new(1)),
    ///     Some(noraft::LogEntry::Term(noraft::Term::ZERO))
    /// );
    /// assert_eq!(entries.get_entry(noraft::LogIndex::new(2)), None);
    ///
    /// // Drop all entries.
    /// entries.truncate(0);
    /// assert_eq!(entries.len(), 0);
    /// assert_eq!(entries.get_entry(noraft::LogIndex::new(1)), None);
    /// ```
    pub fn truncate(&mut self, len: usize) {
        if self.len() <= len {
            return;
        }

        let last_index = LogIndex::new(self.prev_position.index.get() + len as u64);
        let Some(last_term) = self.get_term(last_index) else {
            unreachable!();
        };
        self.last_position.term = last_term;
        self.last_position.index = last_index;
        self.terms.split_off(&last_index.next());
        self.configs.split_off(&last_index.next());
    }

    /// Returns the suffix after `new_prev_position`.
    ///
    /// The returned [`LogEntries`] has `new_prev_position` as its previous position
    /// and contains all entries after that position.
    ///
    /// This method returns [`None`] if `new_prev_position` is not contained in this
    /// [`LogEntries`] instance. The position must match both the index and the term.
    ///
    /// # Examples
    ///
    /// ```
    /// fn pos(term: u64, index: u64) -> noraft::LogPosition {
    ///     noraft::LogPosition {
    ///         term: noraft::Term::new(term),
    ///         index: noraft::LogIndex::new(index),
    ///     }
    /// }
    ///
    /// let entries = noraft::LogEntries::from_iter(
    ///     noraft::LogPosition::ZERO,
    ///     vec![
    ///         noraft::LogEntry::Term(noraft::Term::ZERO),
    ///         noraft::LogEntry::Command,
    ///         noraft::LogEntry::Term(noraft::Term::new(1)),
    ///         noraft::LogEntry::Command,
    ///     ],
    /// );
    ///
    /// let suffix = entries.since(pos(0, 2)).expect("position should match");
    /// assert_eq!(suffix.prev_position(), pos(0, 2));
    /// assert_eq!(
    ///     suffix.iter_with_positions().collect::<Vec<_>>(),
    ///     vec![
    ///         (pos(1, 3), noraft::LogEntry::Term(noraft::Term::new(1))),
    ///         (pos(1, 4), noraft::LogEntry::Command),
    ///     ],
    /// );
    ///
    /// assert_eq!(entries.since(pos(0, 3)), None); // Term mismatch.
    /// ```
    pub fn since(&self, new_prev_position: LogPosition) -> Option<Self> {
        if !self.contains(new_prev_position) {
            return None;
        }

        let next_index = new_prev_position.index.next();
        Some(Self {
            prev_position: new_prev_position,
            last_position: self.last_position,
            terms: self
                .terms
                .range(next_index..)
                .map(|(index, term)| (*index, *term))
                .collect(),
            configs: self
                .configs
                .range(next_index..)
                .map(|(index, config)| (*index, config.clone()))
                .collect(),
        })
    }

    /// Appends a suffix to these log entries.
    ///
    /// The `suffix.prev_position()` is used as the append anchor.
    /// The caller is expected to pass a suffix whose append anchor is contained
    /// in these log entries. A contained position must match both the index and
    /// the term.
    /// Local entries after the anchor are replaced by the suffix entries.
    ///
    /// Passing an empty suffix truncates local entries after the append anchor.
    /// This method is intended for applying a known suffix replacement, such as
    /// replaying persistent append records. It is not equivalent to handling an
    /// empty AppendEntries RPC heartbeat.
    ///
    /// Returns [`true`] if the suffix was appended.
    /// Returns [`false`] and leaves these log entries unchanged if
    /// `suffix.prev_position()` is not contained in these log entries.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut entries = noraft::LogEntries::from_iter(
    ///     noraft::LogPosition::ZERO,
    ///     [
    ///         noraft::LogEntry::Term(noraft::Term::ZERO),
    ///         noraft::LogEntry::Command,
    ///         noraft::LogEntry::Term(noraft::Term::new(1)),
    ///     ],
    /// );
    /// let suffix = noraft::LogEntries::from_iter(
    ///     noraft::LogPosition::new(noraft::Term::ZERO, noraft::LogIndex::new(2)),
    ///     [
    ///         noraft::LogEntry::Term(noraft::Term::new(2)),
    ///         noraft::LogEntry::Command,
    ///     ],
    /// );
    ///
    /// assert!(entries.append_suffix(&suffix));
    ///
    /// assert_eq!(entries.last_position(), suffix.last_position());
    /// assert_eq!(
    ///     entries.get_entry(noraft::LogIndex::new(3)),
    ///     Some(noraft::LogEntry::Term(noraft::Term::new(2))),
    /// );
    /// ```
    pub fn append_suffix(&mut self, suffix: &Self) -> bool {
        if !self.contains(suffix.prev_position) {
            return false;
        }
        self.append_unchecked(suffix);
        true
    }

    pub(crate) fn append(&mut self, entries: &Self) {
        self.append_unchecked(entries);
    }

    fn append_unchecked(&mut self, entries: &Self) {
        debug_assert!(self.contains(entries.prev_position));
        self.last_position = entries.prev_position;
        self.terms.split_off(&self.last_position.index.next());
        self.configs.split_off(&self.last_position.index.next());

        self.terms.extend(&entries.terms);
        self.configs
            .extend(entries.configs.iter().map(|(k, v)| (*k, v.clone())));
        self.last_position = entries.last_position;
    }

    pub(crate) fn strip_common_prefix(&self, local_entries: &Self) -> Self {
        debug_assert!(local_entries.contains(self.prev_position));
        debug_assert!(!local_entries.contains(self.last_position));

        if self.prev_position == local_entries.last_position {
            return self.clone();
        } else if self.contains(local_entries.last_position) {
            return self
                .since(local_entries.last_position)
                .expect("unreachable");
        }

        // This method only best-effort minimizes the delta to append.
        // It keeps the latest position that is proven to be common, and
        // `append()` will truncate the local suffix after that position.
        let mut last_common_position = self.prev_position;
        for (&index, &term) in &self.terms {
            let position = LogPosition { term, index };
            if local_entries.contains(position) {
                last_common_position = position;
                continue;
            }

            // The entry immediately before a mismatching Term boundary may
            // still be common, but that boundary can also be beyond the local
            // range. Only use the predecessor after verifying it locally.
            let candidate = LogPosition {
                term: last_common_position.term,
                index: LogIndex::new(index.get() - 1),
            };
            if local_entries.contains(candidate) {
                last_common_position = candidate;
            }
            return self
                .since(last_common_position)
                .expect("last_common_position is derived from this log");
        }

        // self.terms is empty
        //
        // [NOTE]
        //
        // This situation should never occur if nodes correctly follow the Raft algorithm.
        // When `self.terms` is empty and the preconditions are met:
        // - `local_entries.contains(self.prev_position)` ensures both logs have matching term at `self.prev_position`
        // - No `LogEntry::Term` entries in `self` means no leader changes occurred after `self.prev_position`
        // Thus, log divergences cannot happen under correct Raft behavior.
        //
        // However, if there is a bug in the implementation or invalid input from the user,
        // the remote log could still diverge despite these conditions.
        //
        // Potential improvements for robustness:
        // - Set a flag indicating 'this node entered an inconsistent state'
        // - Stop further message handling by this node
        // - Provide a mechanism to notify the user of the inconsistency for diagnostics or recovery actions

        self.clone()
    }

    pub(crate) fn handle_snapshot_installed(&mut self, last_included_position: LogPosition) {
        if last_included_position.index < self.prev_position().index {
            return;
        }

        // Rebase pending append entries to the installed snapshot. If the
        // snapshot position is incompatible with this pending suffix, it belongs
        // to a different log branch at the snapshot boundary and must not be
        // restored after the snapshot.
        *self = self
            .since(last_included_position)
            .unwrap_or_else(|| Self::new(last_included_position));
    }
}

impl core::iter::Extend<LogEntry> for LogEntries {
    fn extend<T: IntoIterator<Item = LogEntry>>(&mut self, iter: T) {
        for entry in iter {
            self.push(entry);
        }
    }
}

/// Log index.
///
/// According to the Raft paper, index 0 serves as a sentinel value,
/// and the actual log entries start from index 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LogIndex(u64);

impl LogIndex {
    /// The initial log index (sentinel value)
    pub const ZERO: Self = Self(0);

    /// Makes a new [`LogIndex`] instance.
    pub const fn new(i: u64) -> Self {
        Self(i)
    }

    /// Returns the inner of this index.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Returns the next log index.
    ///
    /// # Examples
    ///
    /// ```
    /// let index = noraft::LogIndex::new(3);
    /// assert_eq!(index.next(), noraft::LogIndex::new(4));
    /// ```
    pub const fn next(self) -> Self {
        Self(self.0 + 1)
    }

    /// Returns the next log index, or [`None`] if the increment would overflow.
    ///
    /// # Examples
    ///
    /// ```
    /// let index = noraft::LogIndex::new(3);
    /// assert_eq!(index.checked_next(), Some(noraft::LogIndex::new(4)));
    ///
    /// let last = noraft::LogIndex::new(u64::MAX);
    /// assert_eq!(last.checked_next(), None);
    /// ```
    pub const fn checked_next(self) -> Option<Self> {
        match self.0.checked_add(1) {
            Some(v) => Some(Self(v)),
            None => None,
        }
    }

    /// Checked addition. Computes `self + rhs`, returning [`None`] if overflow occurred.
    ///
    /// # Examples
    ///
    /// ```
    /// let index = noraft::LogIndex::new(3);
    /// assert_eq!(
    ///     index.checked_add(noraft::LogIndex::new(4)),
    ///     Some(noraft::LogIndex::new(7))
    /// );
    ///
    /// let last = noraft::LogIndex::new(u64::MAX);
    /// assert_eq!(last.checked_add(noraft::LogIndex::new(1)), None);
    /// ```
    pub const fn checked_add(self, rhs: Self) -> Option<Self> {
        match self.0.checked_add(rhs.0) {
            Some(v) => Some(Self(v)),
            None => None,
        }
    }

    /// Checked subtraction. Computes `self - rhs`, returning [`None`] if overflow occurred.
    ///
    /// # Examples
    ///
    /// ```
    /// let index = noraft::LogIndex::new(7);
    /// assert_eq!(
    ///     index.checked_sub(noraft::LogIndex::new(3)),
    ///     Some(noraft::LogIndex::new(4))
    /// );
    ///
    /// let zero = noraft::LogIndex::ZERO;
    /// assert_eq!(zero.checked_sub(noraft::LogIndex::new(1)), None);
    /// ```
    pub const fn checked_sub(self, rhs: Self) -> Option<Self> {
        match self.0.checked_sub(rhs.0) {
            Some(v) => Some(Self(v)),
            None => None,
        }
    }
}

impl From<u64> for LogIndex {
    fn from(value: u64) -> Self {
        Self::new(value)
    }
}

impl From<LogIndex> for u64 {
    fn from(value: LogIndex) -> Self {
        value.get()
    }
}

impl core::ops::Add for LogIndex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.0 + rhs.0)
    }
}

impl core::ops::AddAssign for LogIndex {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl core::ops::Sub for LogIndex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.0 - rhs.0)
    }
}

impl core::ops::SubAssign for LogIndex {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

/// Log position ([`Term`] and [`LogIndex`]).
///
/// A [`LogPosition`] uniquely identifies a [`LogEntry`] stored within a cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LogPosition {
    /// Term of the log entry.
    pub term: Term,

    /// Index of the log entry.
    pub index: LogIndex,
}

impl LogPosition {
    /// The initial log position ([`Term::ZERO`] and [`LogIndex::ZERO`]).
    pub const ZERO: Self = Self::new(Term::ZERO, LogIndex::ZERO);

    /// An invalid log position.
    pub const INVALID: Self = Self::new(Term::new(u64::MAX), LogIndex::ZERO);

    /// Makes a new [`LogPosition`] instance.
    ///
    /// # Examples
    ///
    /// ```
    /// let position = noraft::LogPosition::new(
    ///     noraft::Term::new(2),
    ///     noraft::LogIndex::new(7),
    /// );
    /// assert_eq!(position.term, noraft::Term::new(2));
    /// assert_eq!(position.index, noraft::LogIndex::new(7));
    /// ```
    pub const fn new(term: Term, index: LogIndex) -> Self {
        Self { term, index }
    }

    /// Returns the position at the next log index in the same term.
    ///
    /// # Examples
    ///
    /// ```
    /// let position = noraft::LogPosition::new(
    ///     noraft::Term::new(2),
    ///     noraft::LogIndex::new(7),
    /// );
    /// assert_eq!(
    ///     position.next(),
    ///     noraft::LogPosition::new(noraft::Term::new(2), noraft::LogIndex::new(8)),
    /// );
    /// ```
    pub const fn next(self) -> Self {
        Self::new(self.term, self.index.next())
    }

    /// Returns the position at the next log index in the same term,
    /// or [`None`] if the index would overflow.
    ///
    /// # Examples
    ///
    /// ```
    /// let position = noraft::LogPosition::new(
    ///     noraft::Term::new(2),
    ///     noraft::LogIndex::new(7),
    /// );
    /// assert_eq!(
    ///     position.checked_next(),
    ///     Some(noraft::LogPosition::new(noraft::Term::new(2), noraft::LogIndex::new(8))),
    /// );
    ///
    /// let last = noraft::LogPosition::new(
    ///     noraft::Term::new(2),
    ///     noraft::LogIndex::new(u64::MAX),
    /// );
    /// assert_eq!(last.checked_next(), None);
    /// ```
    pub const fn checked_next(self) -> Option<Self> {
        match self.index.checked_next() {
            Some(index) => Some(Self::new(self.term, index)),
            None => None,
        }
    }

    /// Returns `true` if this position is equal to [`LogPosition::INVALID`].
    pub const fn is_invalid(self) -> bool {
        matches!(self, Self::INVALID)
    }
}

/// Log entry.
///
/// Each log entry within a cluster is uniquely identified by a [`LogPosition`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LogEntry {
    /// A log entry to indicate the start of a new term with a new leader.
    Term(Term),

    /// A log entry for a new cluster configuration.
    ClusterConfig(ClusterConfig),

    /// A log entry for a user-defined command.
    ///
    /// # Note
    ///
    /// This crate does not handle the content of user-defined commands.
    /// Therefore, this variant is represented as a unit.
    /// It is the user's responsibility to manage the mapping from each [`LogEntry::Command`] to
    /// an actual command data.
    Command,
}

/// Commit status of a log entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CommitStatus {
    /// The log entry is currently being committed.
    InProgress,

    /// The log entry has been successfully committed.
    Committed,

    /// The log entry has been rejected.
    Rejected,

    /// The log entry does not exist, typically due to removal by snapshotting.
    ///
    /// It is unknown whether the entry was ever committed or rejected.
    Unknown,
}

impl CommitStatus {
    /// Returns `true` if the status is `InProgress`.
    pub const fn is_in_progress(self) -> bool {
        matches!(self, Self::InProgress)
    }

    /// Returns `true` if the status is `Committed`.
    pub const fn is_committed(self) -> bool {
        matches!(self, Self::Committed)
    }

    /// Returns `true` if the status is `Rejected`.
    pub const fn is_rejected(self) -> bool {
        matches!(self, Self::Rejected)
    }

    /// Returns `true` if the status is `Unknown`.
    pub const fn is_unknown(self) -> bool {
        matches!(self, Self::Unknown)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NodeId;
    use alloc::collections::BTreeSet;
    use alloc::{vec, vec::Vec};

    #[test]
    fn log_entries_append() {
        let mut entries = LogEntries::new(LogPosition::ZERO);
        assert_eq!(entries.last_position(), pos(0, 0));

        // Append entries to the last position.
        entries.append(&two_entries(
            pos(0, 0),
            LogEntry::Term(Term::ZERO),
            LogEntry::Command,
        ));
        assert_eq!(entries.last_position(), pos(0, 2));
        assert_eq!(entries.get_entry(i(0)), None);
        assert_eq!(entries.get_entry(i(1)), Some(LogEntry::Term(Term::ZERO)));
        assert_eq!(entries.get_entry(i(2)), Some(LogEntry::Command));

        // Append entries to the last position again.
        entries.append(&two_entries(
            pos(0, 2),
            LogEntry::Term(Term::new(1)),
            LogEntry::Command,
        ));
        assert_eq!(entries.last_position(), pos(1, 4));
        assert_eq!(entries.get_entry(i(0)), None);
        assert_eq!(entries.get_entry(i(1)), Some(LogEntry::Term(Term::ZERO)));
        assert_eq!(entries.get_entry(i(2)), Some(LogEntry::Command));
        assert_eq!(entries.get_entry(i(3)), Some(LogEntry::Term(Term::new(1))));
        assert_eq!(entries.get_entry(i(4)), Some(LogEntry::Command));

        // Truncate conflicting entries, then append.
        entries.append(&two_entries(
            pos(1, 3),
            LogEntry::Term(Term::new(2)),
            LogEntry::Command,
        ));
        assert_eq!(entries.last_position(), pos(2, 5));
        assert_eq!(entries.get_entry(i(0)), None);
        assert_eq!(entries.get_entry(i(1)), Some(LogEntry::Term(Term::ZERO)));
        assert_eq!(entries.get_entry(i(2)), Some(LogEntry::Command));
        assert_eq!(entries.get_entry(i(3)), Some(LogEntry::Term(Term::new(1))));
        assert_eq!(entries.get_entry(i(4)), Some(LogEntry::Term(Term::new(2))));
        assert_eq!(entries.get_entry(i(5)), Some(LogEntry::Command));

        // Truncate conflicting entries, then append again.
        entries.append(&two_entries(
            pos(0, 2),
            LogEntry::Term(Term::new(3)),
            LogEntry::Command,
        ));
        assert_eq!(entries.last_position(), pos(3, 4));
        assert_eq!(entries.get_entry(i(0)), None);
        assert_eq!(entries.get_entry(i(1)), Some(LogEntry::Term(Term::ZERO)));
        assert_eq!(entries.get_entry(i(2)), Some(LogEntry::Command));
        assert_eq!(entries.get_entry(i(3)), Some(LogEntry::Term(Term::new(3))));
        assert_eq!(entries.get_entry(i(4)), Some(LogEntry::Command));
    }

    #[test]
    fn log_entries_append_suffix_replaces_local_suffix() {
        let mut local_entries = entries(
            LogPosition::ZERO,
            &[
                LogEntry::Term(Term::ZERO),
                LogEntry::Command,
                LogEntry::Term(Term::new(1)),
            ],
        );
        let suffix = entries(
            pos(0, 2),
            &[LogEntry::Term(Term::new(2)), LogEntry::Command],
        );

        assert!(local_entries.append_suffix(&suffix));

        assert_eq!(local_entries.last_position(), pos(2, 4));
        assert_eq!(
            local_entries.iter_with_positions().collect::<Vec<_>>(),
            vec![
                (pos(0, 1), LogEntry::Term(Term::ZERO)),
                (pos(0, 2), LogEntry::Command),
                (pos(2, 3), LogEntry::Term(Term::new(2))),
                (pos(2, 4), LogEntry::Command),
            ]
        );
    }

    #[test]
    fn log_entries_append_suffix_truncates_with_empty_suffix() {
        let mut entries = entries(
            LogPosition::ZERO,
            &[
                LogEntry::Term(Term::ZERO),
                LogEntry::Command,
                LogEntry::Term(Term::new(1)),
            ],
        );
        let suffix = LogEntries::new(pos(0, 2));

        assert!(entries.append_suffix(&suffix));

        assert_eq!(entries.prev_position(), LogPosition::ZERO);
        assert_eq!(entries.last_position(), pos(0, 2));
        assert_eq!(entries.get_entry(i(3)), None);
    }

    #[test]
    fn log_entries_append_suffix_returns_false_for_missing_anchor() {
        let mut local_entries = entries(
            LogPosition::ZERO,
            &[
                LogEntry::Term(Term::ZERO),
                LogEntry::Command,
                LogEntry::Term(Term::new(1)),
            ],
        );
        let suffix = entries(pos(0, 3), &[LogEntry::Command]);
        let original = local_entries.clone();

        let appended = local_entries.append_suffix(&suffix);

        assert!(!appended);
        assert_eq!(suffix.prev_position(), pos(0, 3));
        assert_eq!(local_entries, original);
    }

    #[test]
    fn log_append_suffix_preserves_snapshot_config() {
        let mut snapshot_config = ClusterConfig::new();
        snapshot_config.voters.insert(NodeId::new(1));
        let mut log = Log::new(
            snapshot_config.clone(),
            entries(
                LogPosition::ZERO,
                &[
                    LogEntry::Term(Term::ZERO),
                    LogEntry::Command,
                    LogEntry::Term(Term::new(1)),
                ],
            ),
        );
        let suffix = entries(
            pos(0, 2),
            &[LogEntry::Term(Term::new(2)), LogEntry::Command],
        );

        assert!(log.append_suffix(&suffix));

        assert_eq!(log.snapshot_config(), &snapshot_config);
        assert_eq!(
            log.entries(),
            &entries(
                LogPosition::ZERO,
                &[
                    LogEntry::Term(Term::ZERO),
                    LogEntry::Command,
                    LogEntry::Term(Term::new(2)),
                    LogEntry::Command,
                ]
            )
        );
    }

    #[test]
    fn log_entries_truncate_with_large_len_has_no_effect() {
        let mut entries = entries(
            pos(1, 5),
            &[
                LogEntry::Command,
                LogEntry::Term(Term::new(2)),
                LogEntry::Command,
            ],
        );
        let original = entries.clone();

        entries.truncate(usize::MAX);

        assert_eq!(entries, original);
    }

    #[test]
    fn log_entries_truncate_after_high_index_snapshot() {
        let snapshot_index = u64::from(u32::MAX);
        let mut entries = entries(
            pos(1, snapshot_index),
            &[
                LogEntry::Command,
                LogEntry::Term(Term::new(2)),
                LogEntry::Command,
            ],
        );

        assert_eq!(entries.len(), 3);

        entries.truncate(1);

        assert_eq!(entries.prev_position(), pos(1, snapshot_index));
        assert_eq!(entries.last_position(), pos(1, snapshot_index + 1));
        assert_eq!(
            entries.iter_with_positions().collect::<Vec<_>>(),
            vec![(pos(1, snapshot_index + 1), LogEntry::Command)]
        );
    }

    #[test]
    fn log_entries_since() {
        let mut entries = LogEntries::new(LogPosition::ZERO);
        entries.push(LogEntry::Term(Term::ZERO));
        entries.push(LogEntry::Command);
        entries.push(LogEntry::Term(Term::new(1)));
        entries.push(LogEntry::Command);
        entries.push(LogEntry::Command);

        assert_eq!(entries.since(pos(0, 0)), Some(entries.clone()));

        assert_eq!(
            entries
                .since(pos(0, 2))
                .map(|e| e.iter_with_positions().collect::<Vec<_>>()),
            Some(vec![
                (pos(1, 3), LogEntry::Term(Term::new(1))),
                (pos(1, 4), LogEntry::Command),
                (pos(1, 5), LogEntry::Command)
            ])
        );

        assert_eq!(
            entries
                .since(pos(1, 3))
                .map(|e| e.iter_with_positions().collect::<Vec<_>>()),
            Some(vec![
                (pos(1, 4), LogEntry::Command),
                (pos(1, 5), LogEntry::Command)
            ])
        );

        let suffix = entries
            .since(pos(1, 5))
            .expect("last position should be a valid suffix boundary");
        assert!(suffix.is_empty());
        assert_eq!(suffix.prev_position(), pos(1, 5));
        assert_eq!(suffix.last_position(), pos(1, 5));

        assert_eq!(entries.since(pos(0, 3)), None); // Term mismatch
    }

    #[test]
    fn log_entries_handle_snapshot_installed_keeps_suffix_after_snapshot_position() {
        let mut entries = entries(
            pos(1, 2),
            &[
                LogEntry::Command,
                LogEntry::Command,
                LogEntry::Term(Term::new(2)),
                LogEntry::Command,
            ],
        );

        entries.handle_snapshot_installed(pos(1, 4));

        assert_eq!(entries.prev_position(), pos(1, 4));
        assert_eq!(entries.last_position(), pos(2, 6));
        assert_eq!(
            entries.iter_with_positions().collect::<Vec<_>>(),
            vec![
                (pos(2, 5), LogEntry::Term(Term::new(2))),
                (pos(2, 6), LogEntry::Command),
            ]
        );
    }

    #[test]
    fn log_entries_handle_snapshot_installed_discards_incompatible_suffix() {
        let mut entries = entries(pos(1, 2), &[LogEntry::Command, LogEntry::Command]);

        entries.handle_snapshot_installed(pos(2, 3));

        assert!(entries.is_empty());
        assert_eq!(entries.prev_position(), pos(2, 3));
        assert_eq!(entries.last_position(), pos(2, 3));
    }

    #[test]
    fn log_entries_strip_common_prefix() {
        let local_entries = entries(
            LogPosition::ZERO,
            &[
                LogEntry::Term(Term::ZERO),
                LogEntry::Command,
                LogEntry::Term(Term::new(1)),
                LogEntry::Command,
                LogEntry::Command,
            ],
        );
        assert_eq!(local_entries.last_position, pos(1, 5));

        // remove.prev == local.last
        let remote_entries = entries(pos(1, 5), &[LogEntry::Command]);
        assert_eq!(
            remote_entries
                .strip_common_prefix(&local_entries)
                .prev_position,
            pos(1, 5)
        );

        // No divergence
        let remote_entries = entries(pos(1, 4), &[LogEntry::Command, LogEntry::Command]);
        assert_eq!(
            remote_entries
                .strip_common_prefix(&local_entries)
                .prev_position,
            pos(1, 5)
        );

        // Divergence
        let remote_entries = entries(
            pos(1, 4),
            &[
                LogEntry::Term(Term::new(2)),
                LogEntry::Command,
                LogEntry::Term(Term::new(3)),
            ],
        );
        assert_eq!(
            remote_entries
                .strip_common_prefix(&local_entries)
                .prev_position,
            pos(1, 4)
        );

        let remote_entries = entries(
            pos(1, 3),
            &[
                LogEntry::Term(Term::new(1)),
                LogEntry::Term(Term::new(2)),
                LogEntry::Command,
            ],
        );
        assert_eq!(
            remote_entries
                .strip_common_prefix(&local_entries)
                .prev_position,
            pos(1, 4)
        );
    }

    #[test]
    fn log_entries_strip_common_prefix_falls_back_when_term_is_outside_local_range() {
        let local_entries = entries(
            LogPosition::ZERO,
            &[
                LogEntry::Term(Term::new(1)),
                LogEntry::Command,
                LogEntry::Command,
                LogEntry::Term(Term::new(3)),
            ],
        );
        let remote_entries = entries(
            pos(1, 3),
            &[
                LogEntry::Command,
                LogEntry::Command,
                LogEntry::Term(Term::new(2)),
            ],
        );

        assert!(local_entries.contains(remote_entries.prev_position()));
        assert!(!local_entries.contains(remote_entries.last_position()));
        assert_eq!(
            remote_entries.strip_common_prefix(&local_entries),
            remote_entries
        );
    }

    #[test]
    fn log_index_next() {
        assert_eq!(i(7).next(), i(8));
    }

    #[test]
    fn log_position_new_and_next() {
        let position = LogPosition::new(Term::new(3), i(7));

        assert_eq!(position.term, Term::new(3));
        assert_eq!(position.index, i(7));
        assert_eq!(position.next(), pos(3, 8));
    }

    #[test]
    fn log_position_cmp() {
        assert_eq!(pos(5, 5).cmp(&pos(5, 5)), core::cmp::Ordering::Equal);
        assert_eq!(pos(7, 3).cmp(&pos(5, 5)), core::cmp::Ordering::Greater);
        assert_eq!(pos(3, 7).cmp(&pos(5, 5)), core::cmp::Ordering::Less);
        assert_eq!(pos(5, 7).cmp(&pos(5, 5)), core::cmp::Ordering::Greater);
        assert_eq!(pos(5, 3).cmp(&pos(5, 5)), core::cmp::Ordering::Less);
    }

    #[test]
    fn test_strip_common_prefix_with_config_entry_no_terms() {
        // Remote entries: only a ClusterConfig at index 1, no Term entries
        let remote_entries = {
            let mut entries = LogEntries::new(LogPosition::ZERO);
            let config = ClusterConfig {
                voters: {
                    let mut voters = BTreeSet::new();
                    voters.insert(NodeId::new(0));
                    voters
                },
                new_voters: {
                    let mut new_voters = BTreeSet::new();
                    new_voters.insert(NodeId::new(0));
                    new_voters.insert(NodeId::new(1));
                    new_voters
                },
                non_voters: BTreeSet::new(),
            };
            entries.push(LogEntry::ClusterConfig(config));
            entries
        };

        // Local entries: Term(1) at index 1, then Command entries
        let local_entries = entries(
            LogPosition::ZERO,
            &[
                LogEntry::Term(Term::new(1)),
                LogEntry::Command,
                LogEntry::Command,
                LogEntry::ClusterConfig(ClusterConfig {
                    voters: {
                        let mut voters = BTreeSet::new();
                        voters.insert(NodeId::new(0));
                        voters
                    },
                    new_voters: {
                        let mut new_voters = BTreeSet::new();
                        new_voters.insert(NodeId::new(0));
                        new_voters.insert(NodeId::new(1));
                        new_voters
                    },
                    non_voters: BTreeSet::new(),
                }),
            ],
        );

        // This should not panic
        let result = remote_entries.strip_common_prefix(&local_entries);
        assert_eq!(result.prev_position(), LogPosition::ZERO);
    }

    #[test]
    fn log_index_checked_next() {
        assert_eq!(LogIndex::ZERO.checked_next(), Some(i(1)));
        assert_eq!(i(3).checked_next(), Some(i(4)));
        assert_eq!(i(u64::MAX - 1).checked_next(), Some(i(u64::MAX)));
        assert_eq!(i(u64::MAX).checked_next(), None);
    }

    #[test]
    fn log_index_checked_add() {
        assert_eq!(i(3).checked_add(i(4)), Some(i(7)));
        assert_eq!(i(u64::MAX).checked_add(LogIndex::ZERO), Some(i(u64::MAX)));
        assert_eq!(LogIndex::ZERO.checked_add(i(u64::MAX)), Some(i(u64::MAX)));
        assert_eq!(i(u64::MAX).checked_add(i(1)), None);
        assert_eq!(i(u64::MAX / 2 + 1).checked_add(i(u64::MAX / 2 + 1)), None);
    }

    #[test]
    fn log_index_checked_sub() {
        assert_eq!(i(7).checked_sub(i(3)), Some(i(4)));
        assert_eq!(i(1).checked_sub(i(1)), Some(LogIndex::ZERO));
        assert_eq!(LogIndex::ZERO.checked_sub(i(1)), None);
        assert_eq!(i(u64::MAX - 1).checked_sub(i(u64::MAX)), None);
    }

    #[test]
    fn log_position_checked_next() {
        assert_eq!(pos(2, 7).checked_next(), Some(pos(2, 8)));
        assert_eq!(pos(2, u64::MAX - 1).checked_next(), Some(pos(2, u64::MAX)));
        assert_eq!(pos(2, u64::MAX).checked_next(), None);
    }

    fn two_entries(prev_position: LogPosition, entry0: LogEntry, entry1: LogEntry) -> LogEntries {
        let mut entries = LogEntries::new(prev_position);
        entries.push(entry0);
        entries.push(entry1);
        entries
    }

    fn entries(prev_position: LogPosition, entries: &[LogEntry]) -> LogEntries {
        LogEntries::from_iter(prev_position, entries.iter().cloned())
    }

    fn i(index: u64) -> LogIndex {
        LogIndex::new(index)
    }

    fn pos(term: u64, index: u64) -> LogPosition {
        LogPosition::new(Term::new(term), LogIndex::new(index))
    }
}
