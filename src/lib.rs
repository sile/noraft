//! noraft: Minimal, feature-complete no_std Raft for Rust - no I/O, no dependencies.
//!
//! [Raft]: https://raft.github.io/
//!
//! [`Node`] is the main struct that represents a Raft node.
//! It offers methods for creating a cluster, proposing commands, updating cluster configurations,
//! handling incoming messages, snapshotting, and more.
//!
//! [`Node`] itself does not execute I/O operations.
//! Instead, it generates [`Action`]s that represent pending I/O operations.
//! How to execute these actions is up to the crate user.
//!
//! `noraft` keeps its API surface minimal and lets users choose and integrate their own I/O layer freely.
//!
//! Except for a few optimizations, `noraft` is a very straightforward (yet efficient) implementation of the Raft algorithm.
//! This crate focuses on the core part of the algorithm.
//! So, offering various convenience features (which are not described in the Raft paper) is left to the crate user.
//!
//! The following example outlines a basic usage flow of this crate:
//! ```
//! // Start a node.
//! let mut node = noraft::Node::start(noraft::NodeId::new(0));
//!
//! // Create a three nodes cluster.
//! let commit_position = node.create_cluster(&[
//!     noraft::NodeId::new(0),
//!     noraft::NodeId::new(1),
//!     noraft::NodeId::new(2),
//! ]);
//!
//! // Execute actions requested by the node until the cluster creation is complete.
//! while node.get_commit_status(commit_position).is_in_progress() {
//!     for action in node.actions_mut() {
//!         // How to execute actions is up to the crate user.
//!         match action {
//!            noraft::Action::SetElectionTimeout => { /* ... */ },
//!            noraft::Action::SaveCurrentTerm => { /* ... */ },
//!            noraft::Action::SaveVotedFor => { /* ... */ },
//!            noraft::Action::BroadcastMessage(_) => { /* ... */ },
//!            noraft::Action::AppendLogEntries(_) => { /* ... */ },
//!            noraft::Action::SendMessage(_, _) => { /* ... */ },
//!            noraft::Action::InstallSnapshot(_) => { /* ... */ },
//!         }
//!     }
//!
//!     // If the election timeout is expired, handle it.
//!     if is_election_timeout_expired() {
//!         node.handle_election_timeout();
//!     }
//!
//!     // If a message is received, handle it.
//!     while let Some(message) = try_receive_message() {
//!         node.handle_message(&message).expect("message handling should succeed");
//!     }
//!     # break;
//! }
//!
//! // Propose a user-defined command.
//! let commit_position = node.propose_command();
//!
//! // Execute actions as before.
//!
//! # fn is_election_timeout_expired() -> bool { true }
//! # fn try_receive_message() -> Option<noraft::Message> { None }
//! ```
#![no_std]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

extern crate alloc;

#[cfg(test)]
extern crate std;

mod action;
mod config;
mod log;
mod message;
mod node;
mod quorum;
mod role;

pub use action::{Action, Actions};
pub use config::ClusterConfig;
pub use log::{CommitStatus, Log, LogEntries, LogEntry, LogIndex, LogPosition};
pub use message::Message;
pub use node::{Error, Node, NodeGeneration, NodeId};
pub use role::Role;

/// Term.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Term(u64);

impl Term {
    /// The initial term.
    pub const ZERO: Self = Self(0);

    /// Makes a new [`Term`] instance.
    pub const fn new(t: u64) -> Self {
        Self(t)
    }

    /// Returns the value of this term.
    pub const fn get(self) -> u64 {
        self.0
    }

    pub(crate) const fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

impl From<u64> for Term {
    fn from(value: u64) -> Self {
        Self::new(value)
    }
}

impl From<Term> for u64 {
    fn from(value: Term) -> Self {
        value.get()
    }
}

impl core::ops::Add for Term {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.0 + rhs.0)
    }
}

impl core::ops::AddAssign for Term {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl core::ops::Sub for Term {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.0 - rhs.0)
    }
}

impl core::ops::SubAssign for Term {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}
