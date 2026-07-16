noraft
========

[![noraft](https://img.shields.io/crates/v/noraft.svg)](https://crates.io/crates/noraft)
[![Documentation](https://docs.rs/noraft/badge.svg)](https://docs.rs/noraft)
[![Actions Status](https://github.com/sile/noraft/workflows/CI/badge.svg)](https://github.com/sile/noraft/actions)
![License](https://img.shields.io/crates/l/noraft)


noraft: Minimal, feature-complete no_std Raft for Rust - no I/O, no dependencies.

`noraft` is a minimal but feature-complete, `no_std`, sans I/O implementation of the [Raft] distributed consensus algorithm.

[Raft]: https://raft.github.io/

[`Node`] is the main struct that represents a Raft node.
It offers methods for creating a cluster, proposing commands, updating cluster configurations,
handling incoming messages, snapshotting, and more.

[`Node`] itself does not execute I/O operations.
Instead, it generates [`Action`]s that represent pending I/O operations.
How to execute these actions is up to the crate user.

`noraft` keeps its API surface minimal and lets users choose and integrate their own I/O layer freely.

Except for a few optimizations, `noraft` is a very straightforward (yet efficient) implementation of the Raft algorithm.
This crate focuses on the core part of the algorithm.
So, offering various convenience features (which are not described in the Raft paper) is left to the crate user.

[`Node`]: https://docs.rs/noraft/latest/noraft/struct.Node.html
[`Action`]: https://docs.rs/noraft/latest/noraft/struct.Action.html

The following example outlines a basic usage flow of this crate:
```rust
// Start a node.
let mut node = noraft::Node::start(noraft::NodeId::new(0));

// Create a three nodes cluster.
let commit_position = node.create_cluster(&[
    noraft::NodeId::new(0),
    noraft::NodeId::new(1),
    noraft::NodeId::new(2),
]);

// Execute actions requested by the node until the cluster creation is complete.
while node.get_commit_status(commit_position).is_in_progress() {
    for action in node.actions_mut() {
        // How to execute actions is up to the crate user.
        match action {
           noraft::Action::SetElectionTimeout => { /* ... */ },
           noraft::Action::SaveCurrentTerm => { /* ... */ },
           noraft::Action::SaveVotedFor => { /* ... */ },
           noraft::Action::BroadcastMessage(_) => { /* ... */ },
           noraft::Action::AppendLogEntries(_) => { /* ... */ },
           noraft::Action::SendMessage(_, _) => { /* ... */ },
           noraft::Action::InstallSnapshot(_) => { /* ... */ },
        }
    }

    // If the election timeout is expired, handle it.
    if is_election_timeout_expired() {
        node.handle_election_timeout();
    }

    // If a message is received, handle it.
    while let Some(message) = try_receive_message() {
        node.handle_message(&message).expect("message handling should succeed");
    }
}

// Propose a user-defined command.
let commit_position = node.propose_command();

// Execute actions as before.
```
