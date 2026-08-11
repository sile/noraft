//! Shared runner configuration and generators for the noprop-driven
//! property tests.

use noraft::{ClusterConfig, LogEntry, NodeId, Term};
use std::io::{Error, ErrorKind};

pub const SEED_ENV: &str = "NORAFT_PBT_SEED";
pub const CASES_ENV: &str = "NORAFT_PBT_CASES";

#[derive(Debug, Clone, Copy)]
pub struct RunConfig {
    pub seed: u64,
    pub cases: usize,
}

/// Loads the reproducibility seed and a strictly positive case budget.
///
/// An unset case-budget variable selects `default_cases`. A malformed
/// value or zero is an error so a misspelled override cannot silently
/// fall back or turn a property into a zero-case success.
pub fn run_config(default_cases: usize) -> noprop::TestResult<RunConfig> {
    assert!(
        default_cases > 0,
        "the default case budget must be positive"
    );

    let seed = noprop::seed_from_env_or_time(SEED_ENV)?;
    let cases = match std::env::var(CASES_ENV) {
        Ok(value) => value.parse::<usize>().map_err(|error| {
            Error::new(
                ErrorKind::InvalidInput,
                format!("invalid {CASES_ENV} value {value:?}: {error}"),
            )
        })?,
        Err(std::env::VarError::NotPresent) => default_cases,
        Err(error) => return Err(error.into()),
    };
    if cases == 0 {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!("{CASES_ENV} must be greater than zero"),
        )
        .into());
    }

    Ok(RunConfig { seed, cases })
}

/// Runs a property using the standard noraft PBT environment variables.
pub fn run<F>(default_cases: usize, property: F) -> noprop::TestResult
where
    F: Fn(&mut noprop::TestCaseContext) -> noprop::TestResult,
{
    let config = run_config(default_cases)?;
    noprop::Runner::new(config.seed).run(config.cases, property)?;
    Ok(())
}

/// Samples a bounded length while giving empty, singleton, and maximum
/// lengths explicit probability.
pub fn sample_len(ctx: &mut noprop::TestCaseContext, max: usize) -> usize {
    assert!(max >= 3, "sample_len requires max >= 3");
    noprop::sample_with_boundaries(ctx, &[0, 1, max], noprop::Ratio::one_nth(5), |ctx| {
        noprop::sample_usize_in(ctx, 2..max)
    })
}

/// Samples every representable `u64` except `u64::MAX`, with extra
/// weight on values relevant to `next()` boundary behavior.
pub fn sample_u64_before_max(ctx: &mut noprop::TestCaseContext) -> u64 {
    noprop::sample_with_boundaries(
        ctx,
        &[0, 1, u64::MAX - 1],
        noprop::Ratio::one_nth(5),
        |ctx| {
            noprop::sample_with_rejection(ctx, 8, |ctx| {
                let value = noprop::sample_u64(ctx);
                (value < u64::MAX).then_some(value)
            })
        },
    )
}

/// Samples an arbitrary public `ClusterConfig`, including overlaps
/// between any of the three node sets.
pub fn sample_config(ctx: &mut noprop::TestCaseContext) -> ClusterConfig {
    let mut config = ClusterConfig::new();
    for value in 0..6 {
        let id = NodeId::new(value);
        let membership = noprop::sample_usize_in(ctx, 0..8);
        if membership & 1 != 0 {
            config.voters.insert(id);
        }
        if membership & 2 != 0 {
            config.new_voters.insert(id);
        }
        if membership & 4 != 0 {
            config.non_voters.insert(id);
        }
    }
    config
}

/// Samples a non-joint configuration whose voters and non-voters are
/// disjoint by construction.
pub fn sample_normal_config(ctx: &mut noprop::TestCaseContext) -> ClusterConfig {
    let mut config = ClusterConfig::new();
    for value in 0..6 {
        let id = NodeId::new(value);
        match noprop::sample_usize_in(ctx, 0..3) {
            0 => {
                config.voters.insert(id);
            }
            1 => {
                config.non_voters.insert(id);
            }
            _ => {}
        }
    }
    config
}

pub fn sample_log_entry(ctx: &mut noprop::TestCaseContext) -> LogEntry {
    match noprop::sample_weighted_index(ctx, &[3, 1, 1]) {
        0 => LogEntry::Command,
        1 => LogEntry::Term(Term::new(noprop::sample_u64(ctx))),
        _ => LogEntry::ClusterConfig(sample_config(ctx)),
    }
}
