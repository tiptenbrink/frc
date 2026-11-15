// The README links its items to docs.rs. These definitions come first, so rustdoc links the same
// names to the items themselves and warns when one of them no longer exists.
//! [`FrcPlan`]: FrcPlan
//! [`FrcPlan::one_frc`]: FrcPlan::one_frc
//! [`FrcPlan::two_frc`]: FrcPlan::two_frc
//! [`FrcPlan::one_frc_ring_sums`]: FrcPlan::one_frc_ring_sums
//! [`FrcPlan::two_frc_ring_sums`]: FrcPlan::two_frc_ring_sums
//! [`RingSums`]: RingSums
//! [`ring_counts`]: ring_counts
//! [`binom_split`]: binom_split
#![doc = include_str!("../README.md")]
#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::missing_errors_doc, clippy::missing_panics_doc)]

mod binomial;
mod cpu;
mod frc;
mod platform;

pub use binomial::{NegativeCount, binom_split};
pub use cpu::CpuFeatures;
pub use frc::{FrcPlan, RingSums, ring_counts};
