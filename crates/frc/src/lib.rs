// The README links its items to docs.rs. These definitions come first, so rustdoc links the same
// names to the items themselves and warns when one of them no longer exists.
//! [`FrcPlan`]: FrcPlan
//! [`FrcPlan::one_image`]: FrcPlan::one_image
//! [`FrcPlan::two_images`]: FrcPlan::two_images
//! [`FrcPlan::one_image_ring_sums`]: FrcPlan::one_image_ring_sums
//! [`FrcPlan::two_images_ring_sums`]: FrcPlan::two_images_ring_sums
//! [`FrcPlan::first_crossing`]: FrcPlan::first_crossing
//! [`RingSums`]: RingSums
//! [`Rings`]: Rings
//! [`Rings::first_crossing`]: Rings::first_crossing
//! [`RingBinning`]: RingBinning
//! [`RingBinning::Nearest`]: RingBinning::Nearest
//! [`RingBinning::Floor`]: RingBinning::Floor
//! [`Threshold`]: Threshold
//! [`Crossing`]: Crossing
//! [`Window`]: Window
//! [`binom_split`]: binom_split
#![doc = include_str!("../README.md")]
#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::missing_errors_doc, clippy::missing_panics_doc)]

mod binomial;
mod cpu;
mod frc;
mod platform;
mod rings;
mod threshold;
mod window;

pub use binomial::{NegativeCount, binom_split};
pub use cpu::CpuFeatures;
pub use frc::{FrcPlan, RingSums};
pub use rings::{RingBinning, Rings};
pub use threshold::{Crossing, Threshold};
pub use window::Window;
