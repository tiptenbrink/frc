// The README links its items to docs.rs. These definitions come first, so rustdoc links the same
// names to the items themselves and warns when one of them no longer exists.
//! [`FrcPlan`]: FrcPlan
//! [`FrcPlan::one_image`]: FrcPlan::one_image
//! [`FrcPlan::two_images`]: FrcPlan::two_images
//! [`FrcPlan::one_image_ring_sums`]: FrcPlan::one_image_ring_sums
//! [`FrcPlan::two_images_ring_sums`]: FrcPlan::two_images_ring_sums
//! [`RingSums`]: RingSums
//! [`Rings`]: Rings
//! [`Rings::Nearest`]: Rings::Nearest
//! [`Rings::Floor`]: Rings::Floor
//! [`Threshold`]: Threshold
//! [`first_crossing`]: first_crossing
//! [`Crossing::resolution`]: Crossing::resolution
//! [`Window`]: Window
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
mod rings;
mod threshold;
mod window;

pub use binomial::{NegativeCount, binom_split};
pub use cpu::CpuFeatures;
pub use frc::{FrcPlan, RingSums, ring_counts};
pub use rings::Rings;
pub use threshold::{Crossing, Threshold, first_crossing};
pub use window::Window;
