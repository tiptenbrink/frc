//! An exact binomial split of images of counts, and Fourier ring correlation (FRC) from one image
//! or two.
//!
//! [`binom_split`] replaces every count n with a sample k of binomial(n, 1/2). A Poisson count
//! with mean λ becomes one with mean λ / 2, and so does n - k, independently of k: one image of
//! photon counts becomes two independent images of the same scene.
//!
//! [`FrcPlan`] computes FRC curves of n x n images: [`FrcPlan::one_frc`] of one image of counts,
//! split in two as `binom_split` splits it, and [`FrcPlan::two_frc`] of two images. The FRC at ring
//! r is Re(Σ F1 conj F2) / sqrt(Σ |F1|² Σ |F2|²) over the frequencies f with r <= |f| < r + 1,
//! where F1 and F2 are the spectra of the two images, for every r < n / 2.
//!
//! Both run in parallel on rayon's thread pool: the global one, or the one a call runs in. With
//! the same input and seed they give the same result, to the last bit, on every CPU and any
//! number of threads.
//!
//! ```
//! use frc::{FrcPlan, binom_split};
//!
//! let n = 64;
//! let counts: Vec<i32> = (0..n * n).map(|i| 50 + (i % 7) as i32).collect();
//!
//! // One image of counts in two halves.
//! let mut k = counts.clone();
//! binom_split(&mut k, 7, frc::CpuFeatures::Detected).unwrap();
//! let rest: Vec<i32> = counts.iter().zip(&k).map(|(n, k)| n - k).collect();
//!
//! // A plan is made once for a size and reused for every image of that size.
//! let mut plan = FrcPlan::new(n);
//! let curve = plan.one_frc(&counts, 7).unwrap();
//! assert_eq!(curve.len(), n / 2);
//!
//! // The same split by hand gives the same curve.
//! let as_f32 = |v: &[i32]| v.iter().map(|&c| c as f32).collect::<Vec<_>>();
//! assert_eq!(plan.two_frc(&as_f32(&k), &as_f32(&rest)), curve);
//! ```
//!
//! The Python package `frc-core` wraps this crate, and `frc` builds a NumPy API on that.

#![deny(unsafe_code)]

mod binomial;
mod frc;
mod platform;

pub use binomial::{CpuFeatures, NegativeCount, binom_split};
pub use frc::{FrcPlan, RingSums, ring_counts};
