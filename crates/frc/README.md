# frc

Computes the Fourier ring correlation (FRC) of images, which microscopy uses to measure resolution: of two images of one scene, or of one image of photon counts that it splits in two.

To compute an FRC curve:

1. Make an [`FrcPlan`] for the size of your images. A plan is for square images of n x n pixels, and you reuse it for every image of that size.
2. Call [`FrcPlan::one_frc`] with one image of counts and a seed, or [`FrcPlan::two_frc`] with two images.
3. Read the curve. Element r is the FRC of ring r, which holds the frequencies f with r <= |f| < r + 1, for every r < n / 2.

```rust
use frc::{CpuFeatures, FrcPlan, binom_split};

let n = 64;
let mut plan = FrcPlan::new(n);

// One image of photon counts, in row-major order.
let counts: Vec<i32> = (0..n * n).map(|i| 50 + (i % 7) as i32).collect();
let curve = plan.one_frc(&counts, 7).unwrap();
assert_eq!(curve.len(), n / 2);

// Two images of the same scene: here the two halves of the same split, made by hand.
let mut samples = counts.clone();
binom_split(&mut samples, n, 7, CpuFeatures::Detected).unwrap();
let first: Vec<f32> = samples.iter().map(|&k| k as f32).collect();
let second: Vec<f32> = counts.iter().zip(&samples).map(|(c, k)| (c - k) as f32).collect();
assert_eq!(plan.two_frc(&first, &second), curve);
```

## The curve

The FRC of a ring is Re(Σ F1 conj F2) / sqrt(Σ |F1|² Σ |F2|²), where F1 and F2 are the Fourier transforms of the two images and the sums run over the frequencies of the ring. [`FrcPlan::one_frc_ring_sums`] and [`FrcPlan::two_frc_ring_sums`] return the three sums and the number of frequencies of each ring as [`RingSums`], and [`ring_counts`] returns only those numbers. The rings are the bins of DIPlib's `RadialSum`.

## One image of counts

`one_frc` splits every count c of the image into a sample k of binomial(c, 1/2) and c - k. A Poisson count with mean λ becomes two independent Poisson counts with mean λ / 2, so one image becomes two independent images of the same scene (Rieger, Droste, Gerritsma, ten Brink and Stallinga, [Optics Express 32(12), 2024](https://doi.org/10.1364/OE.524683)). The split is exact: k is the number of ones among c random bits.

[`binom_split`] does the same split on its own. It replaces every count of an image with its sample k. With the same seed it draws the samples that `one_frc` draws, so `one_frc` of an image is `two_frc` of the two halves that `binom_split` makes of it.

## Threads and repeatable results

The functions of this crate run on rayon's thread pool: the global pool, or the pool that your call runs in.

- [`binom_split`] gives the same samples for the same image and seed on every CPU and any number of threads.
- [`FrcPlan::one_frc`] and [`FrcPlan::two_frc`] give the same curve for the same input and seed on any number of threads. Between CPUs with different vector instructions, the Fourier transform rounds differently, and a curve differs in its last bits.

The split uses AVX-512 or AVX2 when the CPU has them, and chooses at run time.

## Python

The Python packages [frc-core](https://pypi.org/project/frc-core/) and [frc](https://pypi.org/project/frc/) are built from this crate, in the same [repository](https://github.com/tiptenbrink/frc). frc-core has no dependencies. frc takes NumPy arrays, and adds thresholds and the resolution.

[`FrcPlan`]: https://docs.rs/frc/latest/frc/struct.FrcPlan.html
[`FrcPlan::one_frc`]: https://docs.rs/frc/latest/frc/struct.FrcPlan.html#method.one_frc
[`FrcPlan::two_frc`]: https://docs.rs/frc/latest/frc/struct.FrcPlan.html#method.two_frc
[`FrcPlan::one_frc_ring_sums`]: https://docs.rs/frc/latest/frc/struct.FrcPlan.html#method.one_frc_ring_sums
[`FrcPlan::two_frc_ring_sums`]: https://docs.rs/frc/latest/frc/struct.FrcPlan.html#method.two_frc_ring_sums
[`RingSums`]: https://docs.rs/frc/latest/frc/struct.RingSums.html
[`ring_counts`]: https://docs.rs/frc/latest/frc/fn.ring_counts.html
[`binom_split`]: https://docs.rs/frc/latest/frc/fn.binom_split.html
