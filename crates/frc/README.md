# frc

Computes the Fourier ring correlation (FRC) of microscopy images, from which their resolution follows. It takes two images of the same object, or one image of photon counts that it splits binomially in two (1FRC). The [repository's README](https://github.com/tiptenbrink/frc#readme) explains the method, when 1FRC applies, and which papers to cite.

To compute an FRC curve:

1. Make an [`FrcPlan`] for the size of your images and its [`Rings`]. A plan is for square images of n x n pixels, and you reuse it for every image of that size.
2. Call [`FrcPlan::one_frc`] with one image of photon counts, a [`Window`] and a seed, or [`FrcPlan::two_frc`] with two images that you have windowed. Pass `one_frc` the counts after offset and gain correction and before any other processing.
3. Read the curve. Element r is the FRC of ring r, for every r < n / 2. With the default [`Rings::Nearest`], ring r holds the frequencies f whose distance |f| from the origin rounds to r.
4. Pass the curve and the plan's ring counts to [`first_crossing`] with a [`Threshold`]. It interpolates where the curve first drops below the threshold, and its [`Crossing::resolution`] is the resolution in pixels.

```rust
use frc::{CpuFeatures, FrcPlan, Rings, Threshold, Window, binom_split, first_crossing};

let n = 64;
let mut plan = FrcPlan::new(n, Rings::default());

// One image of photon counts, in row-major order. The default window is a Tukey window.
let counts: Vec<i32> = (0..n * n).map(|i| 50 + (i % 7) as i32).collect();
let window = Window::default();
let curve = plan.one_frc(&counts, window, 7).unwrap();
assert_eq!(curve.len(), n / 2);
let crossing = first_crossing(&curve, plan.ring_counts(), Threshold::OneSeventh);
if let Some(crossing) = crossing {
    println!("resolution: {} pixels", crossing.resolution(n));
}

// Two images of the same scene: here the two halves of the same split, made and windowed by
// hand.
let mut samples = counts.clone();
binom_split(&mut samples, n, 7, CpuFeatures::Detected).unwrap();
let weights = window.weights(n);
let windowed = |half: &dyn Fn(usize) -> i32| -> Vec<f32> {
    (0..n * n).map(|i| half(i) as f32 * (weights[i / n] * weights[i % n])).collect()
};
let first = windowed(&|i| samples[i]);
let second = windowed(&|i| counts[i] - samples[i]);
assert_eq!(plan.two_frc(&first, &second), curve);
```

## The curve and its parts

The FRC of a ring is Re(Σ F1 conj F2) / sqrt(Σ |F1|² Σ |F2|²), where F1 and F2 are the Fourier transforms of the two images and the sums run over the frequencies of the ring. [`FrcPlan::one_frc_ring_sums`] and [`FrcPlan::two_frc_ring_sums`] return the three sums and the number of frequencies of each ring as [`RingSums`], and [`ring_counts`] returns only those numbers. [`Rings::Floor`] gives the rings of earlier versions of frc, the bins of DIPlib's `RadialSum`.

`one_frc` splits every count c into a sample k of binomial(c, 1/2) and c - k. It draws k as the number of ones among c random bits. It multiplies both halves by the window after the split, because a count multiplied by a window is no longer a Poisson count. [`binom_split`] does the same split on its own. With the same seed it draws the samples that `one_frc` draws, so `one_frc` of an image is `two_frc` of the two halves that `binom_split` makes of it, each multiplied by the window.

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
[`Window`]: https://docs.rs/frc/latest/frc/enum.Window.html
[`Rings`]: https://docs.rs/frc/latest/frc/enum.Rings.html
[`Rings::Nearest`]: https://docs.rs/frc/latest/frc/enum.Rings.html#variant.Nearest
[`Rings::Floor`]: https://docs.rs/frc/latest/frc/enum.Rings.html#variant.Floor
[`ring_counts`]: https://docs.rs/frc/latest/frc/fn.ring_counts.html
[`first_crossing`]: https://docs.rs/frc/latest/frc/fn.first_crossing.html
[`Threshold`]: https://docs.rs/frc/latest/frc/enum.Threshold.html
[`Crossing::resolution`]: https://docs.rs/frc/latest/frc/struct.Crossing.html#method.resolution
[`binom_split`]: https://docs.rs/frc/latest/frc/fn.binom_split.html
