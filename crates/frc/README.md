# frc

Fourier ring correlation (FRC) of images, for resolution measurements in microscopy: from two images, or from one image of photon counts split in two by an exact binomial split.

- `binom_split` replaces every count n with a sample k of binomial(n, 1/2). A Poisson count with mean λ becomes one with mean λ / 2, and so does n - k, independently of k: one image becomes two independent images of the same scene (Rieger, Droste, Gerritsma, ten Brink and Stallinga, [Optics Express 32(12), 2024](https://doi.org/10.1364/OE.524683)).
- `FrcPlan` computes FRC curves of n x n images: `one_frc` of one image of counts, split as `binom_split` splits it, and `two_frc` of two images. Ring r holds the frequencies f with r <= |f| < r + 1, as DIPlib's `RadialSum` bins them.

Both run in parallel with rayon, and the split uses AVX-512 or AVX2 when the CPU has them, chosen at run time. With the same input and seed, the result is the same to the last bit on every CPU and any number of threads.

```rust
use frc::FrcPlan;

let n = 256;
let counts: Vec<i32> = vec![100; n * n];
let mut plan = FrcPlan::new(n);
let curve = plan.one_frc(&counts, 1).unwrap();
assert_eq!(curve.len(), n / 2);
```

The Python packages [frc-core](https://pypi.org/project/frc-core/), without dependencies, and [frc](https://pypi.org/project/frc/), with a NumPy API, thresholds and resolution, are built from this crate in the same [repository](https://github.com/tiptenbrink/frc). Its [docs/PERFORMANCE.md](https://github.com/tiptenbrink/frc/blob/main/docs/PERFORMANCE.md) describes where the time goes and the literature behind the choices.
