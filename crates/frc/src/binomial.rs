//! An exact binomial split: every count n becomes a sample of binomial (n, 0.5), the number of
//! ones among n random bits.
//!
//! Sixteen xoshiro256++ generators run side by side, one count per lane, so that every step is
//! vector code. On x86-64 the code is chosen at run time: for AVX-512, for AVX2, where it is
//! written by hand, or for the baseline. Every choice draws the same samples.
//!
//! An image is split in chunks of whole rows. Every chunk has generators of its own, seeded from
//! the call's seed and the chunk's index. The samples therefore depend on the seed and the
//! length of a row alone, and not on how rayon schedules the chunks.

use std::fmt::{Display, Formatter};

use rayon::prelude::*;

use crate::cpu::{CpuFeatures, InstructionSet};

/// Counts that a chunk of a split holds at most, unless one row holds more. A chunk is what one
/// task samples, and what one task of the FRC's row pass transforms: 16 rows of a 4096 x 4096
/// image, so that each such task writes 16 rows into every panel of the spectrum.
const CHUNK_COUNTS: usize = 65536;

/// Returns the number of rows of an image, each of `row_len` counts, that one chunk of a split
/// holds.
///
/// # Panics
///
/// Panics if `row_len` is 0.
pub(crate) fn rows_per_chunk(row_len: usize) -> usize {
    (CHUNK_COUNTS / row_len).max(1)
}

/// xoshiro256++ generators that step together. 16 lanes give the CPU 16 independent chains of
/// instructions to work on at once. With 8 lanes the split was slower.
const LANES: usize = 16;

/// The error of a split of counts of which one is negative.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NegativeCount;

impl Display for NegativeCount {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("count is negative")
    }
}

impl std::error::Error for NegativeCount {}

/// Steps a SplitMix64 generator and returns its output. Its outputs seed the xoshiro256++
/// generators, as the authors of xoshiro256++ recommend.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Returns the generators of chunk `index` of a split with `seed`.
fn chunk_generator(seed: u64, index: usize) -> XoshiroLanes {
    let mut chunk_seed = seed ^ (index as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    XoshiroLanes::seed_from_u64(splitmix64(&mut chunk_seed))
}

/// `L` xoshiro256++ generators. Each of the four words of their state is stored for all lanes
/// together, so that one step of all generators is vector code.
struct XoshiroLanes<const L: usize = LANES> {
    s: [[u64; L]; 4],
}

impl<const L: usize> XoshiroLanes<L> {
    fn seed_from_u64(seed: u64) -> Self {
        let mut state = seed;
        let mut s = [[0; L]; 4];
        for word in s.iter_mut().flatten() {
            *word = splitmix64(&mut state);
        }
        Self { s }
    }

    /// Steps every lane once and writes the lanes' outputs to `out`.
    #[inline(always)]
    fn next_block(&mut self, out: &mut [u64; L]) {
        let [s0, s1, s2, s3] = &mut self.s;
        for i in 0..L {
            out[i] = s0[i]
                .wrapping_add(s3[i])
                .rotate_left(23)
                .wrapping_add(s0[i]);
            let t = s1[i] << 17;
            s2[i] ^= s0[i];
            s3[i] ^= s1[i];
            s1[i] ^= s2[i];
            s0[i] ^= s3[i];
            s2[i] ^= t;
            s3[i] = s3[i].rotate_left(45);
        }
    }
}

/// Replaces the counts of each group, which must not be negative, with samples of binomial
/// (n, 0.5). Lane i takes the bits of count i from generator lane i, 64 at a time, until the
/// largest count of the group is covered. Every step is the same for all lanes, so it compiles to
/// vector code.
///
/// It is inlined into a function of its own for each set of CPU features. A closure between the
/// two would be compiled for the baseline alone.
#[inline(always)]
fn sample_groups_portable(rng: &mut XoshiroLanes, groups: &mut [[i32; LANES]]) {
    for group in groups {
        let largest = group.iter().copied().max().unwrap_or(0) as u32;
        let mut ones = [0u64; LANES];
        let mut words = [0u64; LANES];
        let mut covered = 0u32;
        while covered < largest {
            rng.next_block(&mut words);
            for i in 0..LANES {
                let left = (group[i] as u32).saturating_sub(covered);
                let mask = if left >= 64 {
                    u64::MAX
                } else {
                    (1u64 << left) - 1
                };
                ones[i] += (words[i] & mask).count_ones() as u64;
            }
            covered += 64;
        }
        *group = ones.map(|k| k as i32);
    }
}

/// `sample_groups_portable` compiled for AVX-512, where `rotate_left` becomes `vprolq` and
/// `count_ones` becomes `vpopcntq` (AVX512_VPOPCNTDQ: Ice Lake and later, Zen 4 and later).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512vpopcntdq,avx2,popcnt")]
fn sample_groups_avx512(rng: &mut XoshiroLanes, groups: &mut [[i32; LANES]]) {
    sample_groups_portable(rng, groups)
}

/// Replaces the counts of each group with samples, with the code for `set`. A call of code
/// compiled for CPU features is unsafe from code compiled without them, and this function holds
/// the crate's only unsafe code outside `platform`.
#[allow(unsafe_code)]
fn sample_groups(set: InstructionSet, rng: &mut XoshiroLanes, groups: &mut [[i32; LANES]]) {
    match set {
        InstructionSet::Baseline => sample_groups_portable(rng, groups),
        // SAFETY: `CpuFeatures::instruction_set` gives a set only if the CPU has it.
        #[cfg(target_arch = "x86_64")]
        InstructionSet::Avx2 => unsafe { avx2::sample_groups(rng, groups) },
        #[cfg(target_arch = "x86_64")]
        InstructionSet::Avx512 => unsafe { sample_groups_avx512(rng, groups) },
    }
}

/// Replaces every count of `chunk`, which must not be negative, with a sample of binomial
/// (n, 0.5), `LANES` counts at a time. The last group is padded with zeros.
fn sample_chunk(set: InstructionSet, rng: &mut XoshiroLanes, chunk: &mut [i32]) {
    let (groups, tail) = chunk.as_chunks_mut::<LANES>();
    sample_groups(set, rng, groups);
    if !tail.is_empty() {
        let mut group = [0; LANES];
        group[..tail.len()].copy_from_slice(tail);
        sample_groups(set, rng, std::slice::from_mut(&mut group));
        tail.copy_from_slice(&group[..tail.len()]);
    }
}

/// `sample_groups_portable` by hand for AVX2. LLVM compiles the portable code poorly for AVX2:
/// the state of 16 lanes fills all 16 vector registers, so it spills on every step, and each
/// word's popcount ends in its own `vpsadbw` and repacking. Here half the lanes at a time go
/// through a block of groups, so that their state takes 8 of the registers.
///
/// On a Ryzen 5 3400G (Zen+), which splits every 256-bit instruction in two and shifts on one
/// unit, the split takes two thirds of the time it took with the portable code. On an Intel Core
/// Ultra it takes as long. Four lanes at a time, with no state spilled, were no faster on a Core
/// Ultra 7 155H: the three ports that take its vector instructions limit the sampler there.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    use bytemuck::cast;

    use super::{LANES, XoshiroLanes};

    /// Lanes that one pass takes: two vectors of four, whose state fits in 8 registers.
    const HALF: usize = 8;

    /// Groups of `LANES` counts that the passes over the lanes go through together.
    const BLOCK_GROUPS: usize = 256;

    /// Popcounts of words (at most 8 per byte each) that a byte can add up before `vpsadbw`.
    const BYTE_SUM_WORDS: u32 = 31;

    /// Replaces the counts of each group with samples, as `super::sample_groups_portable` does:
    /// lane i of a group takes words from generator lane i until the group's largest count is
    /// covered.
    #[target_feature(enable = "avx2")]
    pub(super) fn sample_groups(rng: &mut XoshiroLanes, groups: &mut [[i32; LANES]]) {
        for block in groups.chunks_mut(BLOCK_GROUPS) {
            let mut words = [0; BLOCK_GROUPS];
            for (words, group) in words.iter_mut().zip(block.iter()) {
                *words = (group.iter().copied().max().unwrap_or(0) as u32).div_ceil(64);
            }
            for first in (0..LANES).step_by(HALF) {
                sample_half(rng, block, &words, first);
            }
        }
    }

    /// `[f(0), f(1)]`. Compiled for AVX2, it can inline a closure that uses AVX2, which
    /// `std::array::from_fn`, compiled without it, cannot.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn pair(f: impl Fn(usize) -> __m256i) -> [__m256i; 2] {
        [f(0), f(1)]
    }

    #[target_feature(enable = "avx2")]
    fn rotate_left<const L: i32, const R: i32>(x: __m256i) -> __m256i {
        _mm256_or_si256(_mm256_slli_epi64::<L>(x), _mm256_srli_epi64::<R>(x))
    }

    /// The popcount of each byte of `x`, by a table lookup of each half byte.
    #[target_feature(enable = "avx2")]
    fn byte_popcounts(x: __m256i) -> __m256i {
        let nibble_ones = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2,
            3, 3, 4,
        );
        let low_nibbles = _mm256_set1_epi8(0x0f);
        let low = _mm256_and_si256(x, low_nibbles);
        let high = _mm256_and_si256(_mm256_srli_epi16::<4>(x), low_nibbles);
        _mm256_add_epi8(
            _mm256_shuffle_epi8(nibble_ones, low),
            _mm256_shuffle_epi8(nibble_ones, high),
        )
    }

    /// For each 64-bit lane, all ones shifted right by `shift`: all of them for 0 or less, the low
    /// 64 - shift for 1 to 63, none from 64.
    #[target_feature(enable = "avx2")]
    fn ones_shifted_right(shift: __m256i) -> __m256i {
        _mm256_or_si256(
            _mm256_cmpgt_epi64(_mm256_set1_epi64x(1), shift),
            _mm256_srlv_epi64(_mm256_set1_epi64x(-1), shift),
        )
    }

    /// The first four counts of `counts`, which are not negative, as 64-bit lanes.
    #[target_feature(enable = "avx2")]
    fn widened(counts: &[i32]) -> __m256i {
        let four: [i32; 4] = counts[..4].try_into().unwrap();
        _mm256_cvtepu32_epi64(cast::<[i32; 4], __m128i>(four))
    }

    /// The first four words of `words` as a vector.
    fn vector_of(words: &[u64]) -> __m256i {
        cast::<[u64; 4], __m256i>(words[..4].try_into().unwrap())
    }

    /// Samples lanes `first` .. `first + HALF` of every group.
    #[target_feature(enable = "avx2")]
    fn sample_half(
        rng: &mut XoshiroLanes,
        groups: &mut [[i32; LANES]],
        words: &[u32; BLOCK_GROUPS],
        first: usize,
    ) {
        let (zero, word) = (_mm256_setzero_si256(), _mm256_set1_epi64x(64));
        let at = |v: usize| first + 4 * v;
        let mut s: [[__m256i; 2]; 4] =
            [0, 1, 2, 3].map(|w| [vector_of(&rng.s[w][at(0)..]), vector_of(&rng.s[w][at(1)..])]);
        for (group, &words) in groups.iter_mut().zip(words) {
            // shift[v]: 64 - (bits of the lane's count left before this word).
            let mut shift: [__m256i; 2] =
                pair(|v| _mm256_sub_epi64(word, widened(&group[at(v)..])));
            let (mut byte_sums, mut sums) = ([zero; 2], [zero; 2]);
            for j in 0..words {
                for v in 0..2 {
                    let [s0, s1, s2, s3] = [s[0][v], s[1][v], s[2][v], s[3][v]];
                    let out = _mm256_add_epi64(rotate_left::<23, 41>(_mm256_add_epi64(s0, s3)), s0);
                    let s2 = _mm256_xor_si256(s2, s0);
                    let s3 = _mm256_xor_si256(s3, s1);
                    let s1 = _mm256_xor_si256(s1, s2);
                    let s0 = _mm256_xor_si256(s0, s3);
                    let s2 = _mm256_xor_si256(s2, _mm256_slli_epi64::<17>(s[1][v]));
                    let s3 = rotate_left::<45, 19>(s3);
                    [s[0][v], s[1][v], s[2][v], s[3][v]] = [s0, s1, s2, s3];

                    let bits = _mm256_and_si256(out, ones_shifted_right(shift[v]));
                    shift[v] = _mm256_add_epi64(shift[v], word);
                    byte_sums[v] = _mm256_add_epi8(byte_sums[v], byte_popcounts(bits));
                    if j % BYTE_SUM_WORDS == BYTE_SUM_WORDS - 1 {
                        sums[v] = _mm256_add_epi64(sums[v], _mm256_sad_epu8(byte_sums[v], zero));
                        byte_sums[v] = zero;
                    }
                }
            }
            for v in 0..2 {
                sums[v] = _mm256_add_epi64(sums[v], _mm256_sad_epu8(byte_sums[v], zero));
            }
            // The low halves of the 64-bit sums, which are below 2^31, as eight i32s in order.
            let interleaved = _mm256_or_si256(sums[0], _mm256_slli_epi64::<32>(sums[1]));
            let k =
                _mm256_permutevar8x32_epi32(interleaved, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
            group[first..first + HALF].copy_from_slice(&cast::<__m256i, [i32; 8]>(k));
        }
        for (w, s) in s.iter().enumerate() {
            for (v, &s) in s.iter().enumerate() {
                rng.s[w][at(v)..at(v) + 4].copy_from_slice(&cast::<__m256i, [u64; 4]>(s));
            }
        }
    }
}

/// Replaces every count c of an image with a sample of binomial(c, 1/2), in parallel.
///
/// `counts` holds the image in row-major order, in rows of `row_len` counts. The sample of a
/// count is the number of ones among c random bits.
///
/// The same counts, row length and seed give the same samples on every CPU, whatever `cpu`
/// allows, and on any number of threads. For a square image they are the samples that
/// [`FrcPlan::one_frc`](crate::FrcPlan::one_frc) draws with the same seed.
///
/// # Errors
///
/// Returns [`NegativeCount`] if a count is negative. Some counts are then replaced and others
/// are not.
///
/// # Panics
///
/// Panics if `row_len` is 0.
pub fn binom_split(
    counts: &mut [i32],
    row_len: usize,
    seed: u64,
    cpu: CpuFeatures,
) -> Result<(), NegativeCount> {
    let set = cpu.instruction_set();
    counts
        .par_chunks_mut(rows_per_chunk(row_len) * row_len)
        .enumerate()
        .try_for_each(|(index, chunk)| {
            if chunk.iter().any(|&n| n < 0) {
                return Err(NegativeCount);
            }
            sample_chunk(set, &mut chunk_generator(seed, index), chunk);
            Ok(())
        })
}

/// Counts that [`Splitter::split_chunk`] samples at a time, in a buffer on the stack.
const SPLIT_BLOCK: usize = 4096;

/// [`binom_split`] one chunk at a time, for a caller that does other work on a chunk while its
/// memory is in the cache, such as the FFT of its rows. Chunk `index` gets the generators that
/// `binom_split` gives its chunk `index`, so with chunks of [`rows_per_chunk`] rows both draw
/// the same samples.
pub(crate) struct Splitter {
    seed: u64,
    set: InstructionSet,
}

impl Splitter {
    pub(crate) fn new(seed: u64, cpu: CpuFeatures) -> Self {
        Self {
            seed,
            set: cpu.instruction_set(),
        }
    }

    /// Draws a sample k of binomial (n, 0.5) for every count n of `counts`, which is chunk
    /// `index` of an image, and writes `value_of(n, k)` to the same position in `out`.
    ///
    /// # Errors
    ///
    /// Returns [`NegativeCount`] if a count is negative.
    ///
    /// # Panics
    ///
    /// Panics if `counts` and `out` differ in length.
    pub(crate) fn split_chunk<T>(
        &self,
        index: usize,
        counts: &[i32],
        out: &mut [T],
        value_of: impl Fn(i32, i32) -> T,
    ) -> Result<(), NegativeCount> {
        assert_eq!(
            counts.len(),
            out.len(),
            "counts holds {} values and out {}",
            counts.len(),
            out.len()
        );
        if counts.iter().any(|&n| n < 0) {
            return Err(NegativeCount);
        }
        let mut rng = chunk_generator(self.seed, index);
        let mut samples = [0i32; SPLIT_BLOCK];
        for (counts, out) in counts.chunks(SPLIT_BLOCK).zip(out.chunks_mut(SPLIT_BLOCK)) {
            let samples = &mut samples[..counts.len()];
            samples.copy_from_slice(counts);
            sample_chunk(self.set, &mut rng, samples);
            for ((out, &n), &k) in out.iter_mut().zip(counts).zip(samples.iter()) {
                *out = value_of(n, k);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::{SplitMix64, Xoshiro256PlusPlus};

    fn ln_choose(n: u64, k: u64) -> f64 {
        (0..k).map(|i| ((n - i) as f64 / (i + 1) as f64).ln()).sum()
    }

    /// Pearson's chi-squared test against binomial (n, 0.5), with bins merged until each expects
    /// at least 20 samples. The bound sits about six standard deviations above the mean of the
    /// statistic, and the seeds are fixed, so the test does not flake.
    fn assert_binomial_half(name: &str, n: u64, samples: &[i32]) {
        let total = samples.len() as f64;
        let mut counts = vec![0u64; n as usize + 1];
        for &k in samples {
            assert!((0..=n as i32).contains(&k), "{name}: n = {n} gave {k}");
            counts[k as usize] += 1;
        }
        let (mut chi2, mut bins) = (0.0, 0);
        let (mut expected, mut observed) = (0.0, 0.0);
        for k in 0..=n {
            expected += (ln_choose(n, k) + n as f64 * 0.5f64.ln()).exp() * total;
            observed += counts[k as usize] as f64;
            if expected >= 20.0 || k == n {
                chi2 += (observed - expected).powi(2) / expected.max(1e-300);
                bins += 1;
                (expected, observed) = (0.0, 0.0);
            }
        }
        let df = (bins - 1).max(1) as f64;
        let bound = df + 6.0 * (2.0 * df).sqrt() + 10.0;
        assert!(
            chi2 < bound,
            "{name}: n = {n}, chi2 = {chi2:.1} over {bins} bins, bound {bound:.1}"
        );
    }

    const PATHS: [CpuFeatures; 3] = [
        CpuFeatures::Baseline,
        CpuFeatures::UpToAvx2,
        CpuFeatures::Detected,
    ];

    #[test]
    fn samples_follow_binomial_half() {
        for cpu in PATHS {
            for n in [
                0u64, 1, 2, 7, 63, 64, 65, 127, 200, 1000, 1024, 1025, 5000, 40_000,
            ] {
                let mut a = vec![n as i32; 100_000];
                binom_split(&mut a, 1, 0xF00D ^ n, cpu).unwrap();
                assert_binomial_half(&format!("{cpu:?}"), n, &a);
            }
        }
    }

    /// Counts that share a lane's word, as small counts do, must still get disjoint bits: the
    /// samples of neighbouring counts, in the same lane and in the next one, are uncorrelated.
    #[test]
    fn neighbouring_samples_are_uncorrelated() {
        for cpu in PATHS {
            for n in [1, 7, 20, 63, 64, 65, 100] {
                let len = 400_000;
                let mut a = vec![n; len];
                binom_split(&mut a, 1, 0xC0FFEE ^ n as u64, cpu).unwrap();
                let mean = n as f64 / 2.0;
                let variance = n as f64 / 4.0;
                for offset in [1, LANES] {
                    let covariance: f64 = a
                        .iter()
                        .zip(&a[offset..])
                        .map(|(&x, &y)| (x as f64 - mean) * (y as f64 - mean))
                        .sum::<f64>()
                        / (len - offset) as f64;
                    let correlation = covariance / variance;
                    // Six standard deviations of the correlation of independent samples.
                    let bound = 6.0 / ((len - offset) as f64).sqrt();
                    assert!(
                        correlation.abs() < bound,
                        "{cpu:?}: n = {n}, offset {offset}: correlation {correlation:.4}"
                    );
                }
            }
        }
    }

    #[test]
    fn negative_counts_are_refused() {
        let mut a = vec![5; 3 * CHUNK_COUNTS];
        a[CHUNK_COUNTS + 7] = -1;
        assert_eq!(
            binom_split(&mut a, 1, 1, CpuFeatures::Detected),
            Err(NegativeCount)
        );
    }

    #[test]
    fn samples_depend_only_on_the_seed() {
        let input: Vec<i32> = (0..5 * CHUNK_COUNTS as i32).map(|i| i % 300).collect();
        let (mut a, mut b, mut c) = (input.clone(), input.clone(), input);
        binom_split(&mut a, 1, 42, CpuFeatures::Detected).unwrap();
        binom_split(&mut b, 1, 42, CpuFeatures::Detected).unwrap();
        binom_split(&mut c, 1, 43, CpuFeatures::Detected).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn every_path_gives_the_same_samples() {
        // Counts of up to 40 000 need 625 words, so the AVX2 path adds its byte sums many times.
        let input: Vec<i32> = (0..3 * CHUNK_COUNTS as i32)
            .map(|i| {
                if i % 997 == 0 {
                    40_000 - i % 300
                } else {
                    i % 2000
                }
            })
            .collect();
        let samples: Vec<Vec<i32>> = PATHS
            .iter()
            .map(|&cpu| {
                let mut a = input.clone();
                binom_split(&mut a, 1, 7, cpu).unwrap();
                a
            })
            .collect();
        assert_eq!(samples[0], samples[1]);
        assert_eq!(samples[0], samples[2]);
    }

    #[test]
    fn splitter_matches_binom_split() {
        // Rows of 1000 counts: chunks of 65 rows, and a last chunk of 77 counts.
        let row_len = 1000;
        let chunk_len = rows_per_chunk(row_len) * row_len;
        let counts: Vec<i32> = (0..5 * chunk_len as i32 + 77).map(|i| i % 700).collect();
        let mut a = counts.clone();
        binom_split(&mut a, row_len, 9, CpuFeatures::Detected).unwrap();
        let splitter = Splitter::new(9, CpuFeatures::Detected);
        let mut b = vec![(0, 0); counts.len()];
        for (i, (c, o)) in counts
            .chunks(chunk_len)
            .zip(b.chunks_mut(chunk_len))
            .enumerate()
        {
            splitter.split_chunk(i, c, o, |n, k| (n, k)).unwrap();
        }
        assert!(
            b.iter()
                .zip(&a)
                .zip(&counts)
                .all(|((&(n, k), &ka), &c)| n == c && k == ka)
        );
        let mut negative = counts[..100].to_vec();
        negative[50] = -3;
        assert!(
            splitter
                .split_chunk(0, &negative, &mut b[..100], |n, k| (n, k))
                .is_err()
        );
    }

    #[test]
    fn splitmix64_is_rand_xoshiros() {
        let mut state = 0x1234_5678_9ABC_DEF0;
        let mut reference = SplitMix64::seed_from_u64(state);
        for _ in 0..10 {
            assert_eq!(splitmix64(&mut state), reference.next_u64());
        }
    }

    /// Each lane is a plain xoshiro256++ generator.
    #[test]
    fn lanes_are_xoshiro256plusplus() {
        let mut lanes = XoshiroLanes::<4>::seed_from_u64(5);
        let mut scalar: Vec<Xoshiro256PlusPlus> = (0..4)
            .map(|i| {
                let mut seed = [0u8; 32];
                for (w, bytes) in seed.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                    bytes.copy_from_slice(&lanes.s[w][i].to_le_bytes());
                }
                Xoshiro256PlusPlus::from_seed(seed)
            })
            .collect();
        let mut out = [0; 4];
        for _ in 0..10 {
            lanes.next_block(&mut out);
            for (o, s) in out.iter().zip(scalar.iter_mut()) {
                assert_eq!(*o, s.next_u64());
            }
        }
    }
}
