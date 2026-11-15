//! Fourier ring correlation of two images, or of one image of counts split in two.
//!
//! Both images go through one complex FFT, of z = a + i b. For one image of counts n, a is the
//! binomial sample k and b is n - k, both multiplied by the window. With Z the spectrum of z, the spectra of the two images are
//! F1(f) = (Z(f) + conj Z(-f)) / 2 and F2(f) = (Z(f) - conj Z(-f)) / 2i. The FRC at ring r is
//! Re(sum F1 conj F2) / sqrt(sum |F1|^2 sum |F2|^2) over the frequencies f of ring r, for
//! r < n / 2. [`Rings`] says which frequencies those are.
//!
//! The image passes through memory twice:
//!
//! 1. The row pass takes a block of rows at a time. It writes z into a buffer that fits in the
//!    cache, transforms the rows there, and copies them into the spectrum.
//! 2. The column pass takes a panel of the spectrum at a time: 8 columns and the 8 columns that
//!    mirror them. It copies the panel into a buffer that fits in the cache, transforms the
//!    columns there, and adds their ring terms to the ring sums. A panel holds both Z(f) and
//!    Z(-f) for each of its frequencies, and the transformed columns are never written back.
//!
//! The spectrum is stored panel after panel, so the column pass reads memory in order. F1 and F2
//! of a real image satisfy F(-f) = conj F(f), so the ring sums run over half the plane.

use std::convert::Infallible;
use std::hint::select_unpredictable;
use std::ops::Range;
use std::sync::Arc;

use rayon::prelude::*;
use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

use crate::binomial::{NegativeCount, Splitter, rows_per_chunk};
use crate::cpu::CpuFeatures;
use crate::platform;
use crate::rings::Rings;
use crate::window::Window;

/// The sums over one ring of the spectra F1 and F2 of the two images.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RingSums {
    /// The real part of the sum of F1 conj F2.
    pub correlation: f64,
    /// The sum of |F1|².
    pub power1: f64,
    /// The sum of |F2|².
    pub power2: f64,
    /// The number of frequencies in the ring.
    pub count: u64,
}

impl RingSums {
    /// Returns the FRC of the ring: `correlation / sqrt(power1 * power2)`.
    ///
    /// The FRC is NaN for a ring in which one of the images has no power.
    pub fn frc(&self) -> f64 {
        self.correlation / (self.power1 * self.power2).sqrt()
    }
}

/// Pairs of mirrored columns, x and n - x, that one panel of the spectrum holds.
const PAIRS_PER_PANEL: usize = 8;

/// Complex values between the end of one column and the start of the next in the column buffer:
/// one cache line. Without it the columns of a 4096 x 4096 spectrum start a multiple of 4 KiB
/// apart, and the values that one row of a panel adds to them all fall in the same cache set.
const COLUMN_PAD: usize = 8;

/// Panels that one task of the column pass takes. The number is fixed so that the ring sums add
/// up in the same order, and to the same last bit, on any number of threads. Rayon's `fold` and
/// `reduce` group the panels as its threads happen to steal them.
const PANELS_PER_TASK: usize = 4;

/// Rows of a panel that the column pass copies into its column buffer at a time.
const PANEL_COPY_ROWS: usize = 8;

/// Rows of a column whose ring terms the column pass computes before it adds them to their rings.
const TERM_ROWS: usize = 256;

/// The image rows that one task of the row pass takes: a chunk of the split, so that a task
/// splits its rows with the generators of one chunk.
fn rows_per_task(n: usize) -> usize {
    rows_per_chunk(n)
}

/// A run of adjacent columns of the spectrum in a panel.
struct ColumnRun {
    /// The first column of the run.
    first: usize,
    /// The position of that column in each row of the panel.
    slot: usize,
    /// The number of columns.
    len: usize,
}

/// The columns that a panel holds, in the order in which each of its rows stores them.
enum PanelColumns {
    /// The columns `first .. first + pairs`, then their mirrors in column order:
    /// `n - first - pairs + 1 ..= n - first`. The mirror of the column at slot s is at slot
    /// `2 * pairs - 1 - s`.
    MirroredPairs { first: usize, pairs: usize },
    /// The columns that are their own mirror: 0 and, for even n, n / 2.
    OwnMirrors,
}

/// A panel of the spectrum: all n rows of a few columns, row after row.
struct Panel {
    /// The index in the spectrum of the panel's first value.
    start: usize,
    columns: PanelColumns,
}

impl Panel {
    /// Returns the number of columns.
    fn width(&self, n: usize) -> usize {
        match self.columns {
            PanelColumns::MirroredPairs { pairs, .. } => 2 * pairs,
            PanelColumns::OwnMirrors => 2 - n % 2,
        }
    }

    /// Returns the columns as runs of adjacent ones. The second run is empty for the panel of
    /// an odd n's only column that is its own mirror.
    fn column_runs(&self, n: usize) -> [ColumnRun; 2] {
        let (first, second, len) = match self.columns {
            PanelColumns::MirroredPairs { first, pairs } => (first, n + 1 - first - pairs, pairs),
            PanelColumns::OwnMirrors => (0, n / 2, 1),
        };
        let second_len = self.width(n) - len;
        [
            ColumnRun {
                first,
                slot: 0,
                len,
            },
            ColumnRun {
                first: second,
                slot: len,
                len: second_len,
            },
        ]
    }
}

/// Returns the panels of an n x n spectrum: `PAIRS_PER_PANEL` pairs of mirrored columns each,
/// from column 1 on, and last the columns that are their own mirror.
fn spectrum_panels(n: usize) -> Vec<Panel> {
    let pairs = (n - 1) / 2;
    let mut panels = Vec::with_capacity(pairs.div_ceil(PAIRS_PER_PANEL) + 1);
    let mut start = 0;
    let columns = (1..=pairs)
        .step_by(PAIRS_PER_PANEL)
        .map(|first| PanelColumns::MirroredPairs {
            first,
            pairs: PAIRS_PER_PANEL.min(pairs + 1 - first),
        })
        .chain([PanelColumns::OwnMirrors]);
    for columns in columns {
        let panel = Panel { start, columns };
        start += panel.width(n) * n;
        panels.push(panel);
    }
    panels
}

/// Returns `len` zeros for the spectrum, with every page touched once, in parallel: each thread
/// faults in its own share of the pages, in order.
///
/// When the row pass touched the pages first, the first FRC of a plan took 60 ms longer than
/// later ones on a Ryzen 5 3400G in a VM: every task of the row pass writes rows into every
/// panel. Touching a page costs much less than writing its zeros, which `par_extend` would do.
/// A plan at 4096 x 4096 took 18 against 24 ms on three threads of that CPU.
fn zeroed_spectrum(len: usize) -> Vec<Complex32> {
    let mut spectrum = platform::zeroed_complex_vec(len);
    let page = 4096 / size_of::<Complex32>();
    spectrum
        .par_chunks_mut(page)
        .for_each(|page| page[0] = Complex32::default());
    spectrum
}

/// Returns the number of frequencies in each ring r < n / 2 of an n x n spectrum.
pub fn ring_counts(n: usize, rings: Rings) -> Vec<u64> {
    let half = n / 2;
    // A frequency with |fx| or |fy| = half lies outside every ring. Below half, |f| = a at one
    // index for a = 0, else at two.
    let indices_with = |a: usize| -> u64 { if a == 0 { 1 } else { 2 } };
    // By |fy| and |fx| >= |fy|, an eighth of the frequencies; the transposed ones mirror them.
    // Along |fx| the ring only grows, so a comparison follows it, without square roots.
    let add_row = |mut counts: Vec<u64>, ay: usize| {
        let mut r = rings.of_squared_radius(2 * ay * ay);
        for ax in ay..half {
            while ay * ay + ax * ax >= rings.first_squared_radius(r + 1) {
                r += 1;
            }
            if r >= half {
                break;
            }
            let transposed = if ax == ay { 1 } else { 2 };
            counts[r] += transposed * indices_with(ay) * indices_with(ax);
        }
        counts
    };
    // A plan computes the counts once, and a 4096 x 4096 plan took 5.4 ms for them on one thread
    // of a Core Ultra 7 155H, with every quarter of the frequencies.
    (0..half)
        .into_par_iter()
        .with_min_len(64)
        .fold(|| vec![0; half], add_row)
        .reduce(
            || vec![0; half],
            |mut counts, more| {
                for (count, more) in counts.iter_mut().zip(more) {
                    *count += more;
                }
                counts
            },
        )
}

/// The ring sums of a task of the column pass over the half plane: the frequencies with fy > 0,
/// and those with fy = 0 and fx > 0. Element r holds the three sums of ring r, for r < n / 2.
/// Element n / 2 holds the terms of the origin.
type HalfPlaneSums = Vec<[f64; 3]>;

fn add_terms(sum: &mut [f64; 3], terms: [f32; 3]) {
    for (sum, term) in sum.iter_mut().zip(terms) {
        *sum += term as f64;
    }
}

/// Returns the terms of the origin from Z(0), where F1 = Re Z and F2 = Im Z, in f64.
///
/// An f64 holds the product of two f32s exactly. The FRC of ring 0, which holds the origin
/// alone, is therefore exactly 1 for two images with positive sums, and not 1 with a rounding
/// error of either sign. The half-bit threshold is exactly 1 there too, and frc's search for
/// the resolution starts from which of the two is larger.
fn origin_terms(z: Complex32) -> [f64; 3] {
    let (f1, f2) = (z.re as f64, z.im as f64);
    [f1 * f2, f1 * f1, f2 * f2]
}

/// Returns Re(F1 conj F2), |F1|² and |F2|² of a frequency f from Z(f) and Z(-f).
#[inline(always)]
fn ring_terms(z: Complex32, z_mirror: Complex32) -> [f32; 3] {
    let mirror = z_mirror.conj();
    let f1 = (z + mirror) * 0.5;
    // (z - mirror) / 2i.
    let difference = z - mirror;
    let f2 = Complex32::new(0.5 * difference.im, -0.5 * difference.re);
    [f1.re * f2.re + f1.im * f2.im, f1.norm_sqr(), f2.norm_sqr()]
}

/// The ring terms of up to `TERM_ROWS` rows of a pair of mirrored columns: element t holds, for
/// each row, the sum of term t of the two frequencies (fx, fy) and (-fx, fy).
type PairTerms = [[f32; TERM_ROWS]; 3];

/// Writes to `terms`, for each row fy of `rows`, `weight` times the sum of the ring terms of the
/// frequencies (fx, fy) and (-fx, fy), where `column` is column fx and `mirror` column -fx.
///
/// The loop is one that LLVM turns into vector code: it only reads the columns, forward and
/// backward, and every row's terms depend on that row alone.
fn pair_ring_terms(
    terms: &mut PairTerms,
    column: &[Complex32],
    mirror: &[Complex32],
    rows: Range<usize>,
    weight: f32,
) {
    // Row fy of a column is mirrored by row n - fy of its mirror column.
    let mirrored_rows = column.len() + 1 - rows.end..column.len() + 1 - rows.start;
    let of_column = column[rows.clone()]
        .iter()
        .zip(mirror[mirrored_rows.clone()].iter().rev());
    let of_mirror = mirror[rows].iter().zip(column[mirrored_rows].iter().rev());
    let [correlation, power1, power2] = terms;
    let terms = correlation.iter_mut().zip(power1).zip(power2);
    // Zipped iterators index nothing, so no bounds check keeps the loop from becoming vector
    // code.
    for ((((correlation, power1), power2), (&z, &z_mirror)), (&w, &w_mirror)) in
        terms.zip(of_column).zip(of_mirror)
    {
        let (a, b) = (ring_terms(z, z_mirror), ring_terms(w, w_mirror));
        *correlation = weight * (a[0] + b[0]);
        *power1 = weight * (a[1] + b[1]);
        *power2 = weight * (a[2] + b[2]);
    }
}

/// Adds to `sums` the ring terms of the frequencies (fx, fy) and (-fx, fy) with fy >= 1, as far
/// as they lie in a ring r < n / 2, times `weight`. `column` is column fx of the spectrum and
/// `mirror` column -fx.
#[allow(clippy::too_many_arguments)]
fn add_pair_ring_sums(
    n: usize,
    rings: Rings,
    fx: usize,
    column: &[Complex32],
    mirror: &[Complex32],
    weight: f32,
    terms: &mut PairTerms,
    sums: &mut [[f64; 3]],
) {
    let half = n / 2;
    // The rows whose frequencies lie in a ring r < half: fx² + fy² below the start of ring half.
    let end = (rings.first_squared_radius(half) - fx * fx - 1).isqrt() + 1;
    for first in (1..end).step_by(TERM_ROWS) {
        let len = TERM_ROWS.min(end - first);
        pair_ring_terms(terms, column, mirror, first..first + len, weight);
        // Down the column |f| grows by at most 1 a row, so each row is in the ring of the row
        // before it or in the next one, and a comparison follows the ring without a square root.
        // Whether a row enters the next ring is a choice without a branch: such a branch is
        // taken on about two rows in three, and a CPU predicts it poorly.
        let mut ring = rings.of_squared_radius(fx * fx + first * first);
        // The smallest fy² in the next ring.
        let mut next_ring_at = rings.first_squared_radius(ring + 1) - fx * fx;
        for i in 0..len {
            let entered = (first + i) * (first + i) >= next_ring_at;
            let next_ring_after = next_ring_at + rings.squared_width(ring + 1);
            ring += entered as usize;
            next_ring_at = select_unpredictable(entered, next_ring_after, next_ring_at);
            for (sum, terms) in sums[ring].iter_mut().zip(&*terms) {
                *sum += terms[i] as f64;
            }
        }
    }
}

/// Adds to `sums` the ring terms of the frequencies of `panel` in the half plane. `columns`
/// holds the panel's transformed columns, the column at slot s from index `s * stride` on.
fn add_panel_ring_sums(
    n: usize,
    rings: Rings,
    panel: &Panel,
    columns: &[Complex32],
    stride: usize,
    terms: &mut PairTerms,
    sums: &mut HalfPlaneSums,
) {
    let half = n / 2;
    let column = |slot: usize| &columns[slot * stride..][..n];
    match panel.columns {
        PanelColumns::MirroredPairs { first, pairs } => {
            for pair in 0..pairs {
                let fx = first + pair;
                // For odd n the last pair has fx = half, outside every ring.
                if fx >= half {
                    break;
                }
                let (z, mirror) = (column(pair), column(2 * pairs - 1 - pair));
                // Row 0 is in the half plane for fx > 0 only. Frequency (fx, 0) lies in ring fx
                // with both kinds of rings.
                add_terms(&mut sums[fx], ring_terms(z[0], mirror[0]));
                add_pair_ring_sums(n, rings, fx, z, mirror, 1.0, terms, sums);
            }
        }
        // Column n / 2 of an even n has |fx| = half, outside every ring.
        PanelColumns::OwnMirrors => {
            let z = column(0);
            sums[half] = origin_terms(z[0]);
            // A pair of this column with itself counts every frequency twice.
            add_pair_ring_sums(n, rings, 0, z, z, 0.5, terms, sums);
        }
    }
}

/// Copies a panel of `width` columns into `columns`, the column at slot s from index
/// `s * stride` on. It loads `PANEL_COPY_ROWS` rows and then stores them to each column at once.
///
/// Storing each value right after loading it took 18 ms per 4096 x 4096 spectrum in some
/// processes and 47 ms in others on a Ryzen 5 3400G. Both ran the same instructions and missed
/// the cache as often, so the CPU probably made the loads wait for the stores in the slow ones.
/// In blocks the copy takes 11 ms in every process.
fn panel_into_columns(data: &[Complex32], width: usize, columns: &mut [Complex32], stride: usize) {
    let rows = data.len() / width;
    let block_rows = rows / PANEL_COPY_ROWS * PANEL_COPY_ROWS;
    let (blocks, rest) = data.split_at(block_rows * width);
    for (b, block) in blocks.chunks_exact(PANEL_COPY_ROWS * width).enumerate() {
        let y = b * PANEL_COPY_ROWS;
        for slot in 0..width {
            let mut values = [Complex32::default(); PANEL_COPY_ROWS];
            for (r, v) in values.iter_mut().enumerate() {
                *v = block[r * width + slot];
            }
            columns[slot * stride + y..][..PANEL_COPY_ROWS].copy_from_slice(&values);
        }
    }
    for (r, row) in rest.chunks_exact(width).enumerate() {
        for (slot, &v) in row.iter().enumerate() {
            columns[slot * stride + block_rows + r] = v;
        }
    }
}

/// Splits the spectrum for the tasks of the row pass. Task t gets `panels.len()` elements, at
/// `t * panels.len()`: for each panel in order, its rows `t * rows .. (t + 1) * rows`.
fn task_rows_of_panels<'a>(
    mut spectrum: &'a mut [Complex32],
    panels: &[Panel],
    n: usize,
    rows: usize,
) -> Vec<&'a mut [Complex32]> {
    let tasks = n.div_ceil(rows);
    let mut task_rows = Vec::new();
    task_rows.resize_with(tasks * panels.len(), Default::default);
    for (p, panel) in panels.iter().enumerate() {
        let width = panel.width(n);
        let (data, rest) = std::mem::take(&mut spectrum).split_at_mut(width * n);
        spectrum = rest;
        for (task, rows_of_panel) in data.chunks_mut(rows * width).enumerate() {
            task_rows[task * panels.len() + p] = rows_of_panel;
        }
    }
    task_rows
}

/// The FFT plan and the working memory for FRC curves of n x n images.
///
/// Make one plan for a size and its rings, and use it for every image of that size. A plan holds
/// an n x n complex spectrum, 8 n² bytes.
pub struct FrcPlan {
    n: usize,
    rings: Rings,
    fft: Arc<dyn Fft<f32>>,
    panels: Vec<Panel>,
    ring_counts: Vec<u64>,
    /// The spectrum after the row FFTs, panel after panel.
    spectrum: Vec<Complex32>,
}

impl FrcPlan {
    /// Creates a plan for n x n images, whose curves sum over `rings`.
    ///
    /// # Panics
    ///
    /// Panics if n < 2.
    pub fn new(n: usize, rings: Rings) -> Self {
        assert!(n >= 2, "images of {n} x {n}, at least 2 x 2 required");
        Self {
            n,
            rings,
            fft: FftPlanner::new().plan_fft_forward(n),
            panels: spectrum_panels(n),
            ring_counts: ring_counts(n, rings),
            spectrum: zeroed_spectrum(n * n),
        }
    }

    /// Returns the side n of the images.
    pub fn size(&self) -> usize {
        self.n
    }

    /// Returns the rings that the plan's curves sum over.
    pub fn rings(&self) -> Rings {
        self.rings
    }

    /// Returns the number of frequencies in each ring r < n / 2, as [`ring_counts`] computes
    /// them.
    pub fn ring_counts(&self) -> &[u64] {
        &self.ring_counts
    }

    /// Computes the FRC curve of one image of counts: its value for each ring r < n / 2.
    ///
    /// `counts` holds the n x n image in row-major order: the counts as they were measured,
    /// without a window. This method splits every count c into a sample k of binomial(c, 1/2)
    /// and c - k, multiplies both images that result by `window`, and correlates them.
    ///
    /// The split comes before the window because it gives two independent images only of
    /// Poisson counts, whose variance is their mean. A count that a window has scaled by w has
    /// w times the mean and w² times the variance, and the noise of its two halves correlates.
    ///
    /// The same counts, window and seed give the same curve: that of [`FrcPlan::two_frc`] for
    /// the samples k that [`binom_split`](crate::binom_split) draws with the same seed and the
    /// rest c - k, each multiplied by the window.
    ///
    /// # Errors
    ///
    /// Returns [`NegativeCount`] if a count is negative.
    ///
    /// # Panics
    ///
    /// Panics if `counts` does not hold n² counts, or if the alpha of a Tukey window is NaN.
    pub fn one_frc(
        &mut self,
        counts: &[i32],
        window: Window,
        seed: u64,
    ) -> Result<Vec<f64>, NegativeCount> {
        let sums = self.one_frc_ring_sums(counts, window, seed)?;
        Ok(sums.iter().map(RingSums::frc).collect())
    }

    /// Computes the sums over each ring r < n / 2 that [`FrcPlan::one_frc`] takes its curve
    /// from.
    ///
    /// # Errors
    ///
    /// Returns [`NegativeCount`] if a count is negative.
    ///
    /// # Panics
    ///
    /// Panics if `counts` does not hold n² counts, or if the alpha of a Tukey window is NaN.
    pub fn one_frc_ring_sums(
        &mut self,
        counts: &[i32],
        window: Window,
        seed: u64,
    ) -> Result<Vec<RingSums>, NegativeCount> {
        self.split_fft_rows(counts, window, seed)?;
        Ok(self.fft_columns_ring_sums())
    }

    /// Computes the FRC curve of two images: its value for each ring r < n / 2.
    ///
    /// `a` and `b` hold the n x n images in row-major order, already multiplied by a window:
    /// this method applies none.
    ///
    /// # Panics
    ///
    /// Panics if `a` or `b` does not hold n² values.
    pub fn two_frc(&mut self, a: &[f32], b: &[f32]) -> Vec<f64> {
        let sums = self.two_frc_ring_sums(a, b);
        sums.iter().map(RingSums::frc).collect()
    }

    /// Computes the sums over each ring r < n / 2 that [`FrcPlan::two_frc`] takes its curve
    /// from.
    ///
    /// # Panics
    ///
    /// Panics if `a` or `b` does not hold n² values.
    pub fn two_frc_ring_sums(&mut self, a: &[f32], b: &[f32]) -> Vec<RingSums> {
        self.check_len("a", a.len());
        self.check_len("b", b.len());
        let Ok(()) = self.fft_rows_of(|_, values, z| {
            for ((z, &a), &b) in z.iter_mut().zip(&a[values.clone()]).zip(&b[values]) {
                *z = Complex32::new(a, b);
            }
            Ok::<(), Infallible>(())
        });
        self.fft_columns_ring_sums()
    }

    fn check_len(&self, name: &str, len: usize) {
        let n = self.n;
        assert_eq!(len, n * n, "{name} holds {len} values, {n} x {n} required");
    }

    /// The row pass of [`FrcPlan::one_frc`], for benchmarks: the split, the window and the row
    /// FFTs.
    #[doc(hidden)]
    pub fn split_fft_rows(
        &mut self,
        counts: &[i32],
        window: Window,
        seed: u64,
    ) -> Result<(), NegativeCount> {
        self.check_len("counts", counts.len());
        let n = self.n;
        let splitter = Splitter::new(seed, CpuFeatures::Detected);
        let taper = window.taper(n);
        self.fft_rows_of(|task, values, z| {
            let first_row = values.start / n;
            splitter.split_chunk(task, &counts[values], z, |count, k| {
                Complex32::new(k as f32, (count - k) as f32)
            })?;
            if let Some(weights) = &taper {
                for (row, &row_weight) in z.chunks_exact_mut(n).zip(&weights[first_row..]) {
                    for (z, &weight) in row.iter_mut().zip(weights) {
                        *z = z.scale(row_weight * weight);
                    }
                }
            }
            Ok(())
        })
    }

    /// The row pass. Each task takes a block of rows: `write_z(task, values, z)` writes the
    /// values of the images at the indices `values` to `z`, in a buffer of the task's own. The
    /// task transforms the rows there and copies them into every panel of the spectrum.
    fn fft_rows_of<E: Send>(
        &mut self,
        write_z: impl Fn(usize, Range<usize>, &mut [Complex32]) -> Result<(), E> + Sync,
    ) -> Result<(), E> {
        let n = self.n;
        let rows = rows_per_task(n);
        let (fft, panels) = (&self.fft, &self.panels);
        let mut task_rows = task_rows_of_panels(&mut self.spectrum, panels, n, rows);
        task_rows
            .par_chunks_mut(panels.len())
            .enumerate()
            .map_init(
                || {
                    (
                        vec![Complex32::default(); rows * n],
                        vec![Complex32::default(); fft.get_inplace_scratch_len()],
                    )
                },
                |(buffer, scratch), (task, rows_of_panels)| {
                    let first = task * rows * n;
                    let buffer = &mut buffer[..(rows * n).min(n * n - first)];
                    write_z(task, first..first + buffer.len(), buffer)?;
                    fft.process_with_scratch(buffer, scratch);
                    let mut panel_row = [Complex32::default(); 2 * PAIRS_PER_PANEL];
                    for (rows_of_panel, panel) in rows_of_panels.iter_mut().zip(panels) {
                        let width = panel.width(n);
                        let runs = panel.column_runs(n);
                        let panel_rows = rows_of_panel.chunks_exact_mut(width);
                        for (row, panel_out) in buffer.chunks_exact(n).zip(panel_rows) {
                            for run in &runs {
                                panel_row[run.slot..][..run.len]
                                    .copy_from_slice(&row[run.first..][..run.len]);
                            }
                            platform::copy_streaming(panel_out, &panel_row[..width]);
                        }
                    }
                    platform::finish_streaming();
                    Ok(())
                },
            )
            .collect()
    }

    /// The column pass, for benchmarks: the column FFTs and the ring sums of the spectrum that
    /// the row pass left.
    #[doc(hidden)]
    pub fn fft_columns_ring_sums(&self) -> Vec<RingSums> {
        let (n, rings) = (self.n, self.rings);
        let (fft, spectrum) = (&self.fft, &self.spectrum);
        let stride = n + COLUMN_PAD;
        let task_sums: Vec<HalfPlaneSums> = self
            .panels
            .par_chunks(PANELS_PER_TASK)
            .map_init(
                || {
                    (
                        vec![Complex32::default(); 2 * PAIRS_PER_PANEL * stride],
                        vec![Complex32::default(); fft.get_inplace_scratch_len()],
                        [[0f32; TERM_ROWS]; 3],
                    )
                },
                |(columns, scratch, terms), panels| {
                    let mut sums = vec![[0f64; 3]; n / 2 + 1];
                    for panel in panels {
                        let width = panel.width(n);
                        let data = &spectrum[panel.start..][..width * n];
                        panel_into_columns(data, width, columns, stride);
                        for column in columns.chunks_exact_mut(stride).take(width) {
                            fft.process_with_scratch(&mut column[..n], scratch);
                        }
                        add_panel_ring_sums(n, rings, panel, columns, stride, terms, &mut sums);
                    }
                    sums
                },
            )
            .collect();
        self.whole_ring_sums(&task_sums)
    }

    /// Returns the sums over whole rings from those of the tasks over the half plane, added in
    /// the order of the tasks: twice the half plane's, plus the origin's terms in ring 0.
    fn whole_ring_sums(&self, task_sums: &[HalfPlaneSums]) -> Vec<RingSums> {
        let half = self.n / 2;
        let mut sums = vec![[0f64; 3]; half + 1];
        for task in task_sums {
            for (sum, task_sum) in sums.iter_mut().zip(task) {
                for (sum, term) in sum.iter_mut().zip(task_sum) {
                    *sum += term;
                }
            }
        }
        let origin = sums[half];
        sums[..half]
            .iter()
            .zip(&self.ring_counts)
            .enumerate()
            .map(|(r, (sum, &count))| {
                let [c, p1, p2] = sum.map(|v| 2.0 * v);
                let o = if r == 0 { origin } else { [0.0; 3] };
                RingSums {
                    correlation: c + o[0],
                    power1: p1 + o[1],
                    power2: p2 + o[2],
                    count,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Poisson};
    use rand_xoshiro::Xoshiro256PlusPlus;
    use rustfft::num_complex::Complex64;

    /// Spatial frequencies (u, v, amplitude) of the cosine patterns in `pattern_image`.
    const PATTERN_IMAGE_WAVES: [(f64, f64, f64); 4] = [
        (3.0, 5.0, 0.4),
        (17.0, -9.0, 0.2),
        (40.0, 33.0, 0.1),
        (120.0, 7.0, 0.05),
    ];

    /// Poisson counts around `mean` times one plus the cosines of `PATTERN_IMAGE_WAVES`, whose
    /// rings have an FRC near 1 for n > 240, while the noise elsewhere has an FRC near 0. The
    /// waves lie at |f| = 5.83, 19.24, 51.86 and 120.20.
    fn pattern_image(n: usize, mean: f64, seed: u64) -> Vec<i32> {
        let mut image = vec![0; n * n];
        image.par_chunks_mut(n).enumerate().for_each(|(y, row)| {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(
                seed ^ (y as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            );
            for (x, v) in row.iter_mut().enumerate() {
                let phase = |u: f64, w: f64| {
                    2.0 * std::f64::consts::PI * (u * x as f64 + w * y as f64) / n as f64
                };
                let waves: f64 = PATTERN_IMAGE_WAVES
                    .iter()
                    .map(|&(u, w, a)| a * phase(u, w).cos())
                    .sum();
                *v = Poisson::new(mean * (1.0 + waves)).unwrap().sample(&mut rng) as i32;
            }
        });
        image
    }

    /// The signed frequency of index i of a DFT of length n.
    fn frequency(i: usize, n: usize) -> i64 {
        if i <= (n - 1) / 2 {
            i as i64
        } else {
            i as i64 - n as i64
        }
    }

    /// The ring of frequency (fx, fy) by its definition, from |f| in f64.
    fn ring_of(rings: Rings, fy: i64, fx: i64) -> usize {
        let radius = ((fy * fy + fx * fx) as f64).sqrt();
        match rings {
            Rings::Nearest => (radius + 0.5).floor() as usize,
            Rings::Floor => radius.floor() as usize,
        }
    }

    const BOTH_RINGS: [Rings; 2] = [Rings::Nearest, Rings::Floor];

    /// The 2D DFT by its definition, in f64.
    fn naive_dft2(z: &[Complex64], n: usize) -> Vec<Complex64> {
        let w = |k: usize| {
            Complex64::from_polar(1.0, -2.0 * std::f64::consts::PI * k as f64 / n as f64)
        };
        let mut rows = vec![Complex64::default(); n * n];
        for y in 0..n {
            for u in 0..n {
                rows[y * n + u] = (0..n).map(|x| z[y * n + x] * w(u * x % n)).sum();
            }
        }
        let mut out = vec![Complex64::default(); n * n];
        for v in 0..n {
            for u in 0..n {
                out[v * n + u] = (0..n).map(|y| rows[y * n + u] * w(v * y % n)).sum();
            }
        }
        out
    }

    /// The ring sums of two real images by their definition, over the whole plane.
    fn naive_ring_sums(a: &[f32], b: &[f32], n: usize, rings: Rings) -> Vec<RingSums> {
        let spectrum = |image: &[f32]| {
            let z: Vec<Complex64> = image
                .iter()
                .map(|&v| Complex64::new(v as f64, 0.0))
                .collect();
            naive_dft2(&z, n)
        };
        let (f1, f2) = (spectrum(a), spectrum(b));
        let mut sums = vec![RingSums::default(); n / 2];
        for y in 0..n {
            for x in 0..n {
                let r = ring_of(rings, frequency(y, n), frequency(x, n));
                if r < n / 2 {
                    let (a, b) = (f1[y * n + x], f2[y * n + x]);
                    let s = &mut sums[r];
                    s.correlation += (a * b.conj()).re;
                    s.power1 += a.norm_sqr();
                    s.power2 += b.norm_sqr();
                    s.count += 1;
                }
            }
        }
        sums
    }

    /// Two images with different patterns and different brightness.
    fn two_images(n: usize) -> (Vec<f32>, Vec<f32>) {
        let image = |seed: u64, scale: f32| -> Vec<f32> {
            let counts = pattern_image(n, 50.0, seed);
            counts.iter().map(|&v| v as f32 * scale).collect()
        };
        (image(1, 0.5), image(2, 1.0))
    }

    #[test]
    fn two_frc_matches_its_definition() {
        for (n, rings) in [2, 3, 4, 5, 16, 17, 21, 24, 30, 33]
            .into_iter()
            .flat_map(|n| BOTH_RINGS.map(|rings| (n, rings)))
        {
            let (a, b) = two_images(n);
            let got = FrcPlan::new(n, rings).two_frc_ring_sums(&a, &b);
            let want = naive_ring_sums(&a, &b, n, rings);
            assert_eq!(got.len(), want.len());
            for (r, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g.count, w.count, "n = {n}, {rings:?}, ring {r}");
                let scale = w.power1.max(w.power2);
                for (gv, wv) in [
                    (g.correlation, w.correlation),
                    (g.power1, w.power1),
                    (g.power2, w.power2),
                ] {
                    assert!(
                        (gv - wv).abs() <= 1e-5 * scale,
                        "n = {n}, {rings:?}, ring {r}: {gv} against {wv}"
                    );
                }
            }
        }
    }

    /// Sizes above `TERM_ROWS` rows per column and above one task per pass, against a sum over
    /// the whole plane in f64 of the spectrum that rustfft computes.
    #[test]
    fn two_frc_of_large_images_matches_a_sum_over_the_whole_plane() {
        for (n, rings) in [(600, Rings::Nearest), (1030, Rings::Floor)] {
            let (a, b) = two_images(n);
            let got = FrcPlan::new(n, rings).two_frc_ring_sums(&a, &b);
            let fft = FftPlanner::<f32>::new().plan_fft_forward(n);
            let spectrum = |image: &[f32]| {
                let mut z: Vec<Complex32> = image.iter().map(|&v| Complex32::new(v, 0.0)).collect();
                fft.process(&mut z);
                let mut columns: Vec<Complex32> =
                    (0..n * n).map(|i| z[i % n * n + i / n]).collect();
                fft.process(&mut columns);
                columns
            };
            let (f1, f2) = (spectrum(&a), spectrum(&b));
            let mut want = vec![RingSums::default(); n / 2];
            for x in 0..n {
                for y in 0..n {
                    let r = ring_of(rings, frequency(y, n), frequency(x, n));
                    if r < n / 2 {
                        let (a, b) = (f1[x * n + y], f2[x * n + y]);
                        want[r].correlation += (a.re * b.re + a.im * b.im) as f64;
                        want[r].power1 += a.norm_sqr() as f64;
                        want[r].power2 += b.norm_sqr() as f64;
                        want[r].count += 1;
                    }
                }
            }
            for (r, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g.count, w.count, "n = {n}, ring {r}");
                let scale = w.power1.max(w.power2);
                for (gv, wv) in [
                    (g.correlation, w.correlation),
                    (g.power1, w.power1),
                    (g.power2, w.power2),
                ] {
                    assert!(
                        (gv - wv).abs() <= 1e-4 * scale,
                        "n = {n}, ring {r}: {gv} against {wv}"
                    );
                }
            }
        }
    }

    #[test]
    fn ring_counts_match_a_direct_count() {
        for (n, rings) in (2..70)
            .chain([255, 256, 1000])
            .flat_map(|n| BOTH_RINGS.map(|rings| (n, rings)))
        {
            let mut direct = vec![0; n / 2];
            for y in 0..n {
                for x in 0..n {
                    let r = ring_of(rings, frequency(y, n), frequency(x, n));
                    if r < n / 2 {
                        direct[r] += 1;
                    }
                }
            }
            assert_eq!(ring_counts(n, rings), direct, "n = {n}, {rings:?}");
        }
    }

    #[test]
    fn one_frc_is_two_frc_of_its_windowed_halves() {
        // 300 and 1030 take several chunks of the split, the last one shorter.
        for n in [2, 3, 17, 96, 300, 1030] {
            let counts = pattern_image(n, 60.0, 5);
            let mut samples = counts.clone();
            crate::binom_split(&mut samples, n, 13, CpuFeatures::Detected).unwrap();
            let mut plan = FrcPlan::new(n, Rings::default());
            for window in [Window::None, Window::default(), Window::Tukey(1.0)] {
                let weights = window.weights(n);
                let windowed = |half: &dyn Fn(usize) -> i32| -> Vec<f32> {
                    (0..n * n)
                        .map(|i| half(i) as f32 * (weights[i / n] * weights[i % n]))
                        .collect()
                };
                let k = windowed(&|i| samples[i]);
                let rest = windowed(&|i| counts[i] - samples[i]);
                assert_eq!(
                    plan.two_frc_ring_sums(&k, &rest),
                    plan.one_frc_ring_sums(&counts, window, 13).unwrap(),
                    "n = {n}, {window:?}"
                );
            }
        }
    }

    #[test]
    fn pattern_rings_correlate() {
        let n = 512;
        let mut plan = FrcPlan::new(n, Rings::default());
        let image = pattern_image(n, 100.0, 2);
        let frc = plan.one_frc(&image, Window::None, 5).unwrap();
        assert_eq!(frc[0], 1.0);
        assert_eq!(plan.one_frc(&image, Window::default(), 5).unwrap()[0], 1.0);
        // The weakest wave, at ring 120, stands about 90 times above its ring's noise at this
        // size, and its cross terms with the noise move the FRC by a few percent.
        for (u, w, _) in PATTERN_IMAGE_WAVES {
            let r = Rings::default().of_squared_radius((u * u + w * w) as usize);
            assert!(frc[r] > 0.9, "ring {r}: {}", frc[r]);
        }
        // With Floor rings, the wave at |f| = 5.83 lies in ring 5 instead of ring 6.
        let floor = FrcPlan::new(n, Rings::Floor)
            .one_frc(&image, Window::None, 5)
            .unwrap();
        assert!(
            floor[5] > 0.9 && frc[5] < 0.5,
            "{} and {}",
            floor[5],
            frc[5]
        );
        let mut noise: Vec<f64> = frc[130..].iter().map(|v| v.abs()).collect();
        noise.sort_by(f64::total_cmp);
        assert!(
            noise[noise.len() / 2] < 0.05,
            "median |FRC| of the noise rings {}",
            noise[noise.len() / 2]
        );
    }

    #[test]
    fn every_thread_count_gives_the_same_curve() {
        let n = 300;
        let counts = pattern_image(n, 40.0, 11);
        let curve = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                FrcPlan::new(n, Rings::default())
                    .one_frc(&counts, Window::default(), 3)
                    .unwrap()
            })
        };
        let one = curve(1);
        for threads in [2, 3, 8] {
            assert_eq!(curve(threads), one, "{threads} threads");
        }
    }

    #[test]
    fn same_seed_same_curve() {
        let n = 100;
        let counts = pattern_image(n, 30.0, 3);
        let mut plan = FrcPlan::new(n, Rings::default());
        let window = Window::default();
        assert_eq!(
            plan.one_frc(&counts, window, 11).unwrap(),
            plan.one_frc(&counts, window, 11).unwrap()
        );
        assert_ne!(
            plan.one_frc(&counts, window, 11).unwrap(),
            plan.one_frc(&counts, window, 12).unwrap()
        );
        assert_ne!(
            plan.one_frc(&counts, window, 11).unwrap(),
            plan.one_frc(&counts, Window::None, 11).unwrap()
        );
    }

    #[test]
    fn negative_counts_are_refused() {
        let n = 32;
        let mut counts = pattern_image(n, 10.0, 4);
        counts[500] = -1;
        assert_eq!(
            FrcPlan::new(n, Rings::default()).one_frc(&counts, Window::default(), 1),
            Err(NegativeCount)
        );
    }
}
