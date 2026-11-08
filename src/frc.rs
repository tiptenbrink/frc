//! Fourier ring correlation of two images, or of one image of counts split in two, as the Python
//! package frc computes it with `two_frc` and with `one_frc` and method 1: for one image, split
//! every count n into k ~ binomial (n, 0.5) and n - k. Fourier transform both images, and
//! correlate them over rings of equal distance from the origin. The FRC at ring r is
//! Re(sum F1 conj F2) / sqrt(sum |F1|^2 sum |F2|^2), for r < N / 2.
//!
//! Every step is bound by memory bandwidth, so the steps are arranged to move the image as few
//! times as possible:
//!
//! - Both images go through one complex FFT, of z = k + i (n - k) or z = a + i b: their spectra
//!   are F1(f) = (Z(f) + conj Z(-f)) / 2 and F2(f) = (Z(f) - conj Z(-f)) / 2i.
//! - The split writes z a block of rows at a time and transforms those rows while they are in
//!   cache.
//! - The spectrum is stored in panels of 8 columns and their 8 mirrors, so the column FFTs read
//!   memory in order, a panel at a time, without a transpose pass, and each panel holds both
//!   Z(f) and Z(-f) for its ring terms. The ring sums are taken while the panel is in cache, and
//!   the column FFTs are never written back.
//! - F1 and F2 of a real image satisfy F(-f) = conj F(f), so the ring sums run over half the plane.
//!
//! Ring r holds the frequencies f with r <= |f| < r + 1, the bins of DIPlib's `RadialSum`, which
//! frc used before it moved to this crate; both count the same frequencies in every ring.

use std::sync::Arc;

use rayon::prelude::*;
use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

use crate::binomial::{CpuFeatures, NegativeCount, Splitter};
use crate::platform;

/// The sums over one ring of the spectra F1 and F2 of the two halves.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RingSums {
    /// Re(sum F1 conj F2).
    pub correlation: f64,
    /// sum |F1|^2.
    pub power1: f64,
    /// sum |F2|^2.
    pub power2: f64,
    /// The number of frequencies in the ring.
    pub count: u64,
}

impl RingSums {
    pub fn frc(&self) -> f64 {
        self.correlation / (self.power1 * self.power2).sqrt()
    }
}

/// Pairs of mirrored columns, x and n - x, that one panel of the spectrum holds. Each panel holds
/// the 8 columns 8q + 1 ..= 8q + 8 followed by their mirrors n - 8q - 8 ..= n - 8q - 1, row after
/// row, so the column FFTs read and write memory in order, and the ring sums, which need Z(f) and
/// Z(-f), find both in the same panel. The columns that are their own mirror, 0 and for even n
/// also n / 2, share a last panel.
const PAIRS_PER_PANEL: usize = 8;

/// Counts that one task of the split samples and transforms: whole rows, 16 of them at 4096,
/// so that each task writes runs of 16 rows into every panel.
const SPLIT_TASK_COUNTS: usize = 65536;

/// Complex f32s between the starts of two rows of the column buffer beyond the row itself: one
/// cache line, so that rows do not start a multiple of 4 KiB apart, where the values that one
/// row of a panel adds to them would all fall in the same cache set.
const COLUMN_PAD: usize = 8;

/// Columns that one panel of the spectrum holds, where it starts, and for each of its columns the
/// position of the mirror column in the panel.
struct Panel {
    start: usize,
    columns: Vec<usize>,
    mirror_slots: Vec<usize>,
    /// The columns as runs of consecutive ones: (first column, its position in the panel, length).
    runs: Vec<(usize, usize, usize)>,
}

/// The runs of consecutive columns in `columns`, for copying a panel's row in a few pieces.
fn column_runs(columns: &[usize]) -> Vec<(usize, usize, usize)> {
    let mut runs: Vec<(usize, usize, usize)> = Vec::new();
    for (c, &x) in columns.iter().enumerate() {
        match runs.last_mut() {
            Some((first, _, len)) if *first + *len == x => *len += 1,
            _ => runs.push((x, c, 1)),
        }
    }
    runs
}

/// The panels of an n x n spectrum, and where each column lies in them.
struct PanelLayout {
    panels: Vec<Panel>,
    /// For each column: its panel, and its position in the panel.
    column_slots: Vec<(usize, usize)>,
}

impl PanelLayout {
    fn new(n: usize) -> Self {
        let pairs = (n - 1) / 2;
        let mut groups: Vec<Vec<usize>> = (0..pairs.div_ceil(PAIRS_PER_PANEL))
            .map(|q| {
                let first = q * PAIRS_PER_PANEL + 1;
                let last = (first + PAIRS_PER_PANEL - 1).min(pairs);
                (first..=last).chain(n - last..=n - first).collect()
            })
            .collect();
        groups.push(if n.is_multiple_of(2) {
            vec![0, n / 2]
        } else {
            vec![0]
        });
        let mut column_slots = vec![(0, 0); n];
        let mut start = 0;
        let panels = groups
            .into_iter()
            .enumerate()
            .map(|(p, columns)| {
                for (c, &x) in columns.iter().enumerate() {
                    column_slots[x] = (p, c);
                }
                let mirror_slots = columns
                    .iter()
                    .map(|&x| columns.iter().position(|&m| m == mirrored(x, n)).unwrap())
                    .collect();
                let runs = column_runs(&columns);
                let panel = Panel {
                    start,
                    columns,
                    mirror_slots,
                    runs,
                };
                start += panel.columns.len() * n;
                panel
            })
            .collect();
        Self {
            panels,
            column_slots,
        }
    }

    fn index(&self, y: usize, x: usize) -> usize {
        let (p, c) = self.column_slots[x];
        let panel = &self.panels[p];
        panel.start + y * panel.columns.len() + c
    }
}

/// What `one_frc` and `two_frc` need for images of one size, made once like an FFT plan.
pub struct FrcPlan {
    n: usize,
    fft: Arc<dyn Fft<f32>>,
    layout: PanelLayout,
    ring_counts: Vec<u64>,
    /// In the panel layout, the two images as z = k + i (n - k) or z = a + i b, then the
    /// spectrum Z after the row FFTs. `fft_columns` alone transforms the columns in place; `one_frc` does not write them.
    spectrum: Vec<Complex32>,
}

/// The signed frequency of index i of a DFT of length n.
fn frequency(i: usize, n: usize) -> i64 {
    if i <= (n - 1) / 2 {
        i as i64
    } else {
        i as i64 - n as i64
    }
}

fn ring_of(fy: i64, fx: i64) -> usize {
    // f64 takes the square root of these integers exactly, so floor sees perfect squares right.
    ((fy * fy + fx * fx) as f64).sqrt().floor() as usize
}

/// `len` zeros for the spectrum, with every page touched once, in parallel: each thread faults in
/// its own share of the pages, in order. Leaving that to the row pass, whose tasks write their
/// rows into every panel, made the first FRC of a plan take 60 ms longer than later ones on a
/// Ryzen 5 3400G in a VM. Touching a page costs much less than writing its zeros, which
/// `par_extend` would do: 24 against 18 ms for a plan at 4096 x 4096 on three of its threads.
fn zeroed_spectrum(len: usize) -> Vec<Complex32> {
    let mut spectrum = platform::zeroed_complex_vec(len);
    let page = 4096 / size_of::<Complex32>();
    spectrum
        .par_chunks_mut(page)
        .for_each(|page| page[0] = Complex32::default());
    spectrum
}

/// The number of frequencies in each ring r < n / 2 of an n x n spectrum. It goes by |fy| and
/// |fx|, each with the number of indices that have it, and follows the ring along |fx| by
/// comparison, as the ring sums do: a quarter of the frequencies, and no square roots.
pub fn ring_counts(n: usize) -> Vec<u64> {
    let half = n / 2;
    // |f| = a at one index for a = 0 and, for even n, a = n / 2 (frequency -n / 2), else at two.
    let indices_with = |a: usize| -> u64 { if a == 0 || 2 * a == n { 1 } else { 2 } };
    let mut counts = vec![0; half];
    for ay in 0..half {
        let mut r = ay as i64;
        for ax in 0..=half {
            let d2 = (ay * ay + ax * ax) as i64;
            while d2 >= (r + 1) * (r + 1) {
                r += 1;
            }
            if r >= half as i64 {
                break;
            }
            counts[r as usize] += indices_with(ay) * indices_with(ax);
        }
    }
    counts
}

/// The index of frequency -f for the index i of frequency f, in a DFT of length n: (n - i) % n,
/// without the division that % by a run-time n costs.
fn mirrored(i: usize, n: usize) -> usize {
    if i == 0 { 0 } else { n - i }
}

/// The rows that hold the half plane fy > 0 or fy = 0: 0 ..= (n - 1) / 2.
fn half_plane_rows(n: usize) -> usize {
    (n - 1) / 2 + 1
}

/// Ring sums of the half plane, not yet doubled, in slots 0..n/2, and the origin's terms in slot
/// n / 2.
type HalfPlaneSums = Vec<[f64; 3]>;

fn add_terms(a: &mut [f64; 3], b: [f64; 3]) {
    for i in 0..3 {
        a[i] += b[i];
    }
}

/// F1 conj(F2), |F1|^2 and |F2|^2 from Z(f) and Z(-f).
fn ring_terms(z: Complex32, z_mirror: Complex32) -> [f64; 3] {
    let zm = z_mirror.conj();
    let f1 = (z + zm) * 0.5;
    let f2 = (z - zm) * Complex32::new(0.0, -0.5);
    [
        (f1 * f2.conj()).re as f64,
        f1.norm_sqr() as f64,
        f2.norm_sqr() as f64,
    ]
}

/// Adds the terms of the frequencies of `panel` in the half plane to `sums`, where `at(c, y)` is
/// the spectrum at row y of the panel's column c.
fn add_panel_ring_terms(
    n: usize,
    panel: &Panel,
    at: impl Fn(usize, usize) -> Complex32,
    sums: &mut HalfPlaneSums,
) {
    let half = n / 2;
    for (c, &x) in panel.columns.iter().enumerate() {
        let fx = frequency(x, n);
        let m = panel.mirror_slots[c];
        if x == 0 {
            add_terms(&mut sums[half], ring_terms(at(c, 0), at(c, 0)));
        }
        // Row 0 belongs to the half plane only where fx > 0.
        let first = if fx > 0 { 0 } else { 1 };
        // Down the column |f|^2 only grows, and ring r holds r^2 <= |f|^2 < (r + 1)^2: the ring
        // follows with a comparison per row, and a run of rows in one ring is summed apart.
        let mut r = ring_of(first as i64, fx) as i64;
        let mut run = [0f64; 3];
        for y in first..half_plane_rows(n) {
            let d2 = fx * fx + (y * y) as i64;
            while d2 >= (r + 1) * (r + 1) {
                if r < half as i64 {
                    add_terms(&mut sums[r as usize], run);
                }
                run = [0.0; 3];
                r += 1;
            }
            if r >= half as i64 {
                break;
            }
            add_terms(&mut run, ring_terms(at(c, y), at(m, mirrored(y, n))));
        }
        if r < half as i64 {
            add_terms(&mut sums[r as usize], run);
        }
    }
}

/// Panels that one task of the column pass takes. A fixed number, so that the ring sums add up in
/// the same order, and to the same last bit, on any number of threads; rayon's fold and reduce
/// group the panels as its threads happen to steal them.
const PANELS_PER_TASK: usize = 4;

/// The sum of the ring sums of the tasks, in their order.
fn sums_in_order(n: usize, task_sums: &[HalfPlaneSums]) -> HalfPlaneSums {
    let mut sums = vec![[0f64; 3]; n / 2 + 1];
    for task in task_sums {
        for (sum, &term) in sums.iter_mut().zip(task) {
            add_terms(sum, term);
        }
    }
    sums
}

/// Rows of a panel that the column pass copies into its column buffer at a time.
const PANEL_COPY_ROWS: usize = 8;

/// Copies a panel of `width` columns into `columns`, column c from index c stride on. It loads
/// `PANEL_COPY_ROWS` rows and then stores them to each column at once. Storing each value right
/// after loading it took 18 ms per 4096 x 4096 spectrum in some processes and 47 ms in others on
/// a Ryzen 5 3400G, with the same instructions and cache misses, as if the CPU had learned to make
/// the loads wait for the stores; in blocks it takes 11 ms in every process.
fn panel_into_columns(data: &[Complex32], width: usize, columns: &mut [Complex32], stride: usize) {
    let rows = data.len() / width;
    let block_rows = rows / PANEL_COPY_ROWS * PANEL_COPY_ROWS;
    let (blocks, rest) = data.split_at(block_rows * width);
    for (b, block) in blocks.chunks_exact(PANEL_COPY_ROWS * width).enumerate() {
        let y = b * PANEL_COPY_ROWS;
        for c in 0..width {
            let mut values = [Complex32::default(); PANEL_COPY_ROWS];
            for (r, v) in values.iter_mut().enumerate() {
                *v = block[r * width + c];
            }
            columns[c * stride + y..][..PANEL_COPY_ROWS].copy_from_slice(&values);
        }
    }
    for (r, row) in rest.chunks_exact(width).enumerate() {
        for (c, &v) in row.iter().enumerate() {
            columns[c * stride + block_rows + r] = v;
        }
    }
}

/// The spectrum, split into its panels for the tasks of the column pass.
fn split_into_panels<'a>(
    mut spectrum: &'a mut [Complex32],
    layout: &PanelLayout,
    n: usize,
) -> Vec<&'a mut [Complex32]> {
    layout
        .panels
        .iter()
        .map(|panel| {
            let (this, rest) = std::mem::take(&mut spectrum).split_at_mut(panel.columns.len() * n);
            spectrum = rest;
            this
        })
        .collect()
}

/// The spectrum in blocks of `rows` rows for the tasks of the row pass: block b holds rows b rows
/// to (b + 1) rows of every panel, one run per panel, in the order of the panels.
fn row_blocks<'a>(
    spectrum: &'a mut [Complex32],
    layout: &PanelLayout,
    n: usize,
    rows: usize,
) -> Vec<Vec<&'a mut [Complex32]>> {
    let mut blocks: Vec<Vec<&mut [Complex32]>> = (0..n.div_ceil(rows))
        .map(|_| Vec::with_capacity(layout.panels.len()))
        .collect();
    for (data, panel) in split_into_panels(spectrum, layout, n)
        .into_iter()
        .zip(&layout.panels)
    {
        for (block, run) in blocks
            .iter_mut()
            .zip(data.chunks_mut(rows * panel.columns.len()))
        {
            block.push(run);
        }
    }
    blocks
}

/// What the row pass reads, for z.
#[derive(Clone, Copy)]
enum RowSource<'a> {
    /// An image of counts, split with `seed`: z = k + i (n - k).
    Split { counts: &'a [i32], seed: u64 },
    /// Two images: z = a + i b.
    Pair(&'a [f32], &'a [f32]),
    /// The rows that the spectrum holds.
    Spectrum,
}

/// What the column pass does with each panel after its column FFTs.
#[derive(Clone, Copy, PartialEq)]
enum ColumnStep {
    /// Write the transformed columns back to the spectrum.
    WriteBack,
    /// Add the panel's ring terms to the ring sums, and write nothing back.
    RingSums,
}

impl FrcPlan {
    /// A plan for images of n x n counts.
    pub fn new(n: usize) -> Self {
        assert!(n >= 2, "unsupported image size {n}");
        let ring_counts = ring_counts(n);
        Self {
            n,
            fft: FftPlanner::new().plan_fft_forward(n),
            layout: PanelLayout::new(n),
            ring_counts,
            spectrum: zeroed_spectrum(n * n),
        }
    }

    pub fn size(&self) -> usize {
        self.n
    }

    /// The number of frequencies in each ring r < N / 2.
    pub fn ring_counts(&self) -> &[u64] {
        &self.ring_counts
    }

    /// The FRC curve of an image of n x n counts in row-major order, split with `seed`: its value
    /// for each ring r < N / 2. The same counts and seed give the same curve.
    pub fn one_frc(&mut self, counts: &[i32], seed: u64) -> Result<Vec<f64>, NegativeCount> {
        Ok(self
            .one_frc_ring_sums(counts, seed)?
            .iter()
            .map(RingSums::frc)
            .collect())
    }

    /// The ring sums behind `one_frc`.
    pub fn one_frc_ring_sums(
        &mut self,
        counts: &[i32],
        seed: u64,
    ) -> Result<Vec<RingSums>, NegativeCount> {
        self.split_fft_rows(counts, seed)?;
        Ok(self.fft_columns_ring_sums())
    }

    /// The FRC curve of two n x n images in row-major order: its value for each ring r < N / 2.
    pub fn two_frc(&mut self, a: &[f32], b: &[f32]) -> Vec<f64> {
        self.two_frc_ring_sums(a, b)
            .iter()
            .map(RingSums::frc)
            .collect()
    }

    /// The ring sums behind `two_frc`.
    pub fn two_frc_ring_sums(&mut self, a: &[f32], b: &[f32]) -> Vec<RingSums> {
        self.rows_through_buffer(RowSource::Pair(a, b), true)
            .expect("only a split can fail");
        self.fft_columns_ring_sums()
    }

    /// Reads `source` a block of rows at a time and, if `transform`, transforms the rows: each
    /// task works on its rows in a buffer of its own, in row order, and copies them into every
    /// panel of the spectrum at the end.
    fn rows_through_buffer(
        &mut self,
        source: RowSource,
        transform: bool,
    ) -> Result<(), NegativeCount> {
        let n = self.n;
        match source {
            RowSource::Split { counts, .. } => {
                assert_eq!(counts.len(), n * n, "counts must be an n x n image")
            }
            RowSource::Pair(a, b) => {
                assert_eq!(a.len(), n * n, "images must be n x n");
                assert_eq!(b.len(), n * n, "images must be n x n");
            }
            RowSource::Spectrum => {}
        }
        let rows = (SPLIT_TASK_COUNTS / n).max(1);
        let seed = match source {
            RowSource::Split { seed, .. } => seed,
            RowSource::Pair(..) | RowSource::Spectrum => 0,
        };
        let splitter = Splitter::new(seed, CpuFeatures::Detected);
        let (fft, layout) = (&self.fft, &self.layout);
        row_blocks(&mut self.spectrum, layout, n, rows)
            .into_par_iter()
            .enumerate()
            .map_init(
                || {
                    (
                        vec![Complex32::default(); rows * n],
                        vec![Complex32::default(); fft.get_inplace_scratch_len()],
                    )
                },
                |(buffer, scratch), (task, mut runs)| {
                    let y0 = task * rows;
                    let height = rows.min(n - y0);
                    let buffer = &mut buffer[..height * n];
                    let task_rows = y0 * n..(y0 + height) * n;
                    match source {
                        RowSource::Spectrum => {
                            for (run, panel) in runs.iter().zip(&layout.panels) {
                                let width = panel.columns.len();
                                for (r, row) in run.chunks_exact(width).enumerate() {
                                    for &(x, c, len) in &panel.runs {
                                        buffer[r * n + x..r * n + x + len]
                                            .copy_from_slice(&row[c..c + len]);
                                    }
                                }
                            }
                        }
                        RowSource::Split { counts, .. } => {
                            splitter.split_chunk(task, &counts[task_rows], buffer, |n, k| {
                                Complex32::new(k as f32, (n - k) as f32)
                            })?;
                        }
                        RowSource::Pair(a, b) => {
                            for ((z, &a), &b) in buffer
                                .iter_mut()
                                .zip(&a[task_rows.clone()])
                                .zip(&b[task_rows])
                            {
                                *z = Complex32::new(a, b);
                            }
                        }
                    }
                    if transform {
                        fft.process_with_scratch(buffer, scratch);
                    }
                    let mut row_of_panel = [Complex32::default(); 2 * PAIRS_PER_PANEL];
                    for (run, panel) in runs.iter_mut().zip(&layout.panels) {
                        let width = panel.columns.len();
                        for (r, out) in run.chunks_exact_mut(width).enumerate() {
                            for &(x, c, len) in &panel.runs {
                                row_of_panel[c..c + len]
                                    .copy_from_slice(&buffer[r * n + x..r * n + x + len]);
                            }
                            platform::copy_streaming(out, &row_of_panel[..width]);
                        }
                    }
                    platform::finish_streaming();
                    Ok(())
                },
            )
            .collect()
    }

    /// The split alone, without the row FFTs: a step of `one_frc` for checks and benchmarks.
    pub fn split(&mut self, counts: &[i32], seed: u64) -> Result<(), NegativeCount> {
        self.rows_through_buffer(RowSource::Split { counts, seed }, false)
    }

    /// The split and the row FFTs, a block of rows at a time while they are in cache.
    pub fn split_fft_rows(&mut self, counts: &[i32], seed: u64) -> Result<(), NegativeCount> {
        self.rows_through_buffer(RowSource::Split { counts, seed }, true)
    }

    /// The row FFTs alone, of whatever the spectrum holds.
    pub fn fft_rows(&mut self) {
        self.rows_through_buffer(RowSource::Spectrum, true)
            .expect("only a split can fail");
    }

    /// The column FFTs, a panel at a time: each task copies its panel's columns into a buffer that
    /// stays in cache and transforms them there, then does `step` with them.
    fn columns_through_buffer(&mut self, step: ColumnStep) -> HalfPlaneSums {
        let n = self.n;
        let (fft, layout) = (&self.fft, &self.layout);
        let stride = n + COLUMN_PAD;
        let mut panels = split_into_panels(&mut self.spectrum, layout, n);
        let task_sums: Vec<HalfPlaneSums> = panels
            .par_chunks_mut(PANELS_PER_TASK)
            .zip(layout.panels.par_chunks(PANELS_PER_TASK))
            .map_init(
                || {
                    (
                        vec![Complex32::default(); 2 * PAIRS_PER_PANEL * stride],
                        vec![Complex32::default(); fft.get_inplace_scratch_len()],
                    )
                },
                |(columns, scratch), (data, panels)| {
                    let mut sums = vec![[0f64; 3]; n / 2 + 1];
                    for (data, panel) in data.iter_mut().zip(panels) {
                        let width = panel.columns.len();
                        panel_into_columns(data, width, columns, stride);
                        for c in 0..width {
                            fft.process_with_scratch(
                                &mut columns[c * stride..c * stride + n],
                                scratch,
                            );
                        }
                        match step {
                            ColumnStep::WriteBack => {
                                for (y, row) in data.chunks_exact_mut(width).enumerate() {
                                    for (c, v) in row.iter_mut().enumerate() {
                                        *v = columns[c * stride + y];
                                    }
                                }
                            }
                            ColumnStep::RingSums => {
                                add_panel_ring_terms(
                                    n,
                                    panel,
                                    |c, y| columns[c * stride + y],
                                    &mut sums,
                                );
                            }
                        }
                    }
                    sums
                },
            )
            .collect();
        sums_in_order(n, &task_sums)
    }

    /// The column FFTs alone, written back to the spectrum, for checks and benchmarks.
    pub fn fft_columns(&mut self) {
        self.columns_through_buffer(ColumnStep::WriteBack);
    }

    /// The column FFTs and the ring sums in one pass over the spectrum, which it leaves as the
    /// row FFTs left it.
    pub fn fft_columns_ring_sums(&mut self) -> Vec<RingSums> {
        let sums = self.columns_through_buffer(ColumnStep::RingSums);
        self.full_ring_sums(&sums)
    }

    /// The ring sums of the spectrum after `fft_columns`, for checks and benchmarks.
    pub fn ring_sums(&self) -> Vec<RingSums> {
        let n = self.n;
        let panels = split_panels_ref(&self.spectrum, &self.layout, n);
        let task_sums: Vec<HalfPlaneSums> = panels
            .par_chunks(PANELS_PER_TASK)
            .zip(self.layout.panels.par_chunks(PANELS_PER_TASK))
            .map(|(data, panels)| {
                let mut sums = vec![[0f64; 3]; n / 2 + 1];
                for (data, panel) in data.iter().zip(panels) {
                    let width = panel.columns.len();
                    add_panel_ring_terms(n, panel, |c, y| data[y * width + c], &mut sums);
                }
                sums
            })
            .collect();
        let sums = sums_in_order(n, &task_sums);
        self.full_ring_sums(&sums)
    }

    /// The sums over whole rings from those over the half plane: doubled, plus the origin.
    fn full_ring_sums(&self, sums: &HalfPlaneSums) -> Vec<RingSums> {
        let half = self.n / 2;
        let origin = sums[half];
        sums[..half]
            .iter()
            .zip(&self.ring_counts)
            .enumerate()
            .map(|(r, (s, &count))| {
                let [c, p1, p2] = s.map(|v| 2.0 * v);
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

    /// The spectrum, or after `split` the halves, in row-major order.
    pub fn spectrum_row_major(&self) -> Vec<Complex32> {
        let n = self.n;
        (0..n * n)
            .map(|i| self.spectrum[self.layout.index(i / n, i % n)])
            .collect()
    }
}

/// `split_into_panels` for reading.
fn split_panels_ref<'a>(
    mut spectrum: &'a [Complex32],
    layout: &PanelLayout,
    n: usize,
) -> Vec<&'a [Complex32]> {
    layout
        .panels
        .iter()
        .map(|panel| {
            let (this, rest) = spectrum.split_at(panel.columns.len() * n);
            spectrum = rest;
            this
        })
        .collect()
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

    /// A synthetic image for checks and benchmarks: Poisson counts around `mean` times one plus the
    /// cosines of `PATTERN_IMAGE_WAVES`, whose rings (5, 19, 51 and 120, for n > 240) have an FRC near
    /// 1, while the noise elsewhere has an FRC near 0.
    pub(crate) fn pattern_image(n: usize, mean: f64, seed: u64) -> Vec<i32> {
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
    fn naive_ring_sums(h1: &[f64], h2: &[f64], n: usize) -> Vec<RingSums> {
        let as_complex = |h: &[f64]| {
            h.iter()
                .map(|&v| Complex64::new(v, 0.0))
                .collect::<Vec<_>>()
        };
        let (f1, f2) = (
            naive_dft2(&as_complex(h1), n),
            naive_dft2(&as_complex(h2), n),
        );
        let mut sums = vec![RingSums::default(); n / 2];
        for y in 0..n {
            for x in 0..n {
                let r = ring_of(frequency(y, n), frequency(x, n));
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

    #[test]
    fn ring_sums_match_their_definition() {
        for n in [16, 21, 24, 30] {
            let counts = pattern_image(n, 50.0, n as u64);
            let mut plan = FrcPlan::new(n);
            plan.split(&counts, 7).unwrap();
            let halves = plan.spectrum_row_major();
            plan.fft_rows();
            plan.fft_columns();
            let got = plan.ring_sums();
            let h1: Vec<f64> = halves.iter().map(|z| z.re as f64).collect();
            let h2: Vec<f64> = halves.iter().map(|z| z.im as f64).collect();
            assert!(
                h1.iter()
                    .zip(&h2)
                    .zip(&counts)
                    .all(|((a, b), &c)| a + b == c as f64)
            );
            let want = naive_ring_sums(&h1, &h2, n);
            for (r, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g.count, w.count, "n = {n}, ring {r}");
                let scale = w.power1.max(w.power2);
                for (gv, wv) in [
                    (g.correlation, w.correlation),
                    (g.power1, w.power1),
                    (g.power2, w.power2),
                ] {
                    assert!(
                        (gv - wv).abs() <= 1e-5 * scale,
                        "n = {n}, ring {r}: {gv} against {wv}"
                    );
                }
            }
        }
    }

    #[test]
    fn ring_counts_match_a_direct_count() {
        for n in (2..70).chain([255, 256, 1000]) {
            let mut direct = vec![0; n / 2];
            for y in 0..n {
                for x in 0..n {
                    let r = ring_of(frequency(y, n), frequency(x, n));
                    if r < n / 2 {
                        direct[r] += 1;
                    }
                }
            }
            assert_eq!(ring_counts(n), direct, "n = {n}");
        }
    }

    #[test]
    fn fused_ring_sums_match_their_steps() {
        for n in [2, 3, 17, 48, 64] {
            let counts = pattern_image(n, 40.0, 9);
            let mut plan = FrcPlan::new(n);
            plan.split_fft_rows(&counts, 2).unwrap();
            let fused = plan.fft_columns_ring_sums();
            plan.fft_columns();
            let stepwise = plan.ring_sums();
            for (r, (f, s)) in fused.iter().zip(&stepwise).enumerate() {
                assert_eq!(f.count, s.count, "n = {n}, ring {r}");
                for (a, b) in [
                    (f.correlation, s.correlation),
                    (f.power1, s.power1),
                    (f.power2, s.power2),
                ] {
                    assert!(
                        (a - b).abs() <= 1e-9 * b.abs().max(1.0),
                        "n = {n}, ring {r}: {a} against {b}"
                    );
                }
            }
        }
    }

    #[test]
    fn fused_split_matches_its_steps() {
        let n = 64;
        let counts = pattern_image(n, 80.0, 1);
        let mut plan = FrcPlan::new(n);
        plan.split(&counts, 3).unwrap();
        plan.fft_rows();
        let stepwise = plan.spectrum.clone();
        plan.split_fft_rows(&counts, 3).unwrap();
        assert_eq!(stepwise, plan.spectrum);
    }

    #[test]
    fn pattern_rings_correlate() {
        let n = 512;
        let mut plan = FrcPlan::new(n);
        let frc = plan.one_frc(&pattern_image(n, 100.0, 2), 5).unwrap();
        // The weakest wave, at ring 120, stands about 90 times above its ring's noise at this
        // size, and its cross terms with the noise move the FRC by a few percent.
        for r in [5, 19, 51, 120] {
            assert!(frc[r] > 0.9, "ring {r}: {}", frc[r]);
        }
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
            pool.install(|| FrcPlan::new(n).one_frc(&counts, 3).unwrap())
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
        let mut plan = FrcPlan::new(n);
        assert_eq!(
            plan.one_frc(&counts, 11).unwrap(),
            plan.one_frc(&counts, 11).unwrap()
        );
        assert_ne!(
            plan.one_frc(&counts, 11).unwrap(),
            plan.one_frc(&counts, 12).unwrap()
        );
    }

    #[test]
    fn two_frc_matches_its_definition() {
        for n in [16, 21] {
            let a: Vec<f32> = pattern_image(n, 50.0, 1)
                .iter()
                .map(|&v| v as f32 * 0.5)
                .collect();
            let b: Vec<f32> = pattern_image(n, 50.0, 2)
                .iter()
                .map(|&v| v as f32)
                .collect();
            let got = FrcPlan::new(n).two_frc_ring_sums(&a, &b);
            let as_f64 = |h: &[f32]| h.iter().map(|&v| v as f64).collect::<Vec<_>>();
            let want = naive_ring_sums(&as_f64(&a), &as_f64(&b), n);
            for (r, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g.count, w.count, "n = {n}, ring {r}");
                let scale = w.power1.max(w.power2);
                for (gv, wv) in [
                    (g.correlation, w.correlation),
                    (g.power1, w.power1),
                    (g.power2, w.power2),
                ] {
                    assert!(
                        (gv - wv).abs() <= 1e-5 * scale,
                        "n = {n}, ring {r}: {gv} against {wv}"
                    );
                }
            }
        }
    }

    /// `one_frc` is `two_frc` of the two halves of its split.
    #[test]
    fn one_frc_is_two_frc_of_the_halves() {
        let n = 96;
        let counts = pattern_image(n, 60.0, 5);
        let mut plan = FrcPlan::new(n);
        plan.split(&counts, 13).unwrap();
        let halves = plan.spectrum_row_major();
        let k: Vec<f32> = halves.iter().map(|z| z.re).collect();
        let rest: Vec<f32> = halves.iter().map(|z| z.im).collect();
        assert_eq!(plan.two_frc(&k, &rest), plan.one_frc(&counts, 13).unwrap());
    }

    #[test]
    fn negative_counts_are_refused() {
        let n = 32;
        let mut counts = pattern_image(n, 10.0, 4);
        counts[500] = -1;
        assert!(FrcPlan::new(n).one_frc(&counts, 1).is_err());
    }
}
