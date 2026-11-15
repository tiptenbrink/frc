//! The rings of an FRC curve: which frequencies make up each ring, and how many there are.

use rayon::prelude::*;

use crate::threshold::{Crossing, Threshold, crossing_of};

/// Which frequencies f make up each ring r of an FRC curve.
///
/// A frequency belongs to a ring by its distance |f| from the origin, in pixels of the spectrum.
/// Value r of a curve is placed at the spatial frequency r / n. With [`RingBinning::Nearest`],
/// the frequencies of ring r lie around |f| = r. With [`RingBinning::Floor`], they lie between r
/// and r + 1, on average about half a ring above r. The curve then places each ring at a lower
/// frequency than the frequencies it holds, and the resolution comes out coarser.
///
/// The default is [`RingBinning::Nearest`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RingBinning {
    /// Ring r holds the frequencies whose distance |f| rounds to r, which are those with
    /// r - 1/2 ≤ |f| < r + 1/2.
    #[default]
    Nearest,
    /// Ring r holds the frequencies with r ≤ |f| < r + 1, as DIPlib's `RadialSum` bins them.
    /// Earlier versions of frc used these rings.
    Floor,
}

impl RingBinning {
    /// Returns the ring of the frequencies f with |f|² = `squared`.
    pub(crate) fn of_squared_radius(self, squared: usize) -> usize {
        match self {
            // round(sqrt(d)) = floor((sqrt(4 d) + 1) / 2), and the floor of half of
            // sqrt(4 d) + 1 depends on the integer part of sqrt(4 d) alone.
            Self::Nearest => (4 * squared).isqrt().div_ceil(2),
            Self::Floor => squared.isqrt(),
        }
    }

    /// Returns the smallest |f|² of the frequencies in ring r, for r >= 1.
    pub(crate) fn first_squared_radius(self, r: usize) -> usize {
        match self {
            // The smallest integer d with sqrt(d) >= r - 1/2: d >= r² - r + 1/4.
            Self::Nearest => r * r - r + 1,
            Self::Floor => r * r,
        }
    }

    /// Returns how much larger the smallest |f|² of ring r + 1 is than that of ring r, for
    /// r >= 1.
    #[inline(always)]
    pub(crate) fn squared_width(self, r: usize) -> usize {
        2 * r + (self == Self::Floor) as usize
    }
}

/// The rings r < n / 2 of the spectrum of n x n images: their binning and the number of
/// frequencies in each.
///
/// Make one to find where the FRC curve of an n x n image crosses its threshold. Computing the
/// rings needs no Fourier transform: an [`FrcPlan`](crate::FrcPlan) holds one, and you can also
/// make one on its own.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rings {
    n: usize,
    binning: RingBinning,
    counts: Vec<u64>,
}

impl Rings {
    /// Computes the rings of the spectrum of n x n images.
    ///
    /// # Panics
    ///
    /// Panics if n < 2.
    pub fn new(n: usize, binning: RingBinning) -> Self {
        assert!(n >= 2, "images of {n} x {n}, at least 2 x 2 required");
        Self {
            n,
            binning,
            counts: count_frequencies(n, binning),
        }
    }

    /// Returns the side n of the images.
    pub fn size(&self) -> usize {
        self.n
    }

    /// Returns which frequencies make up each ring.
    pub fn binning(&self) -> RingBinning {
        self.binning
    }

    /// Returns the number of frequencies in each ring r < n / 2.
    pub fn counts(&self) -> &[u64] {
        &self.counts
    }

    /// Returns the point at which `curve` first drops below `threshold`, or `None` if it does
    /// not.
    ///
    /// `curve` holds the FRC of ring r at index r, as [`FrcPlan::one_image`](crate::FrcPlan::one_image)
    /// returns it for images of this size and binning. This method finds the first ring i >= 1
    /// whose FRC is below its threshold. It draws a straight line through the FRC minus the
    /// threshold at rings i - 1 and i, and returns the point at which the line is zero. A ring
    /// whose FRC is NaN counts as not below the threshold.
    ///
    /// # Panics
    ///
    /// Panics if `curve` holds more values than there are rings, n / 2.
    pub fn first_crossing(&self, curve: &[f64], threshold: Threshold) -> Option<Crossing> {
        assert!(
            curve.len() <= self.counts.len(),
            "curve holds {} values, but images of {n} x {n} have {} rings",
            curve.len(),
            self.counts.len(),
            n = self.n
        );
        crossing_of(curve, &self.counts, self.n, threshold)
    }
}

/// Returns the number of frequencies in each ring r < n / 2 of an n x n spectrum.
fn count_frequencies(n: usize, binning: RingBinning) -> Vec<u64> {
    let half = n / 2;
    // A frequency with |fx| or |fy| = half lies outside every ring. Below half, |f| = a at one
    // index for a = 0, else at two.
    let indices_with = |a: usize| -> u64 { if a == 0 { 1 } else { 2 } };
    // By |fy| and |fx| >= |fy|, an eighth of the frequencies; the transposed ones mirror them.
    // Along |fx| the ring only grows, so a comparison follows it, without square roots.
    let add_row = |mut counts: Vec<u64>, ay: usize| {
        let mut r = binning.of_squared_radius(2 * ay * ay);
        for ax in ay..half {
            while ay * ay + ax * ax >= binning.first_squared_radius(r + 1) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rings_hold_the_radii_that_round_to_them() {
        for squared in 0..200_000 {
            let radius = (squared as f64).sqrt();
            assert_eq!(
                RingBinning::Nearest.of_squared_radius(squared),
                (radius + 0.5).floor() as usize,
                "|f|² = {squared}"
            );
            assert_eq!(
                RingBinning::Floor.of_squared_radius(squared),
                radius.floor() as usize,
                "|f|² = {squared}"
            );
        }
    }

    #[test]
    fn each_ring_starts_where_the_ring_before_it_ends() {
        for binning in [RingBinning::Nearest, RingBinning::Floor] {
            for r in 1..2000 {
                let first = binning.first_squared_radius(r);
                assert_eq!(binning.of_squared_radius(first), r, "{binning:?}, ring {r}");
                assert_eq!(
                    binning.of_squared_radius(first - 1),
                    r - 1,
                    "{binning:?}, ring {r}"
                );
                assert_eq!(
                    binning.first_squared_radius(r + 1) - first,
                    binning.squared_width(r),
                    "{binning:?}, ring {r}"
                );
            }
        }
    }

    #[test]
    fn a_crossing_knows_the_resolution_of_its_size() {
        let rings = Rings::new(14, RingBinning::Nearest);
        let curve = [1.0, 0.6, 0.3, 0.2, 0.1, 0.2];
        let crossing = rings
            .first_crossing(&curve, Threshold::Fixed(0.15))
            .unwrap();
        assert_eq!((crossing.ring, crossing.resolution), (3.5, 4.0));
    }

    #[test]
    #[should_panic(expected = "curve holds 8 values, but images of 14 x 14 have 7 rings")]
    fn a_curve_of_another_size_is_refused() {
        Rings::new(14, RingBinning::Nearest).first_crossing(&[0.5; 8], Threshold::OneSeventh);
    }
}
