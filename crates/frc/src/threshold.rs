//! The thresholds that an FRC curve is compared with, and where a curve crosses one.

/// The threshold that an FRC curve is compared with to find the resolution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Threshold {
    /// The fixed threshold 1/7 (Rosenthal and Henderson, 2003).
    OneSeventh,
    /// The 1/2-bit threshold (van Heel and Schatz, 2005). It depends on the number of
    /// frequencies in a ring, and is 1 for the ring of the origin alone.
    HalfBit,
    /// A fixed threshold of this value.
    Fixed(f64),
}

impl Threshold {
    /// Returns the threshold of a ring of `count` frequencies.
    pub fn of_ring(self, count: u64) -> f64 {
        match self {
            Self::OneSeventh => 1.0 / 7.0,
            Self::Fixed(value) => value,
            Self::HalfBit => {
                // The signal-to-noise ratio of each half of the data for 1/2 bit of information.
                let snr = 0.5 * 2f64.sqrt() - 0.5;
                let root = (count as f64).sqrt();
                let factor = 2.0 * snr.sqrt();
                // In this order of operations, as in frc's Python, the threshold of the ring of
                // the origin alone is exactly 1, as its FRC is.
                (snr * root + (factor + 1.0)) / ((snr + 1.0) * root + factor)
            }
        }
    }
}

/// The point at which an FRC curve first drops below its threshold.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Crossing {
    /// The position of the crossing in rings: from ring 0 at the origin up, with fractions
    /// between two rings.
    pub ring: f64,
    /// The threshold at the crossing.
    pub threshold: f64,
}

impl Crossing {
    /// Returns the resolution of an n x n image in pixels: n / ring.
    pub fn resolution(&self, n: usize) -> f64 {
        n as f64 / self.ring
    }
}

/// Returns the point at which `curve` first drops below `threshold`, or `None` if it does not.
///
/// `curve` holds the FRC of ring r at index r, as [`FrcPlan::one_image`](crate::FrcPlan::one_image)
/// returns it, and `ring_counts` the number of frequencies in each ring, as
/// [`FrcPlan::ring_counts`](crate::FrcPlan::ring_counts) returns them. This function finds the
/// first ring i >= 1 whose FRC is below its threshold. It draws a straight line through the
/// FRC minus the threshold at rings i - 1 and i, and returns the point at which the line is
/// zero. A ring whose FRC is NaN counts as not below the threshold.
///
/// # Panics
///
/// Panics if `ring_counts` holds fewer values than `curve`.
pub fn first_crossing(
    curve: &[f64],
    ring_counts: &[u64],
    threshold: Threshold,
) -> Option<Crossing> {
    assert!(
        ring_counts.len() >= curve.len(),
        "ring_counts holds {} values, at least {} required",
        ring_counts.len(),
        curve.len()
    );
    let threshold_of = |r: usize| threshold.of_ring(ring_counts[r]);
    let above = |r: usize| curve[r] - threshold_of(r);
    let i = (1..curve.len()).find(|&r| above(r) < 0.0)?;
    let (before, after) = (above(i - 1), above(i));
    // Ring i - 1 is at or above its threshold, or NaN. Only ring 0, which is not searched, can
    // lie below it, for a curve that starts below its threshold.
    let fraction = if before.is_nan() {
        1.0
    } else {
        (before / (before - after)).clamp(0.0, 1.0)
    };
    let (threshold_before, threshold_after) = (threshold_of(i - 1), threshold_of(i));
    Some(Crossing {
        ring: (i - 1) as f64 + fraction,
        threshold: threshold_before + fraction * (threshold_after - threshold_before),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_half_bit_threshold_is_one_at_the_origin_and_falls_with_the_ring_size() {
        assert_eq!(Threshold::HalfBit.of_ring(1), 1.0);
        let (eight, thousand) = (
            Threshold::HalfBit.of_ring(8),
            Threshold::HalfBit.of_ring(1000),
        );
        assert!(
            1.0 > eight && eight > thousand && thousand > 0.1716,
            "{eight}, {thousand}"
        );
        // For large rings it approaches snr / (snr + 1) = 0.1716.
        assert!((Threshold::HalfBit.of_ring(1 << 40) - 0.17157).abs() < 1e-4);
    }

    #[test]
    fn the_crossing_lies_on_the_line_between_two_rings() {
        let counts = [1, 8, 12, 16, 28, 24];
        let curve = [1.0, 0.6, 0.3, 0.2, 0.1, 0.2];
        let crossing = first_crossing(&curve, &counts, Threshold::Fixed(0.15)).unwrap();
        assert_eq!(
            crossing,
            Crossing {
                ring: 3.5,
                threshold: 0.15
            }
        );
        assert_eq!(crossing.resolution(14), 4.0);
        // The first crossing counts, and a curve that never drops below has none.
        let crossing = first_crossing(&curve, &counts, Threshold::Fixed(0.4)).unwrap();
        assert!((crossing.ring - (1.0 + 0.2 / 0.3)).abs() < 1e-12);
        assert_eq!(
            first_crossing(&curve, &counts, Threshold::Fixed(0.05)),
            None
        );
    }

    #[test]
    fn the_half_bit_crossing_interpolates_the_threshold_too() {
        let counts = [1, 8, 12, 16];
        let curve = [1.0, 0.9, 0.2, 0.1];
        let crossing = first_crossing(&curve, &counts, Threshold::HalfBit).unwrap();
        let (t1, t2) = (
            Threshold::HalfBit.of_ring(8),
            Threshold::HalfBit.of_ring(12),
        );
        let fraction = (0.9 - t1) / ((0.9 - t1) - (0.2 - t2));
        assert!((crossing.ring - (1.0 + fraction)).abs() < 1e-12);
        assert!((crossing.threshold - (t1 + fraction * (t2 - t1))).abs() < 1e-12);
    }

    #[test]
    fn a_curve_that_starts_below_its_threshold_crosses_at_ring_0() {
        let crossing = first_crossing(&[0.5, 0.3, 0.2], &[1, 8, 12], Threshold::Fixed(0.6));
        assert_eq!(crossing.map(|c| c.ring), Some(0.0));
    }

    #[test]
    fn nan_rings_are_not_below_the_threshold() {
        let counts = [1, 8, 12, 16];
        let crossing = first_crossing(&[1.0, f64::NAN, 0.05, 0.3], &counts, Threshold::OneSeventh);
        assert_eq!(crossing.map(|c| c.ring), Some(2.0));
    }
}
