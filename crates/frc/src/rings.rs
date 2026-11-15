//! Which frequencies make up each ring of an FRC curve.

/// Which frequencies f make up each ring r of an FRC curve.
///
/// A frequency belongs to a ring by its distance |f| from the origin, in pixels of the spectrum.
/// Value r of a curve is placed at the spatial frequency r / n. With [`Rings::Nearest`], the
/// frequencies of ring r lie around |f| = r. With [`Rings::Floor`], they lie between r and
/// r + 1, on average about half a ring above r. The curve then places each ring at a lower
/// frequency than the frequencies it holds, and the resolution comes out coarser.
///
/// The default is [`Rings::Nearest`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Rings {
    /// Ring r holds the frequencies whose distance |f| rounds to r, which are those with
    /// r - 1/2 ≤ |f| < r + 1/2.
    #[default]
    Nearest,
    /// Ring r holds the frequencies with r ≤ |f| < r + 1, as DIPlib's `RadialSum` bins them.
    /// Earlier versions of frc used these rings.
    Floor,
}

impl Rings {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rings_hold_the_radii_that_round_to_them() {
        for squared in 0..200_000 {
            let radius = (squared as f64).sqrt();
            assert_eq!(
                Rings::Nearest.of_squared_radius(squared),
                (radius + 0.5).floor() as usize,
                "|f|² = {squared}"
            );
            assert_eq!(
                Rings::Floor.of_squared_radius(squared),
                radius.floor() as usize,
                "|f|² = {squared}"
            );
        }
    }

    #[test]
    fn each_ring_starts_where_the_ring_before_it_ends() {
        for rings in [Rings::Nearest, Rings::Floor] {
            for r in 1..2000 {
                let first = rings.first_squared_radius(r);
                assert_eq!(rings.of_squared_radius(first), r, "{rings:?}, ring {r}");
                assert_eq!(
                    rings.of_squared_radius(first - 1),
                    r - 1,
                    "{rings:?}, ring {r}"
                );
                assert_eq!(
                    rings.first_squared_radius(r + 1) - first,
                    rings.squared_width(r),
                    "{rings:?}, ring {r}"
                );
            }
        }
    }
}
