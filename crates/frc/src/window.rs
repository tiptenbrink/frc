//! The window that tapers an image to zero at its edges.

use std::f64::consts::PI;

/// The window that [`FrcPlan::one_frc`](crate::FrcPlan::one_frc) multiplies both halves of an
/// image by, after it has split the counts.
///
/// The Fourier transform takes an image to repeat, so its opposite edges meet with a jump. A
/// window that falls to zero at the edges removes the jump, which would otherwise correlate
/// between the two halves. The window of an image is the product of [`Window::weights`] along
/// its rows and along its columns.
///
/// The default is the Tukey window with an alpha of 1/8.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Window {
    /// No window: every weight is 1.
    None,
    /// The symmetric Tukey window, as `scipy.signal.windows.tukey` computes it. Alpha is the
    /// fraction of the window that its two cosine tapers take together. An alpha of 0 or less
    /// gives no window, and one of 1 or more the Hann window.
    Tukey(f64),
}

impl Default for Window {
    fn default() -> Self {
        Self::Tukey(0.125)
    }
}

impl Window {
    /// Returns the weights of a window of `len` points.
    ///
    /// # Panics
    ///
    /// Panics if the alpha of a Tukey window is NaN.
    pub fn weights(self, len: usize) -> Vec<f32> {
        self.taper(len).unwrap_or_else(|| vec![1.0; len])
    }

    /// Returns the weights of a window of `len` points, or `None` if they are all 1.
    pub(crate) fn taper(self, len: usize) -> Option<Vec<f32>> {
        let Self::Tukey(alpha) = self else {
            return None;
        };
        assert!(!alpha.is_nan(), "the alpha of the Tukey window is NaN");
        if alpha <= 0.0 || len <= 1 {
            return None;
        }
        // The window rises over the points 0..=rise, and falls over the points that mirror them.
        let last = (len - 1) as f64;
        let alpha = alpha.min(1.0);
        let rise = alpha * last / 2.0;
        let weights = (0..len).map(|i| {
            let from_edge = i.min(len - 1 - i) as f64;
            if from_edge <= rise.floor() {
                (0.5 * (1.0 - (PI * from_edge / rise).cos())) as f32
            } else {
                1.0
            }
        });
        Some(weights.collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tukey_weights_are_those_of_scipy() {
        // scipy.signal.windows.tukey(9, 0.5) and tukey(10, 0.5).
        assert_eq!(
            Window::Tukey(0.5).weights(9),
            [0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0]
        );
        let expected: [f64; 10] = [
            0.0, 0.41317591, 0.96984631, 1.0, 1.0, 1.0, 1.0, 0.96984631, 0.41317591, 0.0,
        ];
        for (&weight, expected) in Window::Tukey(0.5).weights(10).iter().zip(expected) {
            assert!(
                (f64::from(weight) - expected).abs() < 1e-7,
                "{weight} and {expected}"
            );
        }
    }

    #[test]
    fn an_alpha_of_one_gives_the_hann_window() {
        let hann: Vec<f32> = (0..8)
            .map(|i| (0.5 - 0.5 * (2.0 * PI * i as f64 / 7.0).cos()) as f32)
            .collect();
        for (weight, hann) in Window::Tukey(1.0).weights(8).iter().zip(&hann) {
            assert!((weight - hann).abs() < 1e-7, "{weight} and {hann}");
        }
        assert_eq!(Window::Tukey(3.0).weights(8), Window::Tukey(1.0).weights(8));
    }

    #[test]
    fn windows_without_a_taper_weigh_every_point_by_one() {
        assert_eq!(Window::None.weights(4), [1.0; 4]);
        assert_eq!(Window::Tukey(0.0).weights(4), [1.0; 4]);
        assert_eq!(Window::Tukey(-1.0).weights(4), [1.0; 4]);
        assert_eq!(Window::default().weights(1), [1.0]);
        assert_eq!(Window::default().weights(0), [0.0f32; 0]);
    }
}
