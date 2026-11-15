//! The Python module `frc_core._internal`: the crate frc for Python, without NumPy. It takes
//! any array that supports the buffer protocol and returns `memoryview`s.

#![deny(unsafe_code)]

use std::sync::{Mutex, MutexGuard, PoisonError};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod buffer_memory;
mod buffers;

use buffers::{InputBuffer, new_memoryview};

#[pymodule]
#[pyo3(name = "_internal")]
fn internal_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(binom_split, m)?)?;
    m.add_class::<FrcPlanPy>()?;
    m.add_class::<RingsPy>()?;
    m.add_class::<CrossingPy>()?;
    Ok(())
}

/// Returns the seed of a call: the caller's, or a new one from the operating system.
fn call_seed(seed: Option<u64>) -> u64 {
    seed.unwrap_or_else(rand::random)
}

/// Fails unless images of n x n are large enough for an FRC.
fn check_size(n: usize) -> PyResult<()> {
    if n < 2 {
        return Err(PyValueError::new_err(format!(
            "images must be at least 2 x 2, not {n} x {n}"
        )));
    }
    Ok(())
}

/// Returns the Tukey window with this alpha, or no window for an alpha of 0.
fn tukey_window(alpha: f64) -> PyResult<frc::Window> {
    if !(0.0..=1.0).contains(&alpha) {
        return Err(PyValueError::new_err(format!(
            "tukey_alpha is {alpha}, but must be from 0 to 1"
        )));
    }
    Ok(match alpha {
        0.0 => frc::Window::None,
        _ => frc::Window::Tukey(alpha),
    })
}

/// Returns the ring binning that `name` names, "nearest" or "floor".
fn binning_named(name: &str) -> PyResult<frc::RingBinning> {
    match name {
        "nearest" => Ok(frc::RingBinning::Nearest),
        "floor" => Ok(frc::RingBinning::Floor),
        _ => Err(PyValueError::new_err(format!(
            "binning is {name:?}, but must be 'nearest' or 'floor'"
        ))),
    }
}

/// Returns the name of `binning`, as `binning_named` takes it.
fn binning_name(binning: frc::RingBinning) -> &'static str {
    match binning {
        frc::RingBinning::Nearest => "nearest",
        frc::RingBinning::Floor => "floor",
    }
}

/// Returns the threshold that `threshold` names: "1/7", the default, "half_bit", or a float.
fn threshold_of(threshold: Option<&Bound<'_, PyAny>>) -> PyResult<frc::Threshold> {
    let Some(threshold) = threshold else {
        return Ok(frc::Threshold::OneSeventh);
    };
    if let Ok(name) = threshold.extract::<&str>() {
        return match name {
            "1/7" => Ok(frc::Threshold::OneSeventh),
            "half_bit" => Ok(frc::Threshold::HalfBit),
            _ => Err(PyValueError::new_err(format!(
                "threshold is {name:?}, but must be '1/7', 'half_bit' or a float"
            ))),
        };
    }
    Ok(frc::Threshold::Fixed(threshold.extract::<f64>()?))
}

/// Returns `values` as a `memoryview` of one dimension.
fn memoryview_of<'py, T: buffers::OutputElement>(
    py: Python<'py>,
    values: &[T],
) -> PyResult<Bound<'py, PyAny>> {
    new_memoryview::<T>(py, &[values.len()], |out| {
        out.copy_from_slice(values);
        Ok(())
    })
}

/// Splits every count of an array of non-negative integers in two.
///
/// Returns a memoryview of int32 values with the shape of the array. It holds, for each
/// count c, a sample k of binomial(c, 0.5): the number of ones among c random bits. If c is
/// Poisson-distributed with mean m, then k and c - k are independent and Poisson-distributed
/// with mean m / 2.
///
/// The same array and seed give the same samples on every CPU. For a square image they are
/// the samples that FrcPlan.one_image draws with the same seed, so
/// plan.one_image(img, seed, tukey_alpha=0) is plan.two_images(k, img - k). Without a seed,
/// every call draws a new seed from the operating system.
#[pyfunction]
#[pyo3(signature = (a, seed=None))]
fn binom_split<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyAny>> {
    let input = InputBuffer::get(a, "a")?;
    let shape = input.shape().to_vec();
    // The rows of an array of two or more dimensions lie along its last axis.
    let row_len = match shape[..] {
        [.., last] if shape.len() >= 2 => last.max(1),
        _ => 1,
    };
    let seed = call_seed(seed);
    new_memoryview::<i32>(py, &shape, |out| {
        input.write_counts(out)?;
        frc::binom_split(out, row_len, seed, frc::CpuFeatures::Detected)
            .map_err(|_| input.invalid_count_error())
    })
}

/// The rings r < n / 2 of the spectrum of n x n images: their binning and the number of
/// frequencies in each.
///
/// Make one to find where the FRC curve of an n x n image crosses its threshold. Computing the
/// rings needs no Fourier transform: an FrcPlan holds one, and you can also make one on its
/// own. binning says which frequencies f make up ring r. With "nearest", ring r holds the
/// frequencies whose distance |f| from the origin rounds to r. With "floor", it holds those
/// with r ≤ |f| < r + 1, as DIPlib's RadialSum bins them and as earlier versions of frc did.
#[pyclass(name = "Rings", module = "frc_core._internal", frozen)]
struct RingsPy {
    rings: frc::Rings,
}

#[pymethods]
impl RingsPy {
    #[new]
    #[pyo3(signature = (n, binning="nearest"))]
    fn new(n: usize, binning: &str) -> PyResult<Self> {
        check_size(n)?;
        Ok(Self {
            rings: frc::Rings::new(n, binning_named(binning)?),
        })
    }

    /// The side n of the images.
    #[getter]
    fn size(&self) -> usize {
        self.rings.size()
    }

    /// Which frequencies make up each ring: "nearest" or "floor".
    #[getter]
    fn binning(&self) -> &'static str {
        binning_name(self.rings.binning())
    }

    /// Counts the frequencies in each ring r < n / 2.
    ///
    /// Returns a memoryview of uint64 values: the number of frequencies f in ring r.
    fn counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        memoryview_of(py, self.rings.counts())
    }

    /// Finds the point at which a curve first drops below its threshold.
    ///
    /// curve holds the FRC of ring r at index r, as FrcPlan.one_image and two_images return it
    /// for images of this size and binning. The method finds the first ring i >= 1 whose FRC is
    /// below the threshold, and draws a straight line through the FRC minus the threshold at
    /// rings i - 1 and i. It returns the point at which the line is zero as a Crossing, or None
    /// if the curve does not drop below the threshold.
    ///
    /// threshold is "1/7", the default, "half_bit" for the 1/2-bit threshold, which depends on
    /// the number of frequencies in each ring, or a float.
    #[pyo3(signature = (curve, threshold=None))]
    fn first_crossing(
        &self,
        curve: &Bound<'_, PyAny>,
        threshold: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Option<CrossingPy>> {
        let threshold = threshold_of(threshold)?;
        let values = InputBuffer::get(curve, "curve")?.curve()?;
        let (n, rings) = (self.rings.size(), self.rings.counts().len());
        if values.len() > rings {
            return Err(PyValueError::new_err(format!(
                "curve holds {} values, but images of {n} x {n} have {rings} rings",
                values.len()
            )));
        }
        Ok(self
            .rings
            .first_crossing(&values, threshold)
            .map(|crossing| CrossingPy {
                ring: crossing.ring,
                threshold: crossing.threshold,
                resolution: crossing.resolution,
            }))
    }
}

/// The point at which an FRC curve first drops below its threshold.
#[pyclass(name = "Crossing", module = "frc_core._internal", frozen)]
struct CrossingPy {
    /// The position of the crossing in rings: from ring 0 at the origin up, with fractions
    /// between two rings.
    #[pyo3(get)]
    ring: f64,
    /// The threshold at the crossing.
    #[pyo3(get)]
    threshold: f64,
    /// The resolution in pixels: n / ring for n x n images. It is infinite for a curve that
    /// starts below its threshold, which crosses at ring 0.
    #[pyo3(get)]
    resolution: f64,
}

#[pymethods]
impl CrossingPy {
    fn __repr__(&self) -> String {
        format!(
            "Crossing(ring={}, threshold={}, resolution={})",
            self.ring, self.threshold, self.resolution
        )
    }
}

/// The FFT plan and the working memory for FRC curves of n x n images.
///
/// Make one plan for a size and its ring binning, and use it for every image of that size.
/// binning is "nearest" or "floor", as for Rings. A plan computes one curve at a time: threads
/// that share a plan take turns.
#[pyclass(name = "FrcPlan", module = "frc_core._internal", frozen)]
struct FrcPlanPy {
    size: usize,
    rings: Py<RingsPy>,
    plan: Mutex<frc::FrcPlan>,
}

impl FrcPlanPy {
    /// Fails unless `input` is an image of the plan's size.
    fn check_image(&self, input: &InputBuffer) -> PyResult<()> {
        let n = self.size;
        if input.square_size()? != n {
            return Err(PyValueError::new_err(format!(
                "image of shape {:?}, but the plan is for ({n}, {n})",
                input.shape()
            )));
        }
        Ok(())
    }

    /// Returns the plan, once no other thread computes a curve with it. A plan keeps nothing
    /// between two curves, so one that a panic left locked is used as it is.
    fn locked_plan(&self) -> MutexGuard<'_, frc::FrcPlan> {
        self.plan.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[pymethods]
impl FrcPlanPy {
    #[new]
    #[pyo3(signature = (n, binning="nearest"))]
    fn new(py: Python<'_>, n: usize, binning: &str) -> PyResult<Self> {
        check_size(n)?;
        let plan = frc::FrcPlan::new(n, binning_named(binning)?);
        let rings = Py::new(
            py,
            RingsPy {
                rings: plan.rings().clone(),
            },
        )?;
        Ok(Self {
            size: n,
            rings,
            plan: Mutex::new(plan),
        })
    }

    /// The side n of the images.
    #[getter]
    fn size(&self) -> usize {
        self.size
    }

    /// The rings that the plan's curves sum over.
    #[getter]
    fn rings(&self, py: Python<'_>) -> Py<RingsPy> {
        self.rings.clone_ref(py)
    }

    /// Finds the point at which a curve of this plan first drops below its threshold, as
    /// Rings.first_crossing does.
    #[pyo3(signature = (curve, threshold=None))]
    fn first_crossing(
        &self,
        curve: &Bound<'_, PyAny>,
        threshold: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Option<CrossingPy>> {
        self.rings.get().first_crossing(curve, threshold)
    }

    /// Computes the Fourier ring correlation curve of one n x n image of counts.
    ///
    /// Pass the counts as they were measured, without a window. The method splits every count
    /// c of the image into a sample k of binomial(c, 0.5) and c - k, as binom_split does with
    /// the same seed. It multiplies both images that result by a Tukey window, which falls to
    /// zero at the edges, and correlates them, as two_images does. Returns a memoryview of
    /// float64 values, one for each ring r < n / 2.
    ///
    /// tukey_alpha is the fraction of each side that the two tapers of the window take
    /// together, from 0 to 1. With 0 the method applies no window. The window comes after the
    /// split because the split gives two independent images only of Poisson counts, and a
    /// count multiplied by a window is not one.
    ///
    /// The same image, seed and tukey_alpha give the same curve. Without a seed, every call
    /// draws a new seed from the operating system.
    #[pyo3(signature = (img, seed=None, tukey_alpha=0.125))]
    fn one_image<'py>(
        &self,
        py: Python<'py>,
        img: &Bound<'py, PyAny>,
        seed: Option<u64>,
        tukey_alpha: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let input = InputBuffer::get(img, "img")?;
        self.check_image(&input)?;
        let window = tukey_window(tukey_alpha)?;
        let seed = call_seed(seed);
        let curve = py.detach(|| -> PyResult<Vec<f64>> {
            let counts = input.counts()?;
            let mut plan = self.locked_plan();
            plan.one_image(&counts, window, seed)
                .map_err(|_| input.invalid_count_error())
        })?;
        memoryview_of(py, &curve)
    }

    /// Computes the Fourier ring correlation curve of two n x n images.
    ///
    /// Returns a memoryview of float64 values, one for each ring r < n / 2. The value of ring r
    /// is Re(sum F1 conj F2) / sqrt(sum |F1|^2 sum |F2|^2) over the frequencies f of ring r,
    /// where F1 and F2 are the Fourier transforms of the images. The images are read as
    /// float32. The method applies no window: multiply both images by one first.
    fn two_images<'py>(
        &self,
        py: Python<'py>,
        img1: &Bound<'py, PyAny>,
        img2: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (a, b) = (
            InputBuffer::get(img1, "img1")?,
            InputBuffer::get(img2, "img2")?,
        );
        self.check_image(&a)?;
        self.check_image(&b)?;
        let curve = py.detach(|| {
            let (a, b) = (a.floats(), b.floats());
            self.locked_plan().two_images(&a, &b)
        });
        memoryview_of(py, &curve)
    }
}
