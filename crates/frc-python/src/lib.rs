//! The Python module `frc_core._internal`: the crate frc for Python, without NumPy. Its
//! functions take any array that supports the buffer protocol and return `memoryview`s.

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
    m.add_function(wrap_pyfunction!(one_frc, m)?)?;
    m.add_function(wrap_pyfunction!(two_frc, m)?)?;
    m.add_function(wrap_pyfunction!(ring_counts, m)?)?;
    m.add_class::<FrcPlanPy>()?;
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

/// Returns the rings that `name` names, "nearest" or "floor".
fn rings_named(name: &str) -> PyResult<frc::Rings> {
    match name {
        "nearest" => Ok(frc::Rings::Nearest),
        "floor" => Ok(frc::Rings::Floor),
        _ => Err(PyValueError::new_err(format!(
            "rings is {name:?}, but must be 'nearest' or 'floor'"
        ))),
    }
}

/// Returns the name of `rings`, as `rings_named` takes it.
fn rings_name(rings: frc::Rings) -> &'static str {
    match rings {
        frc::Rings::Nearest => "nearest",
        frc::Rings::Floor => "floor",
    }
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
/// the samples that one_frc draws with the same seed, so one_frc(img, seed, tukey_alpha=0) is
/// two_frc(k, img - k). Without a seed, every call draws a new seed from the operating system.
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

/// Computes the Fourier ring correlation curve of one square image of counts.
///
/// Pass the counts as they were measured, without a window. The function splits every count
/// c of the image into a sample k of binomial(c, 0.5) and c - k, as binom_split does with the
/// same seed. It multiplies both images that result by a Tukey window, which falls to zero at
/// the edges, and correlates them, as two_frc does. Returns a memoryview of float64 values,
/// one for each ring r < n / 2 of the n x n image.
///
/// tukey_alpha is the fraction of each side that the two tapers of the window take together,
/// from 0 to 1. With 0 the function applies no window. The window comes after the split
/// because the split gives two independent images only of Poisson counts, and a count
/// multiplied by a window is not one.
///
/// rings says which frequencies f make up ring r. With "nearest", ring r holds those whose
/// distance |f| from the origin rounds to r, and its frequencies lie around r. With "floor", it
/// holds those with r <= |f| < r + 1, as DIPlib's RadialSum and frc 0.1 and 0.2 bin them.
///
/// The same image, seed and tukey_alpha give the same curve. Without a seed, every call draws
/// a new seed from the operating system.
///
/// Each call makes an FrcPlan, which takes time and memory. For several images of one size,
/// make one FrcPlan and call its one_frc.
#[pyfunction]
#[pyo3(signature = (img, seed=None, tukey_alpha=0.125, rings="nearest"))]
fn one_frc<'py>(
    py: Python<'py>,
    img: &Bound<'py, PyAny>,
    seed: Option<u64>,
    tukey_alpha: f64,
    rings: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let n = InputBuffer::get(img, "img")?.square_size()?;
    FrcPlanPy::new(n, rings)?.one_frc(py, img, seed, tukey_alpha)
}

/// Computes the Fourier ring correlation curve of two square images of one size.
///
/// Returns a memoryview of float64 values, one for each ring r < n / 2 of the n x n images.
/// The value of ring r is Re(sum F1 conj F2) / sqrt(sum |F1|^2 sum |F2|^2) over the
/// frequencies f of ring r, where F1 and F2 are the Fourier transforms of the images. rings
/// says which frequencies those are, as for one_frc. The images are read as float32. The
/// function applies no window: multiply both images by one first.
///
/// Each call makes an FrcPlan, which takes time and memory. For several pairs of images of
/// one size, make one FrcPlan and call its two_frc.
#[pyfunction]
#[pyo3(signature = (img1, img2, rings="nearest"))]
fn two_frc<'py>(
    py: Python<'py>,
    img1: &Bound<'py, PyAny>,
    img2: &Bound<'py, PyAny>,
    rings: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let n = InputBuffer::get(img1, "img1")?.square_size()?;
    FrcPlanPy::new(n, rings)?.two_frc(py, img1, img2)
}

/// Counts the frequencies in each ring of an n x n spectrum.
///
/// Returns a memoryview of uint64 values, one for each ring r < n / 2: the number of
/// frequencies f in ring r. rings says which frequencies those are, as for one_frc.
#[pyfunction]
#[pyo3(signature = (n, rings="nearest"))]
fn ring_counts<'py>(py: Python<'py>, n: usize, rings: &str) -> PyResult<Bound<'py, PyAny>> {
    check_size(n)?;
    memoryview_of(py, &frc::ring_counts(n, rings_named(rings)?))
}

/// The FFT plan and the working memory for FRC curves of n x n images.
///
/// Make one plan for a size and its rings, "nearest" or "floor" as for one_frc, and use it for
/// every image of that size. A plan computes one curve at a time: threads that share a plan
/// take turns.
#[pyclass(name = "FrcPlan", module = "frc_core._internal", frozen)]
struct FrcPlanPy {
    size: usize,
    rings: frc::Rings,
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
    #[pyo3(signature = (n, rings="nearest"))]
    fn new(n: usize, rings: &str) -> PyResult<Self> {
        check_size(n)?;
        let rings = rings_named(rings)?;
        Ok(Self {
            size: n,
            rings,
            plan: Mutex::new(frc::FrcPlan::new(n, rings)),
        })
    }

    /// The side n of the images.
    #[getter]
    fn size(&self) -> usize {
        self.size
    }

    /// The rings that the plan's curves sum over: "nearest" or "floor".
    #[getter]
    fn rings(&self) -> &'static str {
        rings_name(self.rings)
    }

    /// Counts the frequencies in each ring r < n / 2, as ring_counts(n, rings) does.
    fn ring_counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ring_counts(py, self.size, rings_name(self.rings))
    }

    /// Computes the FRC curve of an n x n image of counts, as the function one_frc does.
    #[pyo3(signature = (img, seed=None, tukey_alpha=0.125))]
    fn one_frc<'py>(
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
            plan.one_frc(&counts, window, seed)
                .map_err(|_| input.invalid_count_error())
        })?;
        memoryview_of(py, &curve)
    }

    /// Computes the FRC curve of two n x n images, as the function two_frc does.
    fn two_frc<'py>(
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
            self.locked_plan().two_frc(&a, &b)
        });
        memoryview_of(py, &curve)
    }
}
