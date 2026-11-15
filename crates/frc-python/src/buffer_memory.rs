//! The module's only unsafe code: the memory behind a Python buffer, and the array that the
//! module's functions return. `lib.rs` denies unsafe code everywhere else.
#![allow(unsafe_code)]

use std::ffi::{CStr, c_int};
use std::ptr;

use pyo3::buffer::PyUntypedBuffer;
use pyo3::exceptions::PyBufferError;
use pyo3::ffi;
use pyo3::prelude::*;

/// The size from which new memory asks for huge pages, which is NumPy's for its arrays.
#[cfg(target_os = "linux")]
const HUGE_PAGE_MIN_BYTES: usize = 1 << 22;

/// Returns `len` zero words for an [`OutputArray`], allocated as `numpy.zeros` allocates.
///
/// `vec![0; len]` takes its memory from `calloc`, which passes on new pages of the kernel as
/// they are, since they hold zeros already. No pass writes the zeros, and the kernel maps each
/// page when the caller first writes to it, on the caller's threads. On Linux, this function
/// also asks for huge pages, as NumPy does. On a Core Ultra 7 155H with 6 threads,
/// `binom_split` of 4096 x 4096 counts took 15 ms through Python with them, and 25 to 33 ms
/// with pages of 4 KiB.
pub fn zeroed_words(len: usize) -> Vec<u64> {
    #[cfg_attr(not(target_os = "linux"), allow(unused_mut))]
    let mut words = vec![0u64; len];
    #[cfg(target_os = "linux")]
    if size_of_val(&words[..]) >= HUGE_PAGE_MIN_BYTES {
        let start = words.as_mut_ptr().cast::<u8>();
        let skipped = start.align_offset(4096);
        // SAFETY: the range lies in the memory of `words`, and the advice changes no contents.
        // The kernel refuses a start that is not on a page of its own size, which leaves the
        // memory as it was.
        unsafe {
            libc::madvise(
                start.add(skipped).cast(),
                size_of_val(&words[..]) - skipped,
                libc::MADV_HUGEPAGE,
            )
        };
    }
    words
}

/// An array of this module's own, which Python reads and writes through the buffer protocol.
/// `memoryview` and `numpy.asarray` wrap it without a copy.
#[pyclass(frozen, module = "frc_core._internal", name = "Array")]
pub struct OutputArray {
    /// Owns the memory at `start`. Rust neither reads nor writes it once the array exists.
    _words: Vec<u64>,
    start: *mut u64,
    format: &'static CStr,
    item_size: isize,
    shape: Box<[isize]>,
    strides: Box<[isize]>,
}

// SAFETY: `start` points to the memory of `_words`, which lives and stays in place as long as
// the array. Only the holders of a buffer use the pointer, and the buffer protocol leaves it to
// them not to race, as it does for a `bytearray`.
unsafe impl Send for OutputArray {}
// SAFETY: as for Send.
unsafe impl Sync for OutputArray {}

impl OutputArray {
    /// An array of `shape` in row-major order. Its elements, of `item_size` bytes each and of
    /// the struct-module `format`, are the first bytes of `words`.
    pub fn new(
        mut words: Vec<u64>,
        format: &'static CStr,
        item_size: usize,
        shape: &[usize],
    ) -> Self {
        let len: usize = shape.iter().product();
        assert!(
            len * item_size <= size_of_val(&words[..]),
            "words hold too few elements"
        );
        let mut strides = vec![0isize; shape.len()];
        let mut stride = item_size as isize;
        for (axis_stride, &n) in strides.iter_mut().zip(shape).rev() {
            *axis_stride = stride;
            stride *= n.max(1) as isize;
        }
        Self {
            start: words.as_mut_ptr(),
            _words: words,
            format,
            item_size: item_size as isize,
            shape: shape.iter().map(|&n| n as isize).collect(),
            strides: strides.into(),
        }
    }
}

#[pymethods]
impl OutputArray {
    /// Fills `view` with the array, for any request: it is writable and C-contiguous.
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("the view to fill is null"));
        }
        let array = slf.get();
        // A consumer that did not ask for a field must find NULL in it.
        let if_requested = |flag: c_int, pointer: *mut isize| {
            if flags & flag == flag {
                pointer
            } else {
                ptr::null_mut()
            }
        };
        let format = match flags & ffi::PyBUF_FORMAT {
            0 => ptr::null_mut(),
            _ => array.format.as_ptr().cast_mut(),
        };
        let shape = if_requested(ffi::PyBUF_ND, array.shape.as_ptr().cast_mut());
        let strides = if_requested(ffi::PyBUF_STRIDES, array.strides.as_ptr().cast_mut());
        let (buf, item_size) = (array.start, array.item_size);
        let (len, ndim) = (array.shape.iter().product::<isize>(), array.shape.len());
        // SAFETY: Python passes a view to fill in. Every pointer in it points into the array,
        // which the view keeps alive through `obj`, and which never moves or frees what they
        // point to. Consumers do not write to the format, the shape or the strides.
        unsafe {
            (*view).buf = buf.cast();
            (*view).len = len * item_size;
            (*view).itemsize = item_size;
            (*view).readonly = 0;
            (*view).ndim = ndim as c_int;
            (*view).format = format;
            (*view).shape = shape;
            (*view).strides = strides;
            (*view).suboffsets = ptr::null_mut();
            (*view).internal = ptr::null_mut();
            (*view).obj = slf.into_ptr();
        }
        Ok(())
    }
}

/// The bytes that hold the elements of `buffer`, and the offset in them of element (0, ..., 0).
/// Elements along axes with negative strides lie before that offset. Indirect buffers, with
/// suboffsets, must be refused before.
pub fn buffer_memory(buffer: &PyUntypedBuffer) -> (&[u8], usize) {
    assert!(buffer.suboffsets().is_none(), "indirect buffer");
    let (shape, strides) = (buffer.shape(), buffer.strides());
    if shape.contains(&0) {
        return (&[], 0);
    }
    let (mut low, mut high) = (0isize, 0isize);
    for (&n, &stride) in shape.iter().zip(strides) {
        let reach = (n as isize - 1) * stride;
        if reach < 0 {
            low += reach;
        } else {
            high += reach;
        }
    }
    let len = (high - low) as usize + buffer.item_size();
    // SAFETY: the buffer protocol puts element (i0, i1, ...) at buf + i0 s0 + i1 s1 + ..., in
    // memory that the exporter keeps valid while `buffer` is held, which the returned slice
    // cannot outlive. From the first byte of the lowest element to the last byte of the highest,
    // that is low .. high + item_size from buf: one array of the exporter, of which a strided view
    // skips some bytes.
    let memory =
        unsafe { std::slice::from_raw_parts(buffer.buf_ptr().cast::<u8>().offset(low), len) };
    (memory, -low as usize)
}
