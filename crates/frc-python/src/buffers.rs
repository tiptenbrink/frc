//! Arrays in and out of Python without NumPy: inputs through the buffer protocol, which NumPy
//! arrays, `array.array`, `memoryview` and most other array types export, and outputs as a
//! `memoryview` over a new `bytearray`, which `numpy.asarray` wraps without a copy.

use std::borrow::Cow;
use std::ffi::CStr;

use pyo3::buffer::PyUntypedBuffer;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyMemoryView, PyTuple};
use rayon::prelude::*;

use crate::buffer_memory::buffer_memory;

/// The type of a buffer's elements, from its struct-module format and item size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Element {
    Bool,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
}

impl Element {
    /// The element type of a buffer with `format` and `item_size`, if it is one of the above in
    /// native byte order. Integer codes go by the item size, since `l` and `L` are 4 bytes on
    /// Windows and 8 elsewhere.
    fn of(format: &CStr, item_size: usize) -> Option<Self> {
        let (order, code) = match format.to_bytes() {
            [code] => (b'@', *code),
            [order, code] => (*order, *code),
            _ => return None,
        };
        let native = match order {
            b'@' | b'=' => true,
            b'<' => cfg!(target_endian = "little"),
            b'>' | b'!' => cfg!(target_endian = "big"),
            _ => false,
        };
        if !native {
            return None;
        }
        let signed = b"bhilqn".contains(&code);
        let unsigned = b"BHILQN".contains(&code);
        Some(match (code, item_size) {
            (b'?', 1) => Self::Bool,
            (b'f', 4) => Self::F32,
            (b'd', 8) => Self::F64,
            (_, 1) if signed => Self::I8,
            (_, 1) if unsigned => Self::U8,
            (_, 2) if signed => Self::I16,
            (_, 2) if unsigned => Self::U16,
            (_, 4) if signed => Self::I32,
            (_, 4) if unsigned => Self::U32,
            (_, 8) if signed => Self::I64,
            (_, 8) if unsigned => Self::U64,
            _ => return None,
        })
    }

    fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }

    /// The element in `bytes`, which are one element of this type, as an integer, for an
    /// integer or bool element type.
    fn read_integer(self, bytes: &[u8]) -> i128 {
        use bytemuck::pod_read_unaligned as read;
        match self {
            Self::Bool => (bytes[0] != 0) as i128,
            Self::I8 => read::<i8>(bytes) as i128,
            Self::U8 => bytes[0] as i128,
            Self::I16 => read::<i16>(bytes) as i128,
            Self::U16 => read::<u16>(bytes) as i128,
            Self::I32 => read::<i32>(bytes) as i128,
            Self::U32 => read::<u32>(bytes) as i128,
            Self::I64 => read::<i64>(bytes) as i128,
            Self::U64 => read::<u64>(bytes) as i128,
            Self::F32 | Self::F64 => unreachable!("not an integer type"),
        }
    }

    /// The element in `bytes`, which are one element of this type, as an f32.
    fn read_f32(self, bytes: &[u8]) -> f32 {
        match self {
            Self::F32 => bytemuck::pod_read_unaligned::<f32>(bytes),
            Self::F64 => bytemuck::pod_read_unaligned::<f64>(bytes) as f32,
            _ => self.read_integer(bytes) as f32,
        }
    }
}

/// An array that a function reads, held through the buffer protocol until it is dropped.
pub struct InputBuffer {
    buffer: PyUntypedBuffer,
    element: Element,
    /// What the array is to the caller, for error messages: "img", "img1", ...
    name: &'static str,
}

/// The largest count that the split takes: counts are i32 inside.
const MAX_COUNT: i128 = i32::MAX as i128;

impl InputBuffer {
    pub fn get(obj: &Bound<'_, PyAny>, name: &'static str) -> PyResult<Self> {
        let buffer = PyUntypedBuffer::get(obj).map_err(|_| {
            PyTypeError::new_err(format!(
                "{name} must support the buffer protocol, as NumPy arrays do, not be a {}",
                obj.get_type()
                    .name()
                    .map_or_else(|_| "?".to_string(), |n| n.to_string())
            ))
        })?;
        if buffer.suboffsets().is_some() {
            return Err(PyTypeError::new_err(format!(
                "{name} is an indirect buffer, with pointers to its rows, which is not supported; \
                 copy it into a NumPy array first"
            )));
        }
        let element = Element::of(buffer.format(), buffer.item_size()).ok_or_else(|| {
            PyTypeError::new_err(format!(
                "{name} has elements of format {:?}; only booleans, integers, float32 and \
                 float64 in native byte order are supported",
                buffer.format()
            ))
        })?;
        Ok(Self {
            buffer,
            element,
            name,
        })
    }

    pub fn shape(&self) -> &[usize] {
        self.buffer.shape()
    }

    /// The shape, which must be n x n with n >= 2.
    pub fn square_size(&self) -> PyResult<usize> {
        match *self.shape() {
            [n, m] if n == m && n >= 2 => Ok(n),
            _ => Err(PyValueError::new_err(format!(
                "{} must be a square 2D image of at least 2 x 2, not one of shape {:?}",
                self.name,
                self.shape()
            ))),
        }
    }

    /// The elements in row-major order, as a slice of T if the buffer holds contiguous Ts at a
    /// T's alignment.
    fn as_slice<T: bytemuck::Pod>(&self, element: Element) -> Option<&[T]> {
        if self.element != element || !self.buffer.is_c_contiguous() {
            return None;
        }
        bytemuck::try_cast_slice(buffer_memory(&self.buffer).0).ok()
    }

    /// Calls `f` with the offset in `buffer_memory` of each element, in row-major order.
    fn for_each_offset(&self, first: usize, mut f: impl FnMut(usize)) {
        let (shape, strides) = (self.buffer.shape(), self.buffer.strides());
        if shape.contains(&0) {
            return;
        }
        let mut index = vec![0usize; shape.len()];
        loop {
            let offset: isize = index
                .iter()
                .zip(strides)
                .map(|(&i, &s)| i as isize * s)
                .sum();
            f((first as isize + offset) as usize);
            let mut axis = shape.len();
            loop {
                if axis == 0 {
                    return;
                }
                axis -= 1;
                index[axis] += 1;
                if index[axis] < shape[axis] {
                    break;
                }
                index[axis] = 0;
            }
        }
    }

    /// Converts every element with `convert`, which gets its bytes, into `out`, which has one
    /// place per element, in row-major order: in parallel for a contiguous buffer.
    fn convert_into<T: Send>(
        &self,
        out: &mut [T],
        convert: impl Fn(&[u8]) -> PyResult<T> + Sync,
    ) -> PyResult<()> {
        let size = self.buffer.item_size();
        let (memory, first) = buffer_memory(&self.buffer);
        if self.buffer.is_c_contiguous() {
            return out
                .par_iter_mut()
                .zip(memory.par_chunks_exact(size))
                .try_for_each(|(o, bytes)| {
                    *o = convert(bytes)?;
                    Ok(())
                });
        }
        let mut out = out.iter_mut();
        let mut result = Ok(());
        self.for_each_offset(first, |at| {
            if result.is_ok() {
                match convert(&memory[at..at + size]) {
                    Ok(v) => *out.next().expect("one place per element") = v,
                    Err(e) => result = Err(e),
                }
            }
        });
        result
    }

    /// The elements as counts for a binomial split, in row-major order: borrowed if the buffer
    /// already holds contiguous int32s. Fails for floats, negative counts and counts above
    /// 2**31 - 1.
    pub fn counts(&self) -> PyResult<Cow<'_, [i32]>> {
        if self.element.is_float() {
            return Err(PyTypeError::new_err(format!(
                "{} must hold integer counts, not floats; round or cast them first",
                self.name
            )));
        }
        let check = |n: i128| -> PyResult<i32> {
            if n < 0 {
                Err(PyValueError::new_err(format!(
                    "{} holds a negative count: {n}",
                    self.name
                )))
            } else if n > MAX_COUNT {
                Err(PyValueError::new_err(format!(
                    "{} holds a count above {MAX_COUNT}: {n}",
                    self.name
                )))
            } else {
                Ok(n as i32)
            }
        };
        if let Some(counts) = self.as_slice::<i32>(Element::I32) {
            if let Some(&n) = counts.iter().find(|&&n| n < 0) {
                return Err(check(n as i128).unwrap_err());
            }
            return Ok(Cow::Borrowed(counts));
        }
        let element = self.element;
        let mut out = vec![0i32; self.buffer.item_count()];
        self.convert_into(&mut out, |bytes| check(element.read_integer(bytes)))?;
        Ok(Cow::Owned(out))
    }

    /// The elements as float32s, in row-major order: borrowed if the buffer already holds
    /// contiguous float32s.
    pub fn floats(&self) -> PyResult<Cow<'_, [f32]>> {
        if let Some(values) = self.as_slice::<f32>(Element::F32) {
            return Ok(Cow::Borrowed(values));
        }
        let element = self.element;
        let mut out = vec![0f32; self.buffer.item_count()];
        self.convert_into(&mut out, |bytes| Ok(element.read_f32(bytes)))?;
        Ok(Cow::Owned(out))
    }
}

/// The struct-module format of an output element type.
pub trait OutputElement: bytemuck::Pod + Default + Send {
    const FORMAT: &'static str;
}

impl OutputElement for i32 {
    const FORMAT: &'static str = "i";
}

impl OutputElement for f64 {
    const FORMAT: &'static str = "d";
}

impl OutputElement for u64 {
    const FORMAT: &'static str = "Q";
}

/// A new array of zero Ts in a `bytearray`, that `fill` fills while the GIL is released,
/// returned as a `memoryview` of `shape`. `memoryview.cast` takes no zero in a shape, so an
/// array with no elements comes back as a 1D `memoryview` of length 0.
pub fn new_memoryview<'py, T: OutputElement>(
    py: Python<'py>,
    shape: &[usize],
    fill: impl FnOnce(&mut [T]) -> PyResult<()> + Send,
) -> PyResult<Bound<'py, PyAny>> {
    let len: usize = shape.iter().product();
    let bytes =
        PyByteArray::new_with(
            py,
            len * size_of::<T>(),
            |raw| match bytemuck::try_cast_slice_mut::<u8, T>(raw) {
                Ok(values) => py.detach(|| fill(values)),
                Err(_) => {
                    let mut values = vec![T::default(); len];
                    py.detach(|| fill(&mut values))?;
                    raw.copy_from_slice(bytemuck::cast_slice(&values));
                    Ok(())
                }
            },
        )?;
    let view = PyMemoryView::from(&bytes)?;
    if len == 0 {
        return view.call_method1("cast", (T::FORMAT,));
    }
    view.call_method1("cast", (T::FORMAT, PyTuple::new(py, shape)?))
}
