//! Arrays in and out of Python without NumPy. Inputs come through the buffer protocol, which
//! NumPy arrays, `array.array`, `memoryview` and most other array types export. Outputs are a
//! `memoryview` over a new array of this module's own, which `numpy.asarray` wraps without a
//! copy.

use std::borrow::Cow;
use std::ffi::CStr;

use pyo3::buffer::PyUntypedBuffer;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyMemoryView;
use rayon::prelude::*;

use crate::buffer_memory::{OutputArray, buffer_memory, zeroed_words};

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

    /// Reads `bytes`, which hold one element of this type, as an integer.
    ///
    /// # Panics
    ///
    /// Panics for a float type.
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

/// Elements that one task of a conversion takes.
const CONVERT_CHUNK: usize = 1 << 14;

/// Returns `element` as a count, or -1 if it is negative or above `MAX_COUNT`.
fn count_of<S: TryInto<i32>>(element: S) -> i32 {
    element.try_into().unwrap_or(-1)
}

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

    /// Writes every element to `out`, which has one place per element, in row-major order:
    /// read as an `S`, the type of the buffer's elements, and converted by `convert`. A
    /// contiguous buffer is converted in parallel.
    fn convert_into<S: bytemuck::Pod, T: Send>(
        &self,
        out: &mut [T],
        convert: impl Fn(S) -> T + Sync,
    ) {
        let size = size_of::<S>();
        assert_eq!(size, self.buffer.item_size(), "elements are not of type S");
        let (memory, first) = buffer_memory(&self.buffer);
        let converted = |bytes: &[u8]| convert(bytemuck::pod_read_unaligned(bytes));
        if self.buffer.is_c_contiguous() {
            out.par_chunks_mut(CONVERT_CHUNK)
                .zip(memory.par_chunks(CONVERT_CHUNK * size))
                .for_each(|(out, bytes)| {
                    for (out, bytes) in out.iter_mut().zip(bytes.chunks_exact(size)) {
                        *out = converted(bytes);
                    }
                });
            return;
        }
        let mut out = out.iter_mut();
        self.for_each_offset(first, |at| {
            *out.next().expect("one place per element") = converted(&memory[at..at + size]);
        });
    }

    /// Writes the elements to `out` as counts for a binomial split, in row-major order. An
    /// element that is not a count, because it is negative or above 2**31 - 1, becomes -1.
    /// The split refuses that, and [`InputBuffer::invalid_count_error`] then says which
    /// element it was.
    ///
    /// Fails for floats.
    pub fn write_counts(&self, out: &mut [i32]) -> PyResult<()> {
        match self.element {
            Element::Bool => self.convert_into(out, |byte: u8| (byte != 0) as i32),
            Element::I8 => self.convert_into(out, count_of::<i8>),
            Element::U8 => self.convert_into(out, count_of::<u8>),
            Element::I16 => self.convert_into(out, count_of::<i16>),
            Element::U16 => self.convert_into(out, count_of::<u16>),
            Element::I32 => self.convert_into(out, count_of::<i32>),
            Element::U32 => self.convert_into(out, count_of::<u32>),
            Element::I64 => self.convert_into(out, count_of::<i64>),
            Element::U64 => self.convert_into(out, count_of::<u64>),
            Element::F32 | Element::F64 => {
                return Err(PyTypeError::new_err(format!(
                    "{} must hold integer counts, not floats; round or cast them first",
                    self.name
                )));
            }
        }
        Ok(())
    }

    /// Returns the elements as counts for a binomial split, in row-major order, as
    /// [`InputBuffer::write_counts`] writes them. Borrows them if the buffer already holds
    /// contiguous int32s.
    ///
    /// Fails for floats.
    pub fn counts(&self) -> PyResult<Cow<'_, [i32]>> {
        if let Some(counts) = self.as_slice::<i32>(Element::I32) {
            return Ok(Cow::Borrowed(counts));
        }
        let mut counts = vec![0i32; self.buffer.item_count()];
        self.write_counts(&mut counts)?;
        Ok(Cow::Owned(counts))
    }

    /// Returns the error of a split that refused this buffer's counts. It names the first
    /// element that is negative or above 2**31 - 1.
    pub fn invalid_count_error(&self) -> PyErr {
        let size = self.buffer.item_size();
        let (memory, first) = buffer_memory(&self.buffer);
        let mut invalid = None;
        self.for_each_offset(first, |at| {
            let n = self.element.read_integer(&memory[at..at + size]);
            if invalid.is_none() && !(0..=MAX_COUNT).contains(&n) {
                invalid = Some(n);
            }
        });
        let name = self.name;
        PyValueError::new_err(match invalid {
            Some(n) if n < 0 => format!("{name} holds a negative count: {n}"),
            Some(n) => format!("{name} holds a count above {MAX_COUNT}: {n}"),
            None => format!("{name} holds a negative count"),
        })
    }

    /// Returns the elements as float32s, in row-major order. Borrows them if the buffer already
    /// holds contiguous float32s.
    pub fn floats(&self) -> Cow<'_, [f32]> {
        if let Some(values) = self.as_slice::<f32>(Element::F32) {
            return Cow::Borrowed(values);
        }
        let mut out = vec![0f32; self.buffer.item_count()];
        let out_ref = &mut out[..];
        match self.element {
            Element::Bool => self.convert_into(out_ref, |byte: u8| (byte != 0) as u8 as f32),
            Element::I8 => self.convert_into(out_ref, |v: i8| v as f32),
            Element::U8 => self.convert_into(out_ref, |v: u8| v as f32),
            Element::I16 => self.convert_into(out_ref, |v: i16| v as f32),
            Element::U16 => self.convert_into(out_ref, |v: u16| v as f32),
            Element::I32 => self.convert_into(out_ref, |v: i32| v as f32),
            Element::U32 => self.convert_into(out_ref, |v: u32| v as f32),
            Element::I64 => self.convert_into(out_ref, |v: i64| v as f32),
            Element::U64 => self.convert_into(out_ref, |v: u64| v as f32),
            Element::F32 => self.convert_into(out_ref, |v: f32| v),
            Element::F64 => self.convert_into(out_ref, |v: f64| v as f32),
        }
        Cow::Owned(out)
    }
}

/// The struct-module format of an output element type.
pub trait OutputElement: bytemuck::Pod + Send + Sync {
    const FORMAT: &'static CStr;
}

impl OutputElement for i32 {
    const FORMAT: &'static CStr = c"i";
}

impl OutputElement for f64 {
    const FORMAT: &'static CStr = c"d";
}

impl OutputElement for u64 {
    const FORMAT: &'static CStr = c"Q";
}

/// A new array of zero Ts, that `fill` fills while the GIL is released, returned as a
/// `memoryview` of `shape`.
pub fn new_memoryview<'py, T: OutputElement>(
    py: Python<'py>,
    shape: &[usize],
    fill: impl FnOnce(&mut [T]) -> PyResult<()> + Send,
) -> PyResult<Bound<'py, PyAny>> {
    let len: usize = shape.iter().product();
    // Words of 8 bytes are aligned for every T, and their bytes hold whole Ts.
    let mut words = zeroed_words((len * size_of::<T>()).div_ceil(size_of::<u64>()));
    let values = &mut bytemuck::cast_slice_mut::<u64, T>(&mut words)[..len];
    py.detach(|| fill(values))?;
    let array = OutputArray::new(words, T::FORMAT, size_of::<T>(), shape);
    Ok(PyMemoryView::from(Bound::new(py, array)?.as_any())?.into_any())
}
