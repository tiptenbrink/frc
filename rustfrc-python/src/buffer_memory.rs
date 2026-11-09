//! The module's only unsafe code: the memory behind a Python buffer. `lib.rs` denies unsafe code
//! everywhere else.
#![allow(unsafe_code)]

use pyo3::buffer::PyUntypedBuffer;

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
