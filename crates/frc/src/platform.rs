//! The crate's unsafe code, apart from choosing the sampler's code for the CPU in
//! `binomial::sample_chunk`: zeroed memory in 2 MiB pages for the spectrum, and stores that write
//! memory without reading it into the cache first. `lib.rs` denies unsafe code everywhere else.
#![allow(unsafe_code)]

use std::alloc::{Layout, alloc_zeroed, handle_alloc_error};

use rustfft::num_complex::Complex32;

/// `len` zero Complex32s in memory that the system zeroes as each page is first touched, as
/// `vec![0.0; len]` does for f32s; `vec!` writes every byte of other types. On Linux it asks for
/// 2 MiB pages, which Linux gives where transparent huge pages are set to `madvise`, as on most
/// distributions: faulting in 128 MiB then takes 64 faults instead of 32768, and the passes over
/// it miss the TLB less.
pub fn zeroed_complex_vec(len: usize) -> Vec<Complex32> {
    let layout = Layout::array::<Complex32>(len).expect("spectrum too large");
    if layout.size() == 0 {
        return Vec::new();
    }
    // SAFETY: the layout has a nonzero size.
    let memory = unsafe { alloc_zeroed(layout) };
    if memory.is_null() {
        handle_alloc_error(layout);
    }
    #[cfg(target_os = "linux")]
    {
        let (start, end) = (memory as usize, memory as usize + layout.size());
        let first_page = start.next_multiple_of(4096);
        if first_page < end {
            // SAFETY: the range lies within the new allocation, and MADV_HUGEPAGE changes no
            // contents: it only asks for larger pages.
            unsafe {
                libc::madvise(
                    first_page as *mut libc::c_void,
                    end - first_page,
                    libc::MADV_HUGEPAGE,
                );
            }
        }
    }
    // SAFETY: the memory comes from the global allocator with the layout of `len` Complex32s,
    // which a Vec of capacity `len` has; Complex32 is two f32s, and zero bits are 0 + 0i.
    unsafe { Vec::from_raw_parts(memory.cast::<Complex32>(), len, len) }
}

/// Copies `src` to `dst` with streaming stores where `dst` allows them: they write whole cache
/// lines without reading them from memory first, which a normal store does. Call
/// `finish_streaming` before anything reads `dst`.
pub fn copy_streaming(dst: &mut [Complex32], src: &[Complex32]) {
    assert_eq!(dst.len(), src.len());
    #[cfg(target_arch = "x86_64")]
    if (dst.as_ptr() as usize).is_multiple_of(16) && dst.len().is_multiple_of(2) {
        use std::arch::x86_64::{_mm_loadu_ps, _mm_stream_ps};
        let (d, s) = (dst.as_mut_ptr() as *mut f32, src.as_ptr() as *const f32);
        for i in (0..2 * dst.len()).step_by(4) {
            // SAFETY: i + 4 <= 2 len for both slices, which hold 2 len f32s each; d + i is
            // 16-byte aligned; and SSE is part of x86-64.
            unsafe { _mm_stream_ps(d.add(i), _mm_loadu_ps(s.add(i))) };
        }
        return;
    }
    dst.copy_from_slice(src)
}

/// Orders the streaming stores of this thread before whatever it writes or reads next, so that
/// the threads that read the memory after a rayon join see them.
pub fn finish_streaming() {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: SSE is part of x86-64.
    unsafe {
        std::arch::x86_64::_mm_sfence()
    };
}
