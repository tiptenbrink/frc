//! Which CPU instructions the crate's vector code uses.

/// The instructions that a call may use beyond the baseline of its target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuFeatures {
    /// Only the baseline, whatever the CPU has. On x86-64 that is SSE2.
    Baseline,
    /// What the CPU reports at run time, up to AVX2 and `popcnt`, and never AVX-512.
    UpToAvx2,
    /// Everything the CPU reports at run time that this crate has code for.
    Detected,
}

/// The instructions that a piece of vector code is compiled for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InstructionSet {
    /// The baseline of the target.
    Baseline,
    /// AVX2 and `popcnt`.
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// AVX2, and AVX-512 with a rotate (`vprolq`) and a popcount (`vpopcntq`) of 64-bit lanes:
    /// Ice Lake and later, Zen 4 and later.
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

impl CpuFeatures {
    /// Returns the widest instruction set that these features allow and the CPU has.
    pub(crate) fn instruction_set(self) -> InstructionSet {
        #[cfg(target_arch = "x86_64")]
        {
            let avx2 = self != Self::Baseline
                && is_x86_feature_detected!("popcnt")
                && is_x86_feature_detected!("avx2");
            let avx512 = self == Self::Detected
                && avx2
                && is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512vl")
                && is_x86_feature_detected!("avx512vpopcntdq");
            if avx512 {
                return InstructionSet::Avx512;
            }
            if avx2 {
                return InstructionSet::Avx2;
            }
        }
        InstructionSet::Baseline
    }
}
