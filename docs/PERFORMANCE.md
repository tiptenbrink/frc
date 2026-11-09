# Performance notes

How `one_frc` spends its time, what was tried to make each step faster, and the literature behind the choices. The numbers come from a Ryzen 5 3400G (Zen+, AVX2, no AVX-512) in a VM with 3 cores and SMT, at 4096 x 4096 with mean counts around 100, unless a line says otherwise. Earlier work on an AMD EPYC 9V45 (Zen 5, AVX-512) is in the history of the `bench-rng` branch of [rustfrc](https://github.com/tiptenbrink/rustfrc), where this code lived before it moved here.

## Where the time goes

One `one_frc` at 4096 x 4096 on one thread of the 3400G takes about 200 ms; on 3 threads about 71 ms.

| Step | Time on one thread | Share |
|---|---|---|
| Binomial split | ~70 ms | 35% |
| Row and column FFTs (rustfft) | ~80 ms | 40% |
| Ring sums | ~30 ms | 15% |
| Copies: rows into panels, panels into the column buffer | ~20 ms | 10% |

Memory bandwidth is not the limit. The VM reads 28 GB/s at DDR4-2133 and 43 GB/s at DDR4-3600, and a whole `one_frc` moves about 320 MB; raising the memory speed by 70% made `one_frc` 2 to 5% faster.

## The binomial split

Each count n is split into k ~ binomial(n, 1/2) and n - k by counting the ones among n random bits. That is exact for every n, needs no floating point, and vectorizes: 16 xoshiro256++ generators run side by side, one count per lane, and every lane of a group of 16 draws words until the group's largest count is covered. xoshiro256++ is a scrambled linear generator with 256 bits of state that passes the strong statistical test suites (Blackman and Vigna 2021).

AVX2 has no 64-bit popcount, so the AVX2 path counts the ones of each half byte with a table lookup (`vpshufb`) and adds the bytes with `vpsadbw`, the method of Muła, Kurz and Lemire (2018). Their faster Harley-Seal variant pays off over hundreds of words; each lane here sums two or three words per count, too few for its carry-save adders to help. The byte sums are added up over up to 31 words before one `vpsadbw`.

What was measured on the 3400G, `binom_split` of 2048 x 2048 counts on one thread:

- The portable code, compiled for AVX2 by LLVM: 44.3 ms for the baseline build, 32.0 ms with AVX2. LLVM keeps the state of all 16 lanes in the 16 vector registers and spills on every step. Written by hand, with 8 lanes' state in registers across a block of 256 groups, it takes two thirds of that time (4096 x 4096: 106 → 72 ms). On an Intel Core Ultra the hand-written path is as fast as LLVM's.
- Scalar `popcnt` and `bzhi` instead of the table lookup: 39.5 ms (x86-64-v2: 38.1 ms). Zen+ runs each 256-bit instruction as two halves and has a single vector shift unit, yet the vector path still wins.
- The mask of the last word from compares and a per-group mask instead of a variable shift per word: no faster on 3 threads, slower on one.
- `target-cpu=native` builds: no faster than the baseline build with run-time dispatch.

The method draws more bits than the distribution needs. Any exact sampler needs at least the entropy of the distribution in random bits, and the Knuth and Yao (1976) sampler comes within 2 bits of it; binomial(100, 1/2) has an entropy of about 4.4 bits, against the 100 bits counted here, and words of 64 bits waste the rest of the last word. At a mean count of 100 the split uses 65% of the bits it draws, at 10 only 16%.

Carrying each lane's unused bits over to its next count raises that to about 93% and draws 1.45 times fewer words at a mean of 100 and 5 times fewer at 10, while every count still gets fresh bits. It was implemented and measured, and made the split slower: 71 → 84 ms at a mean of 100 and 43 → 49 ms at a mean of 10 on one thread. At low counts the split is bound by its fixed work per group of 16 counts (loading, widening, summing and storing them), not by drawing words, and the carry adds to that work. Words of 32 bits instead of 64 would waste less of the last word, but in simulation saved nothing over carrying.

Samplers that draw a constant number of uniforms per sample, such as BTPE (Kachitvichyanukul and Schmeiser 1988) and BTRD (Hörmann 1993), are the standard choice for large n, but they branch on every sample, and in floating point they are exact only up to rounding. Farach-Colton and Tsai (2015) give an exact sampler for binomial(n, 1/2) that runs in sublinear time. For counts of many thousands, which some summed images have, such a sampler would draw far fewer bits than counting them; for the tens to hundreds of photons per pixel of most fluorescence images, counting ones in vectors is faster.

Splitting a count into binomial halves to compute an FRC from a single image is the method of Rieger, Droste, Gerritsma, ten Brink and Stallinga (2024): the halves of Poisson counts are independent Poisson images.

## The FFTs

The two images go through one complex FFT, of z = k + i (n - k) or z = a + i b, and their spectra follow from Z(f) and Z(-f). Each point goes through two FFTs of length n, along its row and along its column, at about 2.3 ns per point per pass on one 3400G core. rustfft's AVX planner beats its SSE planner (3.3 ns) and scalar planner (8.7 ns) on Zen+ too, despite the split 256-bit units, so rustfft chooses well by itself.

## The copies and the panel layout

The spectrum is stored in panels of 8 column pairs, so that the column FFTs read memory in order and each panel holds both Z(f) and Z(-f) for its ring terms. The row pass writes each block of rows into every panel with streaming stores, which write whole cache lines without reading them first. The column pass copies a panel into a column buffer that stays in cache, transforms it there, and takes the ring sums from it.

- Copying one value at a time, a load then a store, ran 2.7 times slower (18 against 47 ms per spectrum) in about half the processes on the 3400G. The mode was set when a process started and held for its whole life, whatever memory it got: the same instructions, cache misses, TLB misses and memory refills, only more cycles. It did not depend on the host's load, the CPU clock, the buffer's alignment or physical pages, or the address layout (it stayed with ASLR off), and software prefetching did not remove it. The likely cause is the memory dependence predictor learning to make the loads wait for the stores. Copying 8 rows at a time, loading them first and then storing each column's 8 values at once, takes 8 to 11 ms in every process.
- Panels of 2 or 4 pairs instead of 8, which fit Zen+'s 512 KiB L2 better: no faster.
- The spectrum is allocated as zero pages in 2 MiB pages where Linux allows it (`madvise`), and each page is touched once, in parallel, when a plan is made. Leaving the first touch to the row pass, whose tasks write their rows into every panel, made a plan's first FRC 60 ms slower than later ones in the VM. Touching costs about 10 ms there, and as much as it saves on an Intel Core Ultra. Writing all the zeros instead, as rayon's `par_extend` does, makes a plan 6 ms slower.

## The ring sums

The column pass takes a fixed number of panels per task, sums each task's ring terms in panel order, and adds the tasks' sums in order. With rayon's fold and reduce, which group the panels as threads happen to steal them, the last bits of a curve changed from run to run on several threads; now a curve is the same to the last bit on any number of threads.

Down each column of the half plane, |f| only grows, and ring r holds r <= |f| < r + 1, the bins of DIPlib's `RadialSum`, which frc used before. The ring follows with a comparison per row, without square roots, and the terms of a run of rows in one ring are summed before they are added to the ring.

- A branchless version, which restarts the run with a select and writes it to its ring's slot on every row, was slower: 30 → 41 ms. The branch on a ring change is taken on about 65% of the rows, but predicts well enough, and the extra store per row costs more.
- With p = Z(f) Z(-f) and s = |Z(f)|^2 + |Z(-f)|^2, the three terms are Im p / 2, (s + 2 Re p) / 4 and (s - 2 Re p) / 4, which take fewer operations. When one image is much fainter than the other, its power is then the difference of two nearly equal sums and loses its digits, so the terms are computed from F1 and F2 directly.

## References

- D. Blackman and S. Vigna, "Scrambled linear pseudorandom number generators", ACM Transactions on Mathematical Software 47(4), 2021. [arXiv:1805.01407](https://arxiv.org/abs/1805.01407)
- M. Farach-Colton and M.-T. Tsai, "Exact sublinear binomial sampling", Algorithmica 73, 2015 (ISAAC 2013). [arXiv:1502.02539](https://arxiv.org/abs/1502.02539)
- W. Hörmann, "The generation of binomial random variates", Journal of Statistical Computation and Simulation 46, 1993. [doi:10.1080/00949659308811496](https://doi.org/10.1080/00949659308811496)
- V. Kachitvichyanukul and B. W. Schmeiser, "Binomial random variate generation", Communications of the ACM 31(2), 1988. [doi:10.1145/42372.42381](https://doi.org/10.1145/42372.42381)
- D. E. Knuth and A. C. Yao, "The complexity of nonuniform random number generation", in Algorithms and Complexity: New Directions and Recent Results, Academic Press, 1976.
- W. Muła, N. Kurz and D. Lemire, "Faster population counts using AVX2 instructions", The Computer Journal 61(1), 2018. [arXiv:1611.07612](https://arxiv.org/abs/1611.07612)
- B. Rieger, I. Droste, F. Gerritsma, T. ten Brink and S. Stallinga, "Single image Fourier ring correlation", Optics Express 32(12), 2024. [TU Delft repository](https://pure.tudelft.nl/ws/portalfiles/portal/210600879/oe-32-12-21767.pdf)
- M. van Heel and M. Schatz, "Fourier shell correlation threshold criteria", Journal of Structural Biology 151(3), 2005, the source of the half-bit threshold that frc computes. [doi:10.1016/j.jsb.2005.05.009](https://doi.org/10.1016/j.jsb.2005.05.009)
