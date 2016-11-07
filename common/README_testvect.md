Measurements
------------

Results normalized per clock; on multi-core systems the average from all cores is taken; SMT taken into account on SMT-capable CPUs.
Formula used for the normalisation:

	flops_per_matrix * matrix_count * threads_per_core / (CPU_freq * duration)

| CPU                       | N-way SIMD ALUs  | flops/clock/core | remarks                                            |
| ------------------------- | ---------------- | ---------------- | -------------------------------------------------- |
| IBM PowerPC 750CL         | 2-way            | 1.51             | g++ 4.6, paired-singles via generic vectors        |
| AMD Bobcat                | 2-way            | 1.38             | clang++ 3.5, SSE2 via intrinsics, native tune [^1] |
| AMD Bobcat                | 2-way            | 1.42             | clang++ 3.5, SSE2 via intrinsics, generic tune     |
| Intel Sandy Bridge        | 8-way            | 9.04             | clang++ 3.6, AVX256 via generic vectors            |
| Intel Ivy Bridge (no SMT) | 8-way            | 8.82             | clang++ 3.6, AVX256 via generic vectors            |
| Intel Ivy Bridge          | 8-way            | 9.09             | clang++ 3.6, AVX256 via generic vectors            |
| Intel Haswell             | 8-way            | 9.56             | clang++ 3.6, AVX256 + FMA3 via generic vectors     |
| Intel Xeon Phi / KNC      | 16-way           | 6.62             | icpc 14.0.4, MIC via intrinsics                    |
| iMX53 Cortex-A8           | 2-way            | 2.23             | clang++ 3.5, NEON via inline asm                   |
| RK3368 Cortex-A53         | 2-way            | 2.40             | clang++ 3.5, A32 NEON via inline asm               |
| RK3368 Cortex-A53         | 2-way            | 2.48             | clang++ 3.6, A64 NEON via intrinsics               |
| MT8163A Cortex-A53        | 2-way            | 2.38             | clang++ 3.6, A32 NEON via inline asm               |
| MT8163A Cortex-A53        | 2-way            | 2.46             | clang++ 3.6, A64 NEON via intrinsics               |
| AppliedMicro X-Gene 1     | 2-way            | 2.71             | clang++ 3.5, A64 NEON via intrinsics               |
| Apple A7                  | 4-way            | 11.07            | apple clang++ 7.0.0, A64 NEON via intrinsics       |
| Apple A8                  | 4-way            | 12.19            | apple clang++ 7.0.0, A64 NEON via intrinsics       |
| Apple A9                  | 4-way            | 16.79            | apple clang++ 7.0.0, A64 NEON via intrinsics       |

[^1]: This is not a mistake - clang++-3.5 -mtune=native produces worse scheduling for btver1 on this code than no tune.
