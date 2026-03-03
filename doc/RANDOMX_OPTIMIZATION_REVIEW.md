# RandomX Optimization Review (XMRig)

Date: 2026-03-02

## Scope
This review focuses on RandomX CPU hashing in the current codebase. No benchmarking was run. The estimated gains below are **rough, CPU-dependent ranges** based on typical RandomX behavior and similar optimization patterns; they must be validated with real benchmarks on target hardware.

## Key Code Areas Reviewed
- `src/crypto/randomx/` (VMs, JIT compilers, scratchpad access)
- `src/crypto/rx/` (RandomX runtime setup)
- `cmake/flags.cmake`, `cmake/cpu.cmake`, `cmake/asm.cmake`

## Findings & Potential Optimizations

### 1) Add optional native-arch build flags on x86_64
**Where**: `cmake/flags.cmake`, `cmake/cpu.cmake`

**Idea**: Introduce a CMake option (e.g., `-DARCH=native`) on x86_64 similar to RISC-V handling, adding `-march=native -mtune=native` (or fine-grained `-mavx2 -mbmi2 -mfma`) for GCC/Clang builds. This allows the compiler to better optimize non-JIT code paths (dataset init, entropy generation, utility loops) and improves instruction selection around RandomX setup.

**Estimated hashrate gain**: **~1–5%** on supported CPUs (larger on newer CPUs with wider SIMD or BMI2).

**Notes**: Keep it opt-in to preserve portability for generic binaries.

---

### 2) Enable Link-Time Optimization (LTO) for GCC/Clang releases
**Where**: `cmake/flags.cmake`

**Idea**: Add `-flto` to release flags (or a `-DENABLE_LTO=ON` option). LTO can inline cross-translation-unit calls in RandomX setup and bytecode interpretation, improving instruction scheduling in hot paths.

**Estimated hashrate gain**: **~1–3%**.

**Notes**: MSVC already uses `/GL`. LTO increases build time and binary size slightly.

---

### 3) Auto-tune scratchpad prefetch mode per CPU
**Where**: `src/crypto/randomx/randomx.cpp`, `src/crypto/rx/Rx.cpp`

**Idea**: Currently `randomx_set_scratchpad_prefetch_mode()` is set from config and defaults to mode `1`. Add a tiny startup micro-benchmark (first init only) to pick the best mode (0–3) per CPU and memory topology. Prefetching can help or hurt depending on cache size and memory bandwidth.

**Estimated hashrate gain**: **~0–3%** (highly CPU-dependent; can be negative if wrong mode).

**Notes**: Keep the config override to force a mode when desired.

---

### 4) Prefer 1GB huge pages automatically when available
**Where**: `src/backend/cpu/CpuConfig.*`, `src/crypto/rx/Rx.cpp`

**Idea**: Detect `pdpe1gb` support and (optionally) auto-enable `--randomx-1gb-pages` when the OS has enough 1GB huge pages reserved. Reduces TLB misses for the RandomX dataset.

**Estimated hashrate gain**: **~1–5%** (sometimes more on large-core CPUs).

**Notes**: Should remain opt-in or guarded by a safe check, because misconfigured huge pages can cause allocation failures.

---

### 5) Add AVX-512 / Zen4-tuned BLAKE2b path for RandomX seed work
**Where**: `src/crypto/randomx/blake2/`, `src/crypto/rx/Rx.cpp`

**Idea**: RandomX uses BLAKE2b during program generation and cache/dataset initialization. A faster AVX-512 BLAKE2b (or Zen4-optimized) backend could reduce per-hash overhead on capable CPUs.

**Estimated hashrate gain**: **~1–4%** on AVX-512-capable CPUs.

**Notes**: Mostly helps CPUs with strong SIMD; impact is smaller if hashing is dominated by memory latency.

---

### 6) Add JIT scheduling tweaks for AMD vs Intel
**Where**: `src/crypto/randomx/jit_compiler_x86.cpp`

**Idea**: `RANDOMX_FLAG_AMD` is already set for Ryzen/Bulldozer. Review instruction scheduling around memory loads, `IMUL_RCP`, and FPU ops to use AMD-friendly ordering (e.g., avoid certain dependency chains). A small, architecture-specific scheduling pass inside the JIT may improve issue/latency utilization.

**Estimated hashrate gain**: **~1–3%** on AMD CPUs.

**Notes**: This is a more invasive change but still maintains algorithmic correctness.

---

### 7) SIMD-unroll register copy/store loops in interpreted VM
**Where**: `src/crypto/randomx/vm_interpreted.cpp`

**Idea**: The interpreted VM is a fallback path. The per-iteration register load/store loops are scalar and could be unrolled or vectorized (SSE/AVX) to reduce overhead when JIT is disabled or unsupported.

**Estimated hashrate gain**: **~2–8%** **for interpreted mode only** (no change for JIT).

**Notes**: Low priority if all target CPUs use JIT; higher impact on restricted platforms.

---

### 8) Optional build-time profile-guided optimization (PGO)
**Where**: Build system (docs or scripts)

**Idea**: Provide a documented PGO build path for GCC/Clang. RandomX has stable hot paths; PGO can improve branch prediction and layout in the JIT compiler and bytecode machine.

**Estimated hashrate gain**: **~1–4%** depending on CPU and workload.

**Notes**: Requires a training run and extra build steps.

## Summary Table
| Optimization | Estimated Gain | Scope |
|---|---:|---|
| Native arch flags (`-march=native`) | 1–5% | All CPUs (opt-in) |
| LTO (`-flto`) | 1–3% | All CPUs (opt-in) |
| Prefetch auto-tuning | 0–3% | CPU/memory dependent |
| Auto 1GB pages | 1–5% | Linux, huge pages configured |
| AVX-512 BLAKE2b | 1–4% | AVX-512 CPUs only |
| AMD JIT scheduling tweaks | 1–3% | AMD CPUs |
| Interpreted VM SIMD | 2–8% | Interpreted-only |
| PGO build path | 1–4% | All CPUs (opt-in) |

## Next Steps
If you want, I can:
- Draft a minimal CMake option for `ARCH=native` and `ENABLE_LTO`.
- Add a lightweight prefetch auto-tuner guarded by config.
- Provide a benchmark script to validate the estimated gains.
