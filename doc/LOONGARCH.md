# XMRig LoongArch port notes

This is the record of moving XMRig 6.26.0 (upstream commit `b2ca724`) to LoongArch: which files were touched and why, how the build is wired, what has been verified, and what is still missing. One convention holds throughout: **LSX** (LoongArch SX, 128 bit vectors) and **LASX** (LoongArch ASX, 256 bit vectors) are two independent instruction sets, the second does not include the first, so the two layers are probed separately, built separately, and picked between at run time.

## 1. The machine the port targets

| item | value |
| --- | --- |
| processor | Loongson-3A6000, 4 cores 8 threads, 1 MiB L2 per core, 16 MiB L3 |
| CPU flags | `lam lam_bh scq ual fpu lsx lasx crc32 complex crypto lspw lvz lbt_x86 lbt_arm lbt_mips` |
| base page size | 16 KiB |
| compiler | GCC 15.3.0 |

One compiler fact matters a lot here: **there are no `__LSX__` or `__LASX__` macros**, only `__loongarch_sx` and `__loongarch_asx`, plus `__loongarch_simd_width`. Upstream XMRig decides whether to build the vector layers from `__SSE2__` and `__AVX2__`, and on LoongArch neither of those ever holds, so everything fell through to the portable fallback. That is where the work started.

```
upstream test        what it should mean here
__SSE2__         =>  LSX
__AVX2__         =>  LASX
```

## 2. The two vector layers

```
configure time: two probe programs compiled in place
        |
        v
  which layer compiles?
  |--  -mlsx           -->  LSX shadow header, sse2lsx.h      -->  BLAKE2b and the CryptoNight family on LSX
  `--  -mlsx -mlasx    -->  LASX shadow headers, blake2b-common.h  -->  BLAKE2b on LASX
        |
        v
run time: pick a layer by the CPU flags
```

The point of the layering: on a machine with only LSX, the LASX layer is not built at all, not merely left unused, so there is no way to end up executing an unsupported instruction.

## 3. What was touched

### Modified files (17)

| file | what changed |
| --- | --- |
| `CMakeLists.txt` | recognize `loongarch64`, wire up the probes |
| `cmake/cpu.cmake` | the LoongArch branch: two probe programs compiled in place, producing `LOONGARCH_CXX_FLAGS` |
| `cmake/flags.cmake` | the `XMRIG_FEATURE_LSX` / `XMRIG_FEATURE_LASX` switches take their values from the probes |
| `cmake/randomx.cmake` | build the two BLAKE2b units, each with its own flags |
| `cmake/asm.cmake` | no x86 assembly on LoongArch, turn it off |
| `src/backend/cpu/cpu.cmake` | use the LoongArch CPU identification units |
| `src/backend/cpu/interfaces/ICpuInfo.h` | add `FLAG_LSX` / `FLAG_LASX`, fix a misplaced bracket in the RISC-V test |
| `src/backend/cpu/platform/BasicCpuInfo.cpp` | widen the flag table, accept `lsx` / `lasx` |
| `src/backend/cpu/platform/BasicCpuInfo.h` | declare the two LoongArch entry points |
| `src/base/kernel/Entry.cpp` | recognize LoongArch at platform startup |
| `src/base/kernel/Platform_unix.cpp` | same, for the architecture name |
| `src/crypto/randomx/blake2/blake2.h` | let `blake2b` pick between the two implementations |
| `src/crypto/rx/Rx.cpp` | pick one of the two `rx_blake2b_compress` units at run time |
| `src/crypto/cn/CryptoNight_x86.h` | include the LSX shadow header |
| `src/crypto/cn/soft_aes.h` | software AES only on LoongArch |
| `src/crypto/common/portable/mm_malloc.h` | `_mm_malloc` and friends fall through to `posix_memalign` |
| `src/version.h` | the version banner prints `LoongArch` |

### Files added by the first adaptation step (6)

| file | purpose |
| --- | --- |
| `src/crypto/common/simd/sse2lsx.h` | the SSE2 to LSX shadow header, the algorithm code above it is untouched |
| `src/crypto/randomx/blake2/blake2b_lsx.c` | BLAKE2b on LSX, `-Ofast -mlsx` |
| `src/crypto/randomx/blake2/lasx/` | BLAKE2b on LASX: `blake2.h`, `blake2b.h`, `blake2b-common.h`, `blake2b-load-lasx.h`, `blake2b_lasx.c` |
| `src/backend/cpu/platform/BasicCpuInfo_loongarch.cpp` | read the machine facts from `/proc/cpuinfo` and `cpucfg` |
| `src/backend/cpu/platform/lscpu_loongarch.cpp` | list the CPU fields |
| `src/crypto/randomx/tests/loongarch_lsx.c`, `loongarch_lasx.c` | the configure time probes, a layer is enabled only when its probe runs |

### Files added later: the RandomX JIT and the fast AES (8 more)

| file | purpose |
| --- | --- |
| `src/crypto/randomx/jit_compiler_la64.cpp/.hpp` | the LA64 JIT: engine table, program and light program generators, the superscalar dataset item function |
| `src/crypto/randomx/jit_compiler_la64_static.S/.hpp` | the executable template: prologue, main loop, the full and light tails, the soft AES helper |
| `src/crypto/randomx/jit_compiler_la64_lasx.cpp/.hpp` | the four lane LASX dataset item generator, used by dataset init |
| `src/crypto/randomx/jit_compiler_la64_lasx_static.S/.hpp` | its executable template |
| plus edits in `intrin_portable.h` (the interpreter's LSX layer), `randomx.cpp`, `aes_hash.cpp/.hpp`, `vm_compiled*.cpp/.hpp`, `vm_interpreted*.cpp/.hpp`, `virtual_machine.cpp` | the LA64 JIT hookup, the fast software AES round selection, the template argument plumbing |

## 4. How the shadow headers are built

There is one rule: **rename and reimplement the `_mm_*` and `_mm256_*` intrinsics, never touch the algorithm itself**. Upstream `blake2b.c` and `CryptoNight_x86.h` stay line for line, which also keeps the port easy to follow upstream.

Some traps that were actually hit:

| written in the shadow header | what upstream expects | why it is written this way |
| --- | --- | --- |
| `_mm_alignr_epi8(a, b, imm)` | byte level concatenate on x86 | LSX has no direct counterpart, it takes `vbsrl_v` and `vbsll_v` with the immediate baked in |
| `_mm_cvtsi128_si32/si64` | extract the first word of a vector | no LSX equivalent, done with `memcpy` |
| `_mm256_blend_epi32(a, b, imm)` | pick lanes by immediate | LASX needs the immediate spread into a mask first, then `xvbitsel_v` |
| `_mm256_broadcastsi128_si256` | widen 128 bit to 256 bit | no LASX counterpart, done by hand |
| `_mm256_zeroupper()` | clear the upper vector halves on x86 | nothing of the sort on LSX/LASX, left empty |
| `_mm_aesenc_si128` and friends | hardware AES on x86 | no such table entry on LoongArch, placeholder only, never called |

One rule has to hold in the shadow headers: **immediates must be literal, never variables**. The LSX/LASX shift style instructions take immediates, and wrapping them as functions gets rejected by the compiler, so those spots are macros.

## 5. How the algorithm switches are set

| family | setting | reason |
| --- | --- | --- |
| the CryptoNight family | on LSX, built only when both layers compile | without the layer there are undefined symbols |
| BLAKE2b | one unit per layer, picked at run time | a LSX only machine must never see LASX instructions |
| GhostRider | off entirely | that family still tests the x86 way |
| MSR / Ryzen | off entirely | x86 machine state registers |

The run time choice happens once, in the initialization step of `Rx.cpp`; the two `rx_blake2b_compress` units are the hand written LSX and LASX versions.

## 6. How to build

```bash
cmake -S . -B build -G Ninja
ninja -C build xmrig
```

After configure, look for these three lines:

```
-- Detected LoongArch architecture (loongarch64)
-- LoongArch LSX: ON LASX: ON (-mlsx -mlasx)
-- LoongArch: GhostRider family is not built
```

The second line is the two vector switches. For a LSX only build, pass `-DWITH_LASX=OFF` and it reads `LSX: ON LASX: OFF`: the probe compiles in place, whatever fails is turned off, no uncompilable objects are left behind.

## 7. What has been verified

The early, still standing checks from the first adaptation step:

| what was verified | result |
| --- | --- |
| BLAKE2b: LSX and LASX against the integer implementation | 180 comparisons, all equal |
| the LASX shadow header | 11 comparisons, all equal |
| RandomX end to end, same machine code as the real binary | 45 comparisons, all equal |
| CPU identification and the two switches | configure log and version output |

The verification method is "hang the harness on the real build": the real link command is taken from `ninja`, the harness object is swapped in, and what runs is the same machine code as the miner. Conclusions reached this way weigh as much as a built in benchmark.

The later RandomX JIT work raised the bar further:

| what was verified | result |
| --- | --- |
| five build trees, one offline harness | 132 digests, all five trees agree |
| the JIT dataset item function | every item byte for byte equal to the interpreter |
| compiled programs against the byte code machine | all digest tests agree |
| the offline 1M benchmark | hash sum `898B6E0431C28A6B`, stable across every build and every optimization round |
| pool mining | shares accepted, on the order of hours |
| the CryptoNight family against the reference vectors of `CryptoNight_test.h` | 105 of 105 inputs: five 76 byte blocks per variant for nineteen variants, ten height carrying inputs for cn/r |
| mining speed, idle 3A6000 | about 900 H/s on 4 threads, about 1012 H/s on 8 |
| dataset init, 8 threads | 2.8 s with the LASX item function, 6.6 s with the 128 bit one |

## 8. Known gaps

One: **the huge page accounting is hardcoded to 2 MiB and 1 GiB**, while these machines only carry a 32 MiB page pool:

```
2336 MiB /  2 MiB = 1168 pages, of which the miner finds 0
2336 MiB / 32 MiB =   73 pages, which is what the kernel could actually give
```

The dataset itself sits on transparent huge pages the whole time, so the `0/1168` line is an accounting artifact, but the accounting is still wrong and worth fixing.

## 9. What could come next

```
huge pages: ask by the machine's own page size
        |
        v
upstream sync discipline
```

The first is small with direct payoff. Beyond that, the big pieces of the port (the JIT, the interpreter's vector layer, the fast AES round, the CryptoNight verification) are done, and the remaining performance work is emitter level tuning, where the wall is the issue width of the core rather than anything a port can change.
