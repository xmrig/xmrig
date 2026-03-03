# Building with Native CPU Optimizations (ARCH=native)

For maximum performance on x86_64 CPUs, you can enable native CPU optimizations during the build. This tells the compiler to use all instruction sets supported by your specific CPU.

## Quick Start

```bash
mkdir build && cd build
cmake .. -DARCH=native
make -j$(nproc)
```

## What This Does

The `ARCH=native` option adds `-march=native -mtune=native` compiler flags, which enable:

- **AVX2** (256-bit SIMD) on Intel Haswell (2013+) and AMD Zen 2 (2019+)
- **BMI2** (bit manipulation) on Intel Haswell (2013+) and AMD Zen 3 (2020+)
- **FMA** (fused multiply-add) on Intel Haswell (2013+) and AMD Zen (2017+)
- **AVX-512** on Intel Ice Lake (2019+) and AMD Zen 4 (2022+)
- CPU-specific scheduling optimizations

## Expected Performance Gain

**~1-5% higher hashrate** on modern CPUs compared to generic builds.

Larger gains typically on:
- Intel Ice Lake / Alder Lake / Raptor Lake
- AMD Zen 3 / Zen 4

## Important Notes

⚠️ **The resulting binary will ONLY work on CPUs with the same or better instruction set.**

If you build on a Ryzen 5950X, the binary will crash on older CPUs without AVX2/BMI2.

**Use native builds when:**
- Building for a specific mining rig
- Building on the same machine where you'll run the miner

**Do NOT use native builds when:**
- Distributing binaries to other users
- Building packages for Linux distributions
- Unsure about the target CPU

## Example

```bash
# On AMD Ryzen 5950X (Zen 3)
cmake .. -DARCH=native -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# This enables: AVX2, BMI2, FMA, AES-NI, and Zen3-specific optimizations
```

## Verification

After building, you can verify the enabled features:

```bash
# Check what instructions are enabled
readelf -p .comment ./xmrig | grep march

# Or run the miner - it will show CPU features in the startup log
./xmrig --version
```

## See Also

- [CMAKE_OPTIONS.md](CMAKE_OPTIONS.md) - Full list of CMake options
- [RANDOMX_OPTIMIZATION_REVIEW.md](../RANDOMX_OPTIMIZATION_REVIEW.md) - Other optimization ideas
