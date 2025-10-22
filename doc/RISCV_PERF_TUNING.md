# RISC-V Performance Optimization Guide

This guide provides comprehensive instructions for optimizing XMRig on RISC-V architectures.

## Build Optimizations

### Compiler Flags Applied Automatically

The CMake build now applies aggressive RISC-V-specific optimizations:

```cmake
# RISC-V ISA with extensions
-march=rv64gcv_zba_zbb_zbc_zbs

# Aggressive compiler optimizations
-funroll-loops              # Unroll loops for ILP (instruction-level parallelism)
-fomit-frame-pointer        # Free up frame pointer register (RISC-V has limited registers)
-fno-common                 # Better code generation for global variables
-finline-functions          # Inline more functions for better cache locality
-ffast-math                 # Relaxed FP semantics (safe for mining)
-flto                       # Link-time optimization for cross-module inlining

# Release build additions
-minline-atomics            # Inline atomic operations for faster synchronization
```

### Optimal Build Command

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

**Expected build time**: 5-15 minutes depending on CPU

## Runtime Optimizations

### 1. Memory Configuration (Most Important)

Enable huge pages to reduce TLB misses and fragmentation:

#### Enable 2MB Huge Pages
```bash
# Calculate required huge pages (1 page = 2MB)
# For 2 GB dataset: 1024 pages
# For cache + dataset: 1536 pages minimum
sudo sysctl -w vm.nr_hugepages=2048
```

Verify:
```bash
grep HugePages /proc/meminfo
# Expected: HugePages_Free should be close to nr_hugepages
```

#### Enable 1GB Huge Pages (Optional but Recommended)

```bash
# Run provided helper script
sudo ./scripts/enable_1gb_pages.sh

# Verify 1GB pages are available
cat /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages
# Should be: >= 1 (one 1GB page)
```

Update config.json:
```json
{
    "cpu": {
        "huge-pages": true
    },
    "randomx": {
        "1gb-pages": true
    }
}
```

### 2. RandomX Mode Selection

| Mode | Memory | Init Time | Throughput | Recommendation |
|------|--------|-----------|-----------|-----------------|
| **light** | 256 MB | 10 sec | Low | Testing, resource-constrained |
| **fast** | 2 GB | 2-5 min* | High | Production (with huge pages) |
| **auto** | 2 GB | Varies | High | Default (uses fast if possible) |

*With optimizations; can be 30+ minutes without huge pages

**For RISC-V, use fast mode with huge pages enabled.**

### 3. Dataset Initialization Threads

Optimal thread count = 60-75% of CPU cores (leaves headroom for OS/other tasks)

```json
{
    "randomx": {
        "init": 4
    }
}
```

Or auto-detect (rewritten for RISC-V):
```json
{
    "randomx": {
        "init": -1
    }
}
```

### 4. CPU Affinity (Optional)

Pin threads to specific cores for better cache locality:

```json
{
    "cpu": {
        "rx/0": [
            { "threads": 1, "affinity": 0 },
            { "threads": 1, "affinity": 1 },
            { "threads": 1, "affinity": 2 },
            { "threads": 1, "affinity": 3 }
        ]
    }
}
```

### 5. CPU Governor (Linux)

Set to performance mode for maximum throughput:

```bash
# Check current governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# Set to performance (requires root)
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Verify
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
# Should output: performance
```

## Configuration Examples

### Minimum (Testing)
```json
{
    "randomx": {
        "mode": "light"
    },
    "cpu": {
        "huge-pages": false
    }
}
```

### Recommended (Balanced)
```json
{
    "randomx": {
        "mode": "auto",
        "init": 4,
        "1gb-pages": true
    },
    "cpu": {
        "huge-pages": true,
        "priority": 2
    }
}
```

### Maximum Performance (Production)
```json
{
    "randomx": {
        "mode": "fast",
        "init": -1,
        "1gb-pages": true,
        "scratchpad_prefetch_mode": 1
    },
    "cpu": {
        "huge-pages": true,
        "priority": 3,
        "yield": false
    }
}
```

## CLI Equivalents

```bash
# Light mode
./xmrig --randomx-mode=light

# Fast mode with 4 init threads
./xmrig --randomx-mode=fast --randomx-init=4

# Benchmark
./xmrig --bench=1M --algo=rx/0

# Benchmark Wownero variant (1 MB scratchpad)
./xmrig --bench=1M --algo=rx/wow

# Mine to pool
./xmrig -o pool.example.com:3333 -u YOUR_WALLET -p x
```

## Performance Diagnostics

### Check if Vector Extensions are Detected

Look for `FEATURES:` line in output:
```
 * CPU:       ky,x60 (uarch ky,x1)
 * FEATURES:  rv64imafdcv zba zbb zbc zbs
```

- `v`: Vector extension (RVV) ✓
- `zba`, `zbb`, `zbc`, `zbs`: Bit manipulation ✓
- If missing, make sure build used `-march=rv64gcv_zba_zbb_zbc_zbs`

### Verify Huge Pages at Runtime

```bash
# Run xmrig with --bench=1M and check output
./xmrig --bench=1M

# Look for line like:
# HUGE PAGES   100%  1 / 1 (1024 MB)
```

- Should show 100% for dataset AND threads
- If less, increase `vm.nr_hugepages` and reboot

### Monitor Performance

```bash
# Run benchmark multiple times to find stable hashrate
./xmrig --bench=1M --algo=rx/0
./xmrig --bench=10M --algo=rx/0
./xmrig --bench=100M --algo=rx/0

# Check system load and memory during mining
while true; do free -h; grep HugePages /proc/meminfo; sleep 2; done
```

## Expected Performance

### Hardware: Orange Pi RV2 (Ky X1, 8 cores @ ~1.5 GHz)

| Config | Mode | Hashrate | Init Time |
|--------|------|----------|-----------|
| Scalar (baseline) | fast | 30 H/s | 10 min |
| Scalar + huge pages | fast | 33 H/s | 2 min |
| RVV (if enabled) | fast | 70-100 H/s | 3 min |

*Actual results depend on CPU frequency, memory speed, and load*

## Troubleshooting

### Long Initialization Times (30+ minutes)

**Cause**: Huge pages not enabled, system using swap
**Solution**:
1. Enable huge pages: `sudo sysctl -w vm.nr_hugepages=2048`
2. Reboot: `sudo reboot`
3. Reduce mining threads to free memory
4. Check available memory: `free -h`

### Low Hashrate (50% of expected)

**Cause**: CPU governor set to power-save, no huge pages, high contention
**Solution**:
1. Set governor to performance: `echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`
2. Enable huge pages
3. Reduce number of mining threads
4. Check system load: `top` or `htop`

### Dataset Init Crashes or Hangs

**Cause**: Insufficient memory, corrupted huge pages
**Solution**:
1. Disable huge pages temporarily: set `huge-pages: false` in config
2. Reduce mining threads
3. Reboot and re-enable huge pages
4. Try light mode: `--randomx-mode=light`

### Out of Memory During Benchmark

**Cause**: Not enough RAM for dataset + cache + threads
**Solution**:
1. Use light mode: `--randomx-mode=light`
2. Reduce mining threads: `--threads=1`
3. Increase available memory (kill other processes)
4. Check: `free -h` before mining

## Advanced Tuning

### Vector Length (VLEN) Detection

RISC-V vector extension variable length (VLEN) affects performance:

```bash
# Check VLEN on your CPU
cat /proc/cpuinfo | grep vlen

# Expected values:
# - 128 bits (16 bytes) = minimum
# - 256 bits (32 bytes) = common
# - 512 bits (64 bytes) = high performance
```

Larger VLEN generally means better performance for vectorized operations.

### Prefetch Optimization

The code automatically optimizes memory prefetching for RISC-V:

```
scratchpad_prefetch_mode: 0 = disabled (slowest)
scratchpad_prefetch_mode: 1 = prefetch.r (default, recommended)
scratchpad_prefetch_mode: 2 = prefetch.w (experimental)
```

### Memory Bandwidth Saturation

If experiencing memory bandwidth saturation (high latency):

1. Reduce mining threads
2. Increase L2/L3 cache by mining fewer threads per core
3. Enable cache QoS (AMD Ryzen): `cache_qos: true`

## Building with Custom Flags

To build with custom RISC-V flags:

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_FLAGS="-march=rv64gcv_zba_zbb_zbc_zbs -O3 -funroll-loops -fomit-frame-pointer" \
      ..
make -j$(nproc)
```

## Future Optimizations

- [ ] Zbk* (crypto) support detection and usage
- [ ] Optimal VLEN-aware algorithm selection
- [ ] Per-core memory affinity (NUMA support)
- [ ] Dynamic thread count adjustment based on thermals
- [ ] Cross-compile optimizations for various RISC-V cores

## References

- [RISC-V Vector Extension Spec](https://github.com/riscv/riscv-v-spec)
- [RISC-V Bit Manipulation Spec](https://github.com/riscv/riscv-bitmanip)
- [RISC-V Crypto Spec](https://github.com/riscv/riscv-crypto)
- [XMRig Documentation](https://xmrig.com/docs)

---

For further optimization, enable RVV intrinsics by replacing `sse2rvv.h` with `sse2rvv_optimized.h` in the build.
