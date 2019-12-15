# v5.3.0
- [#1414](https://github.com/xmrig/xmrig/pull/1414) Added native MSR support for Windows, by using signed **WinRing0 driver** (© 2007-2009 OpenLibSys.org).
- Added new [MSR documentation](https://xmrig.com/docs/miner/randomx-optimization-guide/msr).
- [#1418](https://github.com/xmrig/xmrig/pull/1418) Increased stratum send buffer size.

# v5.2.1
- [#1408](https://github.com/xmrig/xmrig/pull/1408) Added RandomX boost script for Linux (if you don't like run miner with root privileges).
- Added support for [AMD Ryzen MSR registers](https://www.reddit.com/r/MoneroMining/comments/e962fu/9526_hs_on_ryzen_7_3700x_xmrig_520_1gb_pages_msr/) (Linux only).
- Fixed command line option `--randomx-wrmsr` option without parameters.

# v5.2.0
- **[#1388](https://github.com/xmrig/xmrig/pull/1388) Added [1GB huge pages support](https://xmrig.com/docs/miner/hugepages#onegb-huge-pages) for Linux.**
  - Added new option `1gb-pages` in `randomx` object with command line equivalent `--randomx-1gb-pages`.
  - Added automatic huge pages configuration on Linux if use the miner with root privileges.
- **Added [automatic Intel prefetchers configuration](https://xmrig.com/docs/miner/randomx-optimization-guide#intel-specific-optimizations) on Linux.**
   - Added new option `wrmsr` in `randomx` object with command line equivalent `--randomx-wrmsr=6`.
- [#1396](https://github.com/xmrig/xmrig/pull/1396) [#1401](https://github.com/xmrig/xmrig/pull/1401) New performance optimizations for Ryzen CPUs. 
- [#1385](https://github.com/xmrig/xmrig/issues/1385) Added `max-threads-hint` option support for RandomX dataset initialization threads.  
- [#1386](https://github.com/xmrig/xmrig/issues/1386) Added `priority` option support for RandomX dataset initialization threads. 
- For official builds all dependencies (libuv, hwloc, openssl) updated to recent versions.
- Windows `msvc` builds now use Visual Studio 2019 instead of 2017.

# v5.1.1
- [#1365](https://github.com/xmrig/xmrig/issues/1365) Fixed various system response/stability issues.
  - Added new CPU option `yield` and command line equivalent `--cpu-no-yield`.
- [#1363](https://github.com/xmrig/xmrig/issues/1363) Fixed wrong priority of main miner thread.

# v5.1.0
- [#1351](https://github.com/xmrig/xmrig/pull/1351) RandomX optimizations and fixes.
  - Improved RandomX performance (up to +6-7% on Intel CPUs, +2-3% on Ryzen CPUs)
  - Added workaround for Intel JCC erratum bug see https://www.phoronix.com/scan.php?page=article&item=intel-jcc-microcode&num=1 for details.
  - Note! Always disable "Hardware prefetcher" and "Adjacent cacheline prefetch" in BIOS for Intel CPUs to get the optimal RandomX performance.
- [#1307](https://github.com/xmrig/xmrig/issues/1307) Fixed mining resume after donation round for pools with `self-select` feature.
- [#1318](https://github.com/xmrig/xmrig/issues/1318#issuecomment-559676080) Added option `"mode"` (or `--randomx-mode`) for RandomX.
  - Added memory information on miner startup.
  - Added `resources` field to summary API with memory information and load average.

# v5.0.1
- [#1234](https://github.com/xmrig/xmrig/issues/1234) Fixed compatibility with some AMD GPUs.
- [#1284](https://github.com/xmrig/xmrig/issues/1284) Fixed build without RandomX.
- [#1285](https://github.com/xmrig/xmrig/issues/1285) Added command line options `--cuda-bfactor-hint` and `--cuda-bsleep-hint`.
- [#1290](https://github.com/xmrig/xmrig/pull/1290) Fixed 32-bit ARM compilation.

# v5.0.0
This version is first stable unified 3 in 1 GPU+CPU release, OpenCL support built in in miner and not require additional external dependencies on compile time, NVIDIA CUDA available as external [CUDA plugin](https://github.com/xmrig/xmrig-cuda), for convenient, 3 in 1 downloads with recent CUDA version also provided.

This release based on 4.x.x series and include all features from v4.6.2-beta, changelog below include only the most important changes, [full changelog](doc/CHANGELOG_OLD.md) available separately.

- [#1272](https://github.com/xmrig/xmrig/pull/1272) Optimized hashrate calculation.
- [#1263](https://github.com/xmrig/xmrig/pull/1263) Added new option `dataset_host` for NVIDIA GPUs with less than 4 GB memory (RandomX only).
- [#1068](https://github.com/xmrig/xmrig/pull/1068) Added support for `self-select` stratum protocol extension.
- [#1227](https://github.com/xmrig/xmrig/pull/1227) Added new algorithm `rx/arq`, RandomX variant for upcoming ArQmA fork.
- [#808](https://github.com/xmrig/xmrig/issues/808#issuecomment-539297156) Added experimental support for persistent memory for CPU mining threads.
- [#1221](https://github.com/xmrig/xmrig/issues/1221) Improved RandomX dataset memory usage and initialization speed for NUMA machines.
- [#1175](https://github.com/xmrig/xmrig/issues/1175) Fixed support for systems where total count of NUMA nodes not equal usable nodes count.
- Added config option `cpu/max-threads-hint` and command line option `--cpu-max-threads-hint`.
- [#1185](https://github.com/xmrig/xmrig/pull/1185) Added JIT compiler for RandomX on ARMv8.
- Improved API endpoint `GET /2/backends` and added support for this endpoint to [workers.xmrig.info](http://workers.xmrig.info).
- Added command line option `--no-cpu` to disable CPU backend.
- Added OpenCL specific command line options: `--opencl`, `--opencl-devices`, `--opencl-platform`, `--opencl-loader` and `--opencl-no-cache`.
- Added CUDA specific command line options: `--cuda`, `--cuda-loader` and `--no-nvml`.
- Removed command line option `--http-enabled`, HTTP API enabled automatically if any other `--http-*` option provided.
- [#1172](https://github.com/xmrig/xmrig/issues/1172) **Added OpenCL mining backend.**
  - [#268](https://github.com/xmrig/xmrig-amd/pull/268) [#270](https://github.com/xmrig/xmrig-amd/pull/270) [#271](https://github.com/xmrig/xmrig-amd/pull/271) [#273](https://github.com/xmrig/xmrig-amd/pull/273) [#274](https://github.com/xmrig/xmrig-amd/pull/274) [#1171](https://github.com/xmrig/xmrig/pull/1171) Added RandomX support for OpenCL, thanks [@SChernykh](https://github.com/SChernykh).
- Algorithm `cn/wow` removed, as no longer alive. 

# Previous versions
[doc/CHANGELOG_OLD.md](doc/CHANGELOG_OLD.md)
