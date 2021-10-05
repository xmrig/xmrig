# v6.15.2
- [#2606](https://github.com/xmrig/xmrig/pull/2606) Fixed: AstroBWT auto-config ignored `max-threads-hint`.
- Fixed possible crash on Windows (regression in v6.15.1).

# v6.15.1
- [#2586](https://github.com/xmrig/xmrig/pull/2586) Fixed Windows 7 compatibility.
- [#2594](https://github.com/xmrig/xmrig/pull/2594) Added Windows taskbar icon colors.

# v6.15.0
- [#2548](https://github.com/xmrig/xmrig/pull/2548) Added automatic coin detection for daemon mining.
- [#2563](https://github.com/xmrig/xmrig/pull/2563) Added new algorithm RandomX Graft (`rx/graft`).
- [#2565](https://github.com/xmrig/xmrig/pull/2565) AstroBWT: added AVX2 Salsa20 implementation.
- Added support for new CUDA plugin API (previous API still supported).

# v6.14.1
- [#2532](https://github.com/xmrig/xmrig/pull/2532) Refactoring: stable (persistent) algorithms IDs.
- [#2537](https://github.com/xmrig/xmrig/pull/2537) Fixed Termux build.

# v6.14.0
- [#2484](https://github.com/xmrig/xmrig/pull/2484) Added ZeroMQ support for solo mining.
- [#2476](https://github.com/xmrig/xmrig/issues/2476) Fixed crash in DMI memory reader.
- [#2492](https://github.com/xmrig/xmrig/issues/2492) Added missing `--huge-pages-jit` command line option.
- [#2512](https://github.com/xmrig/xmrig/pull/2512) Added show the number of transactions in pool job.

# v6.13.1
- [#2468](https://github.com/xmrig/xmrig/pull/2468) Fixed regression in previous version: don't send miner signature during regular mining.

# v6.13.0
- [#2445](https://github.com/xmrig/xmrig/pull/2445) Added support for solo mining with miner signatures for the upcoming Wownero fork.

# v6.12.2
- [#2280](https://github.com/xmrig/xmrig/issues/2280) GPU backends are now disabled in benchmark mode.
- [#2322](https://github.com/xmrig/xmrig/pull/2322) Improved MSR compatibility with recent Linux kernels and updated `randomx_boost.sh`.
- [#2340](https://github.com/xmrig/xmrig/pull/2340) Fixed AES detection on FreeBSD on ARM.
- [#2341](https://github.com/xmrig/xmrig/pull/2341) `sse2neon` updated to the latest version.
- [#2351](https://github.com/xmrig/xmrig/issues/2351) Fixed help output for `--cpu-priority` and `--cpu-affinity` option.
- [#2375](https://github.com/xmrig/xmrig/pull/2375) Fixed macOS CUDA backend default loader name.
- [#2378](https://github.com/xmrig/xmrig/pull/2378) Fixed broken light mode mining on x86.
- [#2379](https://github.com/xmrig/xmrig/pull/2379) Fixed CL code for KawPow where it assumes everything is AMD.
- [#2386](https://github.com/xmrig/xmrig/pull/2386) RandomX: enabled `IMUL_RCP` optimization for light mode mining.
- [#2393](https://github.com/xmrig/xmrig/pull/2393) RandomX: added BMI2 version for scratchpad prefetch.
- [#2395](https://github.com/xmrig/xmrig/pull/2395) RandomX: rewrote dataset read code.
- [#2398](https://github.com/xmrig/xmrig/pull/2398) RandomX: optimized ARMv8 dataset read.
- Added `argon2/ninja` alias for `argon2/wrkz` algorithm.

# v6.12.1
- [#2296](https://github.com/xmrig/xmrig/pull/2296) Fixed Zen3 assembly code for `cn/upx2` algorithm.

# v6.12.0
- [#2276](https://github.com/xmrig/xmrig/pull/2276) Added support for Uplexa (`cn/upx2` algorithm).
- [#2261](https://github.com/xmrig/xmrig/pull/2261) Show total hashrate if compiled without OpenCL.
- [#2289](https://github.com/xmrig/xmrig/pull/2289) RandomX: optimized `IMUL_RCP` instruction.
- Added support for `--user` command line option for online benchmark.

# v6.11.2
- [#2207](https://github.com/xmrig/xmrig/issues/2207) Fixed regression in HTTP parser and llhttp updated to v5.1.0.

# v6.11.1
- [#2239](https://github.com/xmrig/xmrig/pull/2239) Fixed broken `coin` setting functionality.

# v6.11.0
- [#2196](https://github.com/xmrig/xmrig/pull/2196) Improved DNS subsystem and added new DNS specific options.
- [#2172](https://github.com/xmrig/xmrig/pull/2172) Fixed build on Alpine 3.13.
- [#2177](https://github.com/xmrig/xmrig/pull/2177) Fixed ARM specific compilation error with GCC 10.2.
- [#2214](https://github.com/xmrig/xmrig/pull/2214) [#2216](https://github.com/xmrig/xmrig/pull/2216) [#2235](https://github.com/xmrig/xmrig/pull/2235) Optimized `cn-heavy` algorithm.
- [#2217](https://github.com/xmrig/xmrig/pull/2217) Fixed mining job creation sequence.
- [#2225](https://github.com/xmrig/xmrig/pull/2225) Fixed build without OpenCL support on some systems.
- [#2229](https://github.com/xmrig/xmrig/pull/2229) Don't use RandomX JIT if `WITH_ASM=OFF`.
- [#2228](https://github.com/xmrig/xmrig/pull/2228) Removed useless code for cryptonight algorithms.
- [#2234](https://github.com/xmrig/xmrig/pull/2234) Fixed build error on gcc 4.8.

# v6.10.0
- [#2122](https://github.com/xmrig/xmrig/pull/2122) Fixed pause logic when both pause on battery and user activity are enabled.
- [#2123](https://github.com/xmrig/xmrig/issues/2123) Fixed compatibility with gcc 4.8.
- [#2147](https://github.com/xmrig/xmrig/pull/2147) Fixed many `new job` messages when solo mining.
- [#2150](https://github.com/xmrig/xmrig/pull/2150) Updated `sse2neon.h` to the latest master, fixes build on ARMv7.
- [#2157](https://github.com/xmrig/xmrig/pull/2157) Fixed crash in `cn-heavy` on Zen3 with manual thread count.
- Fixed possible out of order write to log file.
- [http-parser](https://github.com/nodejs/http-parser) replaced to [llhttp](https://github.com/nodejs/llhttp).
- For official builds: libuv, hwloc and OpenSSL updated to latest versions.

# v6.9.0
- [#2104](https://github.com/xmrig/xmrig/pull/2104) Added [pause-on-active](https://xmrig.com/docs/miner/config/misc#pause-on-active) config option and `--pause-on-active=N` command line option.
- [#2112](https://github.com/xmrig/xmrig/pull/2112) Added support for [Tari merge mining](https://github.com/tari-project/tari/blob/development/README.md#tari-merge-mining).
- [#2117](https://github.com/xmrig/xmrig/pull/2117) Fixed crash when GPU mining `cn-heavy` on Zen3 system.

# v6.8.2
- [#2080](https://github.com/xmrig/xmrig/pull/2080) Fixed compile error in Termux.
- [#2089](https://github.com/xmrig/xmrig/pull/2089) Optimized CryptoNight-Heavy for Zen3, 7-8% speedup.

# v6.8.1
- [#2064](https://github.com/xmrig/xmrig/pull/2064) Added documentation for config.json CPU options.
- [#2066](https://github.com/xmrig/xmrig/issues/2066) Fixed AMD GPUs health data readings on Linux.
- [#2067](https://github.com/xmrig/xmrig/pull/2067) Fixed compilation error when RandomX and Argon2 are disabled.
- [#2076](https://github.com/xmrig/xmrig/pull/2076) Added support for flexible huge page sizes on Linux.
- [#2077](https://github.com/xmrig/xmrig/pull/2077) Fixed `illegal instruction` crash on ARM.

# v6.8.0
- [#2052](https://github.com/xmrig/xmrig/pull/2052) Added DMI/SMBIOS reader.
  - Added information about memory modules on the miner startup and for online benchmark.
  - Added new HTTP API endpoint: `GET /2/dmi`.
  - Added new command line option `--no-dmi` or config option `"dmi"`.
  - Added new CMake option `-DWITH_DMI=OFF`.
- [#2057](https://github.com/xmrig/xmrig/pull/2057) Improved MSR subsystem code quality.
- [#2058](https://github.com/xmrig/xmrig/pull/2058) RandomX JIT x86: removed unnecessary instructions.

# v6.7.2
- [#2039](https://github.com/xmrig/xmrig/pull/2039) Fixed solo mining.

# v6.7.1
- [#1995](https://github.com/xmrig/xmrig/issues/1995) Fixed log initialization.
- [#1998](https://github.com/xmrig/xmrig/pull/1998) Added hashrate in the benchmark finished message.
- [#2009](https://github.com/xmrig/xmrig/pull/2009) AstroBWT OpenCL fixes.
- [#2028](https://github.com/xmrig/xmrig/pull/2028) RandomX x86 JIT: removed redundant `CFROUND`.

# v6.7.0
- **[#1991](https://github.com/xmrig/xmrig/issues/1991) Added Apple M1 processor support.**
- **[#1986](https://github.com/xmrig/xmrig/pull/1986) Up to 20-30% faster RandomX dataset initialization with AVX2 on some CPUs.**
- [#1964](https://github.com/xmrig/xmrig/pull/1964) Cleanup and refactoring.
- [#1966](https://github.com/xmrig/xmrig/pull/1966) Removed libcpuid support.
- [#1968](https://github.com/xmrig/xmrig/pull/1968) Added virtual machine detection.
- [#1969](https://github.com/xmrig/xmrig/pull/1969) [#1970](https://github.com/xmrig/xmrig/pull/1970) Fixed errors found by static analysis.
- [#1977](https://github.com/xmrig/xmrig/pull/1977) Fixed: secure JIT and huge pages are incompatible on Windows.
- [#1979](https://github.com/xmrig/xmrig/pull/1979) Term `x64` replaced to `64-bit`.
- [#1980](https://github.com/xmrig/xmrig/pull/1980) Fixed build on gcc 11.
- [#1989](https://github.com/xmrig/xmrig/pull/1989) Fixed broken Dero solo mining.

# v6.6.2
- [#1958](https://github.com/xmrig/xmrig/pull/1958) Added example mining scripts to help new miners.
- [#1959](https://github.com/xmrig/xmrig/pull/1959) Optimized JIT compiler.
- [#1960](https://github.com/xmrig/xmrig/pull/1960) Fixed RandomX init when switching to other algo and back.

# v6.6.1
- Fixed, benchmark validation on NUMA hardware produced incorrect results in some conditions.

# v6.6.0
- Online benchmark protocol upgraded to v2, validation not compatible with previous versions.
  - Single thread benchmark now is cheat-resistant, not possible speedup it with multiple threads.
  - RandomX dataset is now always initialized with static seed, to prevent time cheat by report slow dataset initialization.
  - Zero delay online submission, to make time validation much more precise and strict.
  - DNS cache for online benchmark to prevent unexpected delays.

# v6.5.3
- [#1946](https://github.com/xmrig/xmrig/pull/1946) Fixed MSR mod names in JSON API (v6.5.2 affected).

# v6.5.2
- [#1935](https://github.com/xmrig/xmrig/pull/1935) Separate MSR mod for Zen/Zen2 and Zen3.
- [#1937](https://github.com/xmrig/xmrig/issues/1937) Print path to existing WinRing0 service without verbose option.
- [#1939](https://github.com/xmrig/xmrig/pull/1939) Fixed build with gcc 4.8.
- [#1941](https://github.com/xmrig/xmrig/pull/1941) Added CPUID info to JSON report.
- [#1941](https://github.com/xmrig/xmrig/pull/1942) Fixed alignment modification in memory pool.
- [#1944](https://github.com/xmrig/xmrig/pull/1944) Updated `randomx_boost.sh` with new MSR mod.
- Added `250K` and `500K` offline benchmarks.

# v6.5.1
- [#1932](https://github.com/xmrig/xmrig/pull/1932) New MSR mod for Ryzen, up to +3.5% on Zen2 and +1-2% on Zen3.
- [#1918](https://github.com/xmrig/xmrig/issues/1918) Fixed 1GB huge pages support on ARMv8.
- [#1926](https://github.com/xmrig/xmrig/pull/1926) Fixed compilation on ARMv8 with GCC 9.3.0.
- [#1929](https://github.com/xmrig/xmrig/issues/1929) Fixed build without HTTP.

# v6.5.0
- **Added [online benchmark](https://xmrig.com/benchmark) mode for sharing results.**
  - Added new command line options: `--submit`, `	--verify=ID`, `	--seed=SEED`, `--hash=HASH`.
- [#1912](https://github.com/xmrig/xmrig/pull/1912) Fixed MSR kernel module warning with new Linux kernels.
- [#1925](https://github.com/xmrig/xmrig/pull/1925) Add checking for config files in user home directory.
- Added vendor to ARM CPUs name and added `"arch"` field to API.
- Removed legacy CUDA plugin API.

# v6.4.0
- [#1862](https://github.com/xmrig/xmrig/pull/1862) **RandomX: removed `rx/loki` algorithm.**
- [#1890](https://github.com/xmrig/xmrig/pull/1890) **Added `argon2/chukwav2` algorithm.**
- [#1895](https://github.com/xmrig/xmrig/pull/1895) [#1897](https://github.com/xmrig/xmrig/pull/1897) **Added [benchmark and stress test](https://github.com/xmrig/xmrig/blob/dev/doc/BENCHMARK.md).**
- [#1864](https://github.com/xmrig/xmrig/pull/1864) RandomX: improved software AES performance.
- [#1870](https://github.com/xmrig/xmrig/pull/1870) RandomX: fixed unexpected resume due to disconnect during dataset init.
- [#1872](https://github.com/xmrig/xmrig/pull/1872) RandomX: fixed `randomx_create_vm` call.
- [#1875](https://github.com/xmrig/xmrig/pull/1875) RandomX: fixed crash on x86.
- [#1876](https://github.com/xmrig/xmrig/pull/1876) RandomX: added `huge-pages-jit` config parameter.
- [#1881](https://github.com/xmrig/xmrig/pull/1881) Fixed possible race condition in hashrate counting code.
- [#1882](https://github.com/xmrig/xmrig/pull/1882) [#1886](https://github.com/xmrig/xmrig/pull/1886) [#1887](https://github.com/xmrig/xmrig/pull/1887) [#1893](https://github.com/xmrig/xmrig/pull/1893) General code improvements.
- [#1885](https://github.com/xmrig/xmrig/pull/1885) Added more precise hashrate calculation.
- [#1889](https://github.com/xmrig/xmrig/pull/1889) Fixed libuv performance issue on Linux.

# v6.3.5
- [#1845](https://github.com/xmrig/xmrig/pull/1845) [#1861](https://github.com/xmrig/xmrig/pull/1861) Fixed ARM build and added CMake option `WITH_SSE4_1`.
- [#1846](https://github.com/xmrig/xmrig/pull/1846) KawPow: fixed OpenCL memory leak.
- [#1849](https://github.com/xmrig/xmrig/pull/1849) [#1859](https://github.com/xmrig/xmrig/pull/1859) RandomX: optimized soft AES code.
- [#1850](https://github.com/xmrig/xmrig/pull/1850) [#1852](https://github.com/xmrig/xmrig/pull/1852) General code improvements.
- [#1853](https://github.com/xmrig/xmrig/issues/1853) [#1856](https://github.com/xmrig/xmrig/pull/1856) [#1857](https://github.com/xmrig/xmrig/pull/1857) Fixed crash on old CPUs.

# v6.3.4
- [#1823](https://github.com/xmrig/xmrig/pull/1823) RandomX: added new option `scratchpad_prefetch_mode`.
- [#1827](https://github.com/xmrig/xmrig/pull/1827) [#1831](https://github.com/xmrig/xmrig/pull/1831) Improved nonce iteration performance.
- [#1828](https://github.com/xmrig/xmrig/pull/1828) RandomX: added SSE4.1-optimized Blake2b.
- [#1830](https://github.com/xmrig/xmrig/pull/1830) RandomX: added performance profiler (for developers).
- [#1835](https://github.com/xmrig/xmrig/pull/1835) RandomX: returned old soft AES implementation and added auto-select between the two.
- [#1840](https://github.com/xmrig/xmrig/pull/1840) RandomX: moved more stuff to compile time, small x86 JIT compiler speedup.
- [#1841](https://github.com/xmrig/xmrig/pull/1841) Fixed Cryptonight OpenCL for AMD 20.7.2 drivers.
- [#1842](https://github.com/xmrig/xmrig/pull/1842) RandomX: AES improvements, a bit faster hardware AES code when compiled with MSVC.
- [#1843](https://github.com/xmrig/xmrig/pull/1843) RandomX: improved performance of GCC compiled binaries.

# v6.3.3
- [#1817](https://github.com/xmrig/xmrig/pull/1817) Fixed self-select login sequence.
- Added brand new [build from source](https://xmrig.com/docs/miner/build) documentation.
- New binary downloads for macOS (`macos-x64`), FreeBSD (`freebsd-static-x64`), Linux (`linux-static-x64`), Ubuntu 18.04 (`bionic-x64`), Ubuntu 20.04 (`focal-x64`).
- Generic Linux download `xenial-x64` renamed to `linux-x64`.
- Builds without SSL/TLS support are no longer provided.
- Improved CUDA loader error reporting and fixed plugin load on Linux.
- Fixed build warnings with Clang compiler.
- Fixed colors on macOS.

# v6.3.2
- [#1794](https://github.com/xmrig/xmrig/pull/1794) More robust 1 GB pages handling.
  - Don't allocate 1 GB per thread if 1 GB is the default huge page size.
  - Try to allocate scratchpad from dataset's 1 GB huge pages, if normal huge pages are not available.
  - Correctly initialize RandomX cache if 1 GB pages fail to allocate on a first NUMA node.
- [#1806](https://github.com/xmrig/xmrig/pull/1806) Fixed macOS battery detection.
- [#1809](https://github.com/xmrig/xmrig/issues/1809) Improved auto configuration on ARM CPUs.
  - Added retrieving ARM CPU names, based on lscpu code and database.

# v6.3.1
- [#1786](https://github.com/xmrig/xmrig/pull/1786) Added `pause-on-battery` option, supported on Windows and Linux.
- Added command line options `--randomx-cache-qos` and `--argon2-impl`.

# v6.3.0
- [#1771](https://github.com/xmrig/xmrig/pull/1771) Adopted new SSE2NEON and reduced ARM-specific changes.
- [#1774](https://github.com/xmrig/xmrig/pull/1774) RandomX: Added new option `cache_qos` in `randomx` object for cache QoS support.
- [#1777](https://github.com/xmrig/xmrig/pull/1777) Added support for upcoming Haven offshore fork.
  - [#1780](https://github.com/xmrig/xmrig/pull/1780) CryptoNight OpenCL: fix for long input data.

# v6.2.3
- [#1745](https://github.com/xmrig/xmrig/pull/1745) AstroBWT: fixed OpenCL compilation on some systems.
- [#1749](https://github.com/xmrig/xmrig/pull/1749) KawPow: optimized CPU share verification.
- [#1752](https://github.com/xmrig/xmrig/pull/1752) RandomX: added error message when MSR mod fails.
- [#1754](https://github.com/xmrig/xmrig/issues/1754) Fixed GPU health readings for pre Vega GPUs on Linux.
- [#1756](https://github.com/xmrig/xmrig/issues/1756) Added results and connection reports.
- [#1759](https://github.com/xmrig/xmrig/pull/1759) KawPow: fixed DAG initialization on slower AMD GPUs.
- [#1763](https://github.com/xmrig/xmrig/pull/1763) KawPow: fixed rare duplicate share errors.
- [#1766](https://github.com/xmrig/xmrig/pull/1766) RandomX: small speedup on Ryzen CPUs.

# v6.2.2
- [#1742](https://github.com/xmrig/xmrig/issues/1742) Fixed crash when use HTTP API.

# v6.2.1
- [#1726](https://github.com/xmrig/xmrig/issues/1726) Fixed detection of AVX2/AVX512.
- [#1728](https://github.com/xmrig/xmrig/issues/1728) Fixed, 32 bit Windows builds was crash on start.
- [#1729](https://github.com/xmrig/xmrig/pull/1729) Fixed KawPow crash on old CPUs.
- [#1730](https://github.com/xmrig/xmrig/pull/1730) Improved displaying information for compute errors on GPUs.
- [#1732](https://github.com/xmrig/xmrig/pull/1732) Fixed NiceHash disconnects for KawPow.
- Fixed AMD GPU health (temperatures/power/clocks/fans) readings on Linux.

# v6.2.0-beta
- [#1717](https://github.com/xmrig/xmrig/pull/1717) Added new algorithm `cn/ccx` for Conceal.
- [#1718](https://github.com/xmrig/xmrig/pull/1718) Fixed, linker on Linux was marking entire executable as having an executable stack.
- [#1720](https://github.com/xmrig/xmrig/pull/1720) Fixed broken CryptoNight algorithms family with gcc 10.1.

# v6.0.1-beta
- [#1708](https://github.com/xmrig/xmrig/issues/1708) Added `title` option.
- [#1711](https://github.com/xmrig/xmrig/pull/1711) [cuda] Print errors from KawPow DAG initialization.
- [#1713](https://github.com/xmrig/xmrig/pull/1713) [cuda] Reduced memory usage for KawPow, minimum CUDA plugin version now is 6.1.0.

# v6.0.0-beta
- [#1694](https://github.com/xmrig/xmrig/pull/1694) Added support for KawPow algorithm (Ravencoin) on AMD/NVIDIA.
- Removed previously deprecated `cn/gpu` algorithm.
- Default donation level reduced to 1% but you still can increase it if you like.

# v5.11.3
- [#1718](https://github.com/xmrig/xmrig/pull/1718) Fixed, linker on Linux was marking entire executable as having an executable stack.
- [#1720](https://github.com/xmrig/xmrig/pull/1720) Fixed broken CryptoNight algorithms family with gcc 10.1.

# v5.11.2
- [#1664](https://github.com/xmrig/xmrig/pull/1664) Improved JSON config error reporting.
- [#1668](https://github.com/xmrig/xmrig/pull/1668) Optimized RandomX dataset initialization.
- [#1675](https://github.com/xmrig/xmrig/pull/1675) Fixed cross-compiling on Linux.
- Fixed memory leak in HTTP client.
- Build [dependencies](https://github.com/xmrig/xmrig-deps/releases/tag/v4.1) updated to recent versions.
- Compiler for Windows gcc builds updated to v10.1.

# v5.11.1
- [#1652](https://github.com/xmrig/xmrig/pull/1652) Up to 1% RandomX perfomance improvement on recent AMD CPUs.
- [#1306](https://github.com/xmrig/xmrig/issues/1306) Fixed possible double connection to a pool.
- [#1654](https://github.com/xmrig/xmrig/issues/1654) Fixed build with LibreSSL.

# v5.11.0
- **[#1632](https://github.com/xmrig/xmrig/pull/1632) Added AstroBWT CUDA support ([CUDA plugin](https://github.com/xmrig/xmrig-cuda) v3.0.0 or newer required).**
- [#1605](https://github.com/xmrig/xmrig/pull/1605) Fixed AstroBWT OpenCL for NVIDIA GPUs.
- [#1635](https://github.com/xmrig/xmrig/pull/1635) Added pooled memory allocation of RandomX VMs (+0.5% speedup on Zen2).
- [#1641](https://github.com/xmrig/xmrig/pull/1641) RandomX JIT refactoring, smaller memory footprint and a bit faster overall.
- [#1643](https://github.com/xmrig/xmrig/issues/1643) Fixed build on CentOS 7.

# v5.10.0
- [#1602](https://github.com/xmrig/xmrig/pull/1602) Added AMD GPUs support for AstroBWT algorithm.
- [#1590](https://github.com/xmrig/xmrig/pull/1590) MSR mod automatically deactivated after switching from RandomX algorithms.
- [#1592](https://github.com/xmrig/xmrig/pull/1592) Added AVX2 optimized code for AstroBWT algorithm.
  - Added new config option `astrobwt-avx2` in `cpu` object and command line option `--astrobwt-avx2`.
- [#1596](https://github.com/xmrig/xmrig/issues/1596) Major TLS (Transport Layer Security) subsystem update.
  - Added new TLS options, please check [xmrig-proxy documentation](https://xmrig.com/docs/proxy/tls) for details.
- `cn/gpu` algorithm now disabled by default and will be removed in next major (v6.x.x) release, no ETA for it right now.
- Added command line option `--data-dir`.

# v5.9.0
- [#1578](https://github.com/xmrig/xmrig/pull/1578) Added new RandomKEVA algorithm for upcoming Kevacoin fork, as `"algo": "rx/keva"` or `"coin": "keva"`.
- [#1584](https://github.com/xmrig/xmrig/pull/1584) Fixed invalid AstroBWT hashes after algorithm switching.
- [#1585](https://github.com/xmrig/xmrig/issues/1585) Fixed build without HTTP support.
- Added command line option `--astrobwt-max-size`.

# v5.8.2
- [#1580](https://github.com/xmrig/xmrig/pull/1580) AstroBWT algorithm 20-50% speedup.
  - Added new option `astrobwt-max-size`.
- [#1581](https://github.com/xmrig/xmrig/issues/1581) Fixed macOS build.

# v5.8.1
- [#1575](https://github.com/xmrig/xmrig/pull/1575) Fixed new block detection for DERO solo mining.

# v5.8.0
- [#1573](https://github.com/xmrig/xmrig/pull/1573) Added new AstroBWT algorithm for upcoming DERO fork, as `"algo": "astrobwt"` or `"coin": "dero"`.

# v5.7.0
- **Added SOCKS5 proxies support for Tor https://xmrig.com/docs/miner/tor.**
- [#377](https://github.com/xmrig/xmrig-proxy/issues/377) Fixed duplicate jobs in daemon (solo) mining client.
- [#1560](https://github.com/xmrig/xmrig/pull/1560) RandomX 0.3-0.4% speedup depending on CPU.
- Fixed possible crashes in HTTP client.

# v5.6.0
- [#1536](https://github.com/xmrig/xmrig/pull/1536) Added workaround for new AMD GPU drivers.
- [#1546](https://github.com/xmrig/xmrig/pull/1546) Fixed generic OpenCL code for AMD Navi GPUs.
- [#1551](https://github.com/xmrig/xmrig/pull/1551) Added RandomX JIT for AMD Navi  GPUs.
- Added health information for AMD GPUs (clocks/power/fan/temperature) via ADL (Windows) and sysfs (Linux).
- Fixed possible nicehash nonce overflow in some conditions.
- Fixed wrong OpenCL platform on macOS, option `platform` now ignored on this OS.

# v5.5.3
- [#1529](https://github.com/xmrig/xmrig/pull/1529) Fixed crash on Bulldozer CPUs.

# v5.5.2
- [#1500](https://github.com/xmrig/xmrig/pull/1500) Removed unnecessary code from RandomX JIT compiler.
- [#1502](https://github.com/xmrig/xmrig/pull/1502) Optimizations for AMD Bulldozer.
- [#1508](https://github.com/xmrig/xmrig/pull/1508) Added support for BMI2 instructions.
- [#1510](https://github.com/xmrig/xmrig/pull/1510) Optimized `CFROUND` instruction for RandomX.
- [#1520](https://github.com/xmrig/xmrig/pull/1520) Fixed thread affinity.

# v5.5.1
- [#1469](https://github.com/xmrig/xmrig/issues/1469) Fixed build with gcc 4.8.
- [#1473](https://github.com/xmrig/xmrig/pull/1473) Added RandomX auto-config for mobile Ryzen APUs.
- [#1477](https://github.com/xmrig/xmrig/pull/1477) Fixed build with Clang.
- [#1489](https://github.com/xmrig/xmrig/pull/1489) RandomX JIT compiler tweaks.
- [#1493](https://github.com/xmrig/xmrig/pull/1493) Default value for Intel MSR preset changed to `15`.
- Fixed unwanted resume after RandomX dataset change.

# v5.5.0
- [#179](https://github.com/xmrig/xmrig/issues/179) Added support for [environment variables](https://xmrig.com/docs/miner/environment-variables) in config file.
- [#1445](https://github.com/xmrig/xmrig/pull/1445) Removed `rx/v` algorithm.
- [#1453](https://github.com/xmrig/xmrig/issues/1453) Fixed crash on 32bit systems.
- [#1459](https://github.com/xmrig/xmrig/issues/1459) Fixed crash on very low memory systems.
- [#1465](https://github.com/xmrig/xmrig/pull/1465) Added fix for 1st-gen Ryzen crashes.
- [#1466](https://github.com/xmrig/xmrig/pull/1466) Added `cn-pico/tlo` algorithm.
- Added `--randomx-no-rdmsr` command line option.
- Added console title for Windows with miner name and version.
- On Windows `priority` option now also change base priority.

# v5.4.0
- [#1434](https://github.com/xmrig/xmrig/pull/1434) Added RandomSFX (`rx/sfx`) algorithm for Safex Cash.
- [#1445](https://github.com/xmrig/xmrig/pull/1445) Added RandomV (`rx/v`) algorithm for *new* MoneroV.
- [#1419](https://github.com/xmrig/xmrig/issues/1419) Added reverting MSR changes on miner exit, use `"rdmsr": false,` in `"randomx"` object to disable this feature.
- [#1423](https://github.com/xmrig/xmrig/issues/1423) Fixed conflicts with exists WinRing0 driver service.
- [#1425](https://github.com/xmrig/xmrig/issues/1425) Fixed crash on first generation Zen CPUs (MSR mod accidentally enable Opcache), additionally now you can disable Opcache and enable MSR mod via config `"wrmsr": ["0xc0011020:0x0", "0xc0011021:0x60", "0xc0011022:0x510000", "0xc001102b:0x1808cc16"],`.
- Added advanced usage for `wrmsr` option, for example: `"wrmsr": ["0x1a4:0x6"],` (Intel) and `"wrmsr": ["0xc0011020:0x0", "0xc0011021:0x40:0xffffffffffffffdf", "0xc0011022:0x510000", "0xc001102b:0x1808cc16"],` (Ryzen).
- Added new config option `"verbose"` and command line option `--verbose`.

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
