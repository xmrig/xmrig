# v4.6.2-beta
- [#1274](https://github.com/xmrig/xmrig/issues/1274) Added `--cuda-devices` command line option.
- [#1277](https://github.com/xmrig/xmrig/pull/1277) Fixed function names for clang on Apple.

# v4.6.1-beta
- [#1272](https://github.com/xmrig/xmrig/pull/1272) Optimized hashrate calculation.
- [#1273](https://github.com/xmrig/xmrig/issues/1273) Fixed crash when use `GET /2/backends` API endpoint with disabled CUDA.

# v4.6.0-beta
- [#1263](https://github.com/xmrig/xmrig/pull/1263) Added new option `dataset_host` for NVIDIA GPUs with less than 4 GB memory (RandomX only).

# v4.5.0-beta
- Added NVIDIA CUDA support via external [CUDA plugun](https://github.com/xmrig/xmrig-cuda). XMRig now is unified 3 in 1 miner.

# v4.4.0-beta
- [#1068](https://github.com/xmrig/xmrig/pull/1068) Added support for `self-select` stratum protocol extension.
- [#1240](https://github.com/xmrig/xmrig/pull/1240) Sync with the latest RandomX code.
- [#1241](https://github.com/xmrig/xmrig/issues/1241) Fixed regression with colors on old Windows systems.
- [#1243](https://github.com/xmrig/xmrig/pull/1243) Fixed incorrect OpenCL memory size detection in some cases.
- [#1247](https://github.com/xmrig/xmrig/pull/1247) Fixed ARM64 RandomX code alignment.
- [#1248](https://github.com/xmrig/xmrig/pull/1248) Fixed RandomX code cache cleanup on iOS/Darwin.

# v4.3.1-beta
- Fixed regression in v4.3.0, miner didn't create `cn` mining profile with default config example.

# v4.3.0-beta
- [#1227](https://github.com/xmrig/xmrig/pull/1227) Added new algorithm `rx/arq`, RandomX variant for upcoming ArQmA fork.
- [#808](https://github.com/xmrig/xmrig/issues/808#issuecomment-539297156) Added experimental support for persistent memory for CPU mining threads.
- [#1221](https://github.com/xmrig/xmrig/issues/1221) Improved RandomX dataset memory usage and initialization speed for NUMA machines.

# v4.2.1-beta
- [#1150](https://github.com/xmrig/xmrig/issues/1150) Fixed build on FreeBSD.
- [#1175](https://github.com/xmrig/xmrig/issues/1175) Fixed support for systems where total count of NUMA nodes not equal usable nodes count.
- [#1199](https://github.com/xmrig/xmrig/issues/1199) Fixed excessive memory allocation for OpenCL threads with low intensity.
- [#1212](https://github.com/xmrig/xmrig/issues/1212) Fixed low RandomX performance after fast algorithm switching.

# v4.2.0-beta
- [#1202](https://github.com/xmrig/xmrig/issues/1202) Fixed algorithm verification in donate strategy.
- Added per pool option `coin` with single possible value `monero` for pools without algorithm negotiation, for upcoming Monero fork.
- Added config option `cpu/max-threads-hint` and command line option `--cpu-max-threads-hint`.

# v4.1.0-beta
- **OpenCL backend disabled by default.**.
- [#1183](https://github.com/xmrig/xmrig/issues/1183) Fixed compatibility with systemd.
- [#1185](https://github.com/xmrig/xmrig/pull/1185) Added JIT compiler for RandomX on ARMv8.
- Improved API endpoint `GET /2/backends` and added support for this endpoint to [workers.xmrig.info](http://workers.xmrig.info).
- Added command line option `--no-cpu` to disable CPU backend.
- Added OpenCL specific command line options: `--opencl`, `--opencl-devices`, `--opencl-platform`, `--opencl-loader` and `--opencl-no-cache`.
- Removed command line option `--http-enabled`, HTTP API enabled automatically if any other `--http-*` option provided.

# v4.0.1-beta
- [#1177](https://github.com/xmrig/xmrig/issues/1177) Fixed compatibility with old AMD drivers.
- [#1180](https://github.com/xmrig/xmrig/issues/1180) Fixed possible duplicated shares after algorithm switching.
- Added support for case if not all backend threads successfully started.
- Fixed wrong config file permissions after write (only gcc builds on recent Windows 10 affected).

# v4.0.0-beta
- [#1172](https://github.com/xmrig/xmrig/issues/1172) **Added OpenCL mining backend.**
  - [#268](https://github.com/xmrig/xmrig-amd/pull/268) [#270](https://github.com/xmrig/xmrig-amd/pull/270) [#271](https://github.com/xmrig/xmrig-amd/pull/271) [#273](https://github.com/xmrig/xmrig-amd/pull/273) [#274](https://github.com/xmrig/xmrig-amd/pull/274) [#1171](https://github.com/xmrig/xmrig/pull/1171) Added RandomX support for OpenCL, thanks [@SChernykh](https://github.com/SChernykh).
- Algorithm `cn/wow` removed, as no longer alive. 

# v3.2.0
- Added per pool option `coin` with single possible value `monero` for pools without algorithm negotiation, for upcoming Monero fork.
- [#1183](https://github.com/xmrig/xmrig/issues/1183) Fixed compatibility with systemd.

# v3.1.3
- [#1180](https://github.com/xmrig/xmrig/issues/1180) Fixed possible duplicated shares after algorithm switching.
- Fixed wrong config file permissions after write (only gcc builds on recent Windows 10 affected).

# v3.1.2
- Many RandomX optimizations and fixes.
  - [#1132](https://github.com/xmrig/xmrig/issues/1132) Fixed build on CentOS 7.
  - [#1163](https://github.com/xmrig/xmrig/pull/1163) Optimized soft AES code, up to +30% hashrate on CPU without AES support and other optimizations.
  - [#1166](https://github.com/xmrig/xmrig/pull/1166) Fixed crash when initialize dataset with big threads count (eg 272).
  - [#1168](https://github.com/xmrig/xmrig/pull/1168) Optimized loading from scratchpad.
- [#1128](https://github.com/xmrig/xmrig/issues/1128) Fixed CMake 2.8 compatibility.

# v3.1.1
- [#1133](https://github.com/xmrig/xmrig/issues/1133) Fixed syslog regression.
- [#1138](https://github.com/xmrig/xmrig/issues/1138) Fixed multiple network bugs.
- [#1141](https://github.com/xmrig/xmrig/issues/1141) Fixed log in background mode.
- [#1142](https://github.com/xmrig/xmrig/pull/1142) RandomX hashrate improved by 0.5-1.5% depending on variant and CPU.
- [#1146](https://github.com/xmrig/xmrig/pull/1146) Fixed race condition in RandomX thread init.
- [#1148](https://github.com/xmrig/xmrig/pull/1148) Fixed, on Linux linker marking entire executable as having an executable stack.
- Fixed, for Argon2 algorithms command line options like `--threads` was ignored.
- Fixed command line options for single pool, free order allowed again.

# v3.1.0
- [#1107](https://github.com/xmrig/xmrig/issues/1107#issuecomment-522235892) Added Argon2 algorithm family: `argon2/chukwa` and `argon2/wrkz`.

# v3.0.0
- **[#1111](https://github.com/xmrig/xmrig/pull/1111) Added RandomX (`rx/test`) algorithm for testing and benchmarking.**
- **[#1036](https://github.com/xmrig/xmrig/pull/1036) Added RandomWOW (`rx/wow`) algorithm for [Wownero](http://wownero.org/).**
- **[#1050](https://github.com/xmrig/xmrig/pull/1050) Added RandomXL (`rx/loki`) algorithm for [Loki](https://loki.network/).**
- **[#1077](https://github.com/xmrig/xmrig/issues/1077) Added NUMA support via hwloc**.
- **Added flexible [multi algorithm](doc/CPU.md) configuration.**
- **Added unlimited switching between incompatible algorithms, all mining options can be changed in runtime.**
- [#257](https://github.com/xmrig/xmrig-nvidia/pull/257) New logging subsystem, file and syslog now always without colors.
- [#314](https://github.com/xmrig/xmrig-proxy/issues/314) Added donate over proxy feature.
- [#1007](https://github.com/xmrig/xmrig/issues/1007) Old HTTP API backend based on libmicrohttpd, replaced to custom HTTP server (libuv + http_parser).
- [#1010](https://github.com/xmrig/xmrig/pull/1010#issuecomment-482632107) Added daemon support (solo mining).
- [#1066](https://github.com/xmrig/xmrig/issues/1066#issuecomment-518080529) Added error message if pool not ready for RandomX.
- [#1105](https://github.com/xmrig/xmrig/issues/1105) Improved auto configuration for `cn-pico` algorithm.
- Added commands `pause` and `resume` via JSON RPC 2.0 API (`POST /json_rpc`).
- Added command line option `--export-topology` for export hwloc topology to a XML file.
- Breaked backward compatibility with previous configs and command line, `variant` option replaced to `algo`, global option `algo` removed, all CPU related settings moved to `cpu` object.
- Options `av`, `safe` and `max-cpu-usage` removed.
- Algorithm `cn/msr` renamed to `cn/fast`.
- Algorithm `cn/xtl` removed.
- API endpoint `GET /1/threads` replaced to `GET /2/backends`.
- Added global uptime and extended connection information in API.
- API now return current algorithm.

# v2.99.6-beta
- Added commands `pause` and `resume` via JSON RPC 2.0 API (`POST /json_rpc`).
- Fixed autoconfig regression (since 2.99.5), mostly `rx/wow` was affected by this bug.
- Fixed user job recovery after donation round.
- Information about AVX2 CPU feature how hidden in miner summary.

# v2.99.5-beta
- [#1066](https://github.com/xmrig/xmrig/issues/1066#issuecomment-518080529) Fixed crash and added error message if pool not ready for RandomX.
- [#1092](https://github.com/xmrig/xmrig/issues/1092) Fixed crash if wrong CPU affinity used.
- [#1103](https://github.com/xmrig/xmrig/issues/1103) Improved auto configuration for RandomX for CPUs where L2 cache is limiting factor.
- [#1105](https://github.com/xmrig/xmrig/issues/1105) Improved auto configuration for `cn-pico` algorithm.
- [#1106](https://github.com/xmrig/xmrig/issues/1106) Fixed `hugepages` field in summary API. 
- Added alternative short format for CPU threads.
- Changed format for CPU threads with intensity above 1.
- Name for reference RandomX configuration changed to `rx/test` to avoid potential conflicts in future.

# v2.99.4-beta
- [#1062](https://github.com/xmrig/xmrig/issues/1062) Fixed 32 bit support. **32 bit is slow and deprecated**.
- [#1088](https://github.com/xmrig/xmrig/pull/1088) Fixed macOS compilation.
- [#1095](https://github.com/xmrig/xmrig/pull/1095) Fixed compatibility with hwloc 1.10.x.
- Optimized RandomX initialization and switching, fixed rare crash when re-initialize dataset.
- Fixed ARM build with hwloc.

# v2.99.3-beta
- [#1082](https://github.com/xmrig/xmrig/issues/1082) Fixed hwloc auto configuration on AMD FX CPUs.
- Added command line option `--export-topology` for export hwloc topology to a XML file.

# v2.99.2-beta
- [#1077](https://github.com/xmrig/xmrig/issues/1077) Added NUMA support via **hwloc**.
- Fixed miner freeze when switch between RandomX variants.
- Fixed dataset initialization speed on Linux if thread affinity was used.

# v2.99.1-beta
- [#1072](https://github.com/xmrig/xmrig/issues/1072) Fixed RandomX `seed_hash` re-initialization.

# v2.99.0-beta
- [#1050](https://github.com/xmrig/xmrig/pull/1050) Added RandomXL algorithm for [Loki](https://loki.network/), algorithm name used by miner is `randomx/loki` or `rx/loki`.
- Added [flexible](https://github.com/xmrig/xmrig/blob/evo/doc/CPU.md) multi algorithm configuration.
- Added unlimited switching between incompatible algorithms, all mining options can be changed in runtime.
- Breaked backward compatibility with previous configs and command line, `variant` option replaced to `algo`, global option `algo` removed, all CPU related settings moved to `cpu` object.
- Options `av`, `safe` and `max-cpu-usage` removed.
- Algorithm `cn/msr` renamed to `cn/fast`.
- Algorithm `cn/xtl` removed.
- API endpoint `GET /1/threads` replaced to `GET /2/backends`.

# v2.16.0-beta
- [#1036](https://github.com/xmrig/xmrig/pull/1036) Added RandomWOW (RandomX with different preferences) algorithm support for [Wownero](http://wownero.org/).
  - Algorithm name used by miner is `randomx/wow` or `rx/wow`.
  - Currently runtime algorithm switching NOT supported with other algorithms.

# v2.15.4-beta
- Added global uptime and extended connection information in API.
- API now return current algorithm instead of global algorithm specified in config.
- This version also include all changes from stable version v2.14.4.

# v2.15.3-beta
- [#1014](https://github.com/xmrig/xmrig/issues/1014) Fixed regression, default value for `algo` option was not applied.

# v2.15.2-beta
- [#1010](https://github.com/xmrig/xmrig/pull/1010#issuecomment-482632107) Added daemon support (solo mining).
- [#1012](https://github.com/xmrig/xmrig/pull/1012) Fixed compatibility with clang 9.
- Config subsystem was rewritten, internally JSON is primary format now.
- Fixed regression, big HTTP responses was truncated.

# v2.15.1-beta
- [#1007](https://github.com/xmrig/xmrig/issues/1007) Old HTTP API backend based on libmicrohttpd, replaced to custom HTTP server (libuv + http_parser).
- [#257](https://github.com/xmrig/xmrig-nvidia/pull/257) New logging subsystem, file and syslog now always without colors.

# v2.15.0-beta
- [#314](https://github.com/xmrig/xmrig-proxy/issues/314) Added donate over proxy feature.
  - Added new option `donate-over-proxy`.
  - Added real graceful exit.
  
# v2.14.4
- [#992](https://github.com/xmrig/xmrig/pull/992)  Fixed compilation with Clang 3.5.
- [#1012](https://github.com/xmrig/xmrig/pull/1012) Fixed compilation with Clang 9.0.
- In HTTP API for unknown hashrate now used `null` instead of `0.0`.
- Fixed MSVC 2019 version detection.
- Removed obsolete automatic variants.

# v2.14.1
* [#975](https://github.com/xmrig/xmrig/issues/975) Fixed crash on Linux if double thread mode used.

# v2.14.0
- **[#969](https://github.com/xmrig/xmrig/pull/969) Added new algorithm `cryptonight/rwz`, short alias `cn/rwz` (also known as CryptoNight ReverseWaltz), for upcoming [Graft](https://www.graft.network/) fork.**
- **[#931](https://github.com/xmrig/xmrig/issues/931) Added new algorithm `cryptonight/zls`, short alias `cn/zls` for [Zelerius Network](https://zelerius.org) fork.**
- **[#940](https://github.com/xmrig/xmrig/issues/940) Added new algorithm `cryptonight/double`, short alias `cn/double` (also known as CryptoNight HeavyX), for [X-CASH](https://x-cash.org/).**
- [#951](https://github.com/xmrig/xmrig/issues/951#issuecomment-469581529) Fixed crash if AVX was disabled on OS level.
- [#952](https://github.com/xmrig/xmrig/issues/952) Fixed compile error on some Linux.
- [#957](https://github.com/xmrig/xmrig/issues/957#issuecomment-468890667) Added support for embedded config.
- [#958](https://github.com/xmrig/xmrig/pull/958) Fixed incorrect user agent on ARM platforms.
- [#968](https://github.com/xmrig/xmrig/pull/968) Optimized `cn/r` algorithm performance.

# v2.13.1
- [#946](https://github.com/xmrig/xmrig/pull/946) Optimized software AES implementations for CPUs without hardware AES support. `cn/r`, `cn/wow` up to 2.6 times faster, 4-9% improvements for other algorithms.

# v2.13.0
- **[#938](https://github.com/xmrig/xmrig/issues/938) Added support for new algorithm `cryptonight/r`, short alias `cn/r` (also known as CryptoNightR or CryptoNight variant 4), for upcoming [Monero](https://www.getmonero.org/) fork on March 9, thanks [@SChernykh](https://github.com/SChernykh).**
- [#939](https://github.com/xmrig/xmrig/issues/939) Added support for dynamic (runtime) pools reload.
- [#932](https://github.com/xmrig/xmrig/issues/932) Fixed `cn-pico` hashrate drop, regression since v2.11.0.

# v2.12.0
- [#929](https://github.com/xmrig/xmrig/pull/929) Added support for new algorithm `cryptonight/wow`, short alias `cn/wow` (also known as CryptonightR), for upcoming [Wownero](http://wownero.org) fork on February 14.

# v2.11.0
- [#928](https://github.com/xmrig/xmrig/issues/928) Added support for new algorithm `cryptonight/gpu`, short alias `cn/gpu` (original name `cryptonight-gpu`), for upcoming [Ryo currency](https://ryo-currency.com) fork on February 14.
- [#749](https://github.com/xmrig/xmrig/issues/749) Added support for detect hardware AES in runtime on ARMv8 platforms.
- [#292](https://github.com/xmrig/xmrig/issues/292) Fixed build on ARMv8 platforms if compiler not support hardware AES.

# v2.10.0
- [#904](https://github.com/xmrig/xmrig/issues/904) Added new algorithm `cn-pico/trtl` (aliases `cryptonight-turtle`, `cn-trtl`) for upcoming TurtleCoin (TRTL) fork.
- Default value for option `max-cpu-usage` changed to `100` also this option now deprecated.

# v2.9.4
- [#913](https://github.com/xmrig/xmrig/issues/913) Fixed Masari (MSR) support (this update required for upcoming fork).
- [#915](https://github.com/xmrig/xmrig/pull/915) Improved security, JIT memory now read-only after patching.

# v2.9.3
- [#909](https://github.com/xmrig/xmrig/issues/909) Fixed compile errors on FreeBSD.
- [#912](https://github.com/xmrig/xmrig/pull/912) Fixed, C++ implementation of `cn/half` was produce up to 13% of invalid hashes.

# v2.9.2
- [#907](https://github.com/xmrig/xmrig/pull/907) Fixed crash on Linux.

# v2.9.1
- Restored compatibility with https://stellite.hashvault.pro.

# v2.9.0
- [#899](https://github.com/xmrig/xmrig/issues/899) Added support for new algorithm `cn/half` for Masari and Stellite forks.
- [#834](https://github.com/xmrig/xmrig/pull/834) Added ASM optimized code for AMD Bulldozer.
- [#839](https://github.com/xmrig/xmrig/issues/839) Fixed FreeBSD compile.
- [#857](https://github.com/xmrig/xmrig/pull/857) Fixed impossible to build for macOS without clang.

# v2.8.3
- [#813](https://github.com/xmrig/xmrig/issues/813) Fixed critical bug with Minergate pool and variant 2.

# v2.8.1
- [#768](https://github.com/xmrig/xmrig/issues/768) Fixed build with Visual Studio 2015.
- [#769](https://github.com/xmrig/xmrig/issues/769) Fixed regression, some ANSI escape sequences was in log with disabled colors.
- [#777](https://github.com/xmrig/xmrig/issues/777) Better report about pool connection issues. 
- Simplified checks for ASM auto detection, only AES support necessary.
- Added missing options to `--help` output.

# v2.8.0
- **[#753](https://github.com/xmrig/xmrig/issues/753) Added new algorithm [CryptoNight variant 2](https://github.com/xmrig/xmrig/issues/753) for Monero fork, thanks [@SChernykh](https://github.com/SChernykh).**
  - Added global and per thread option `"asm"` and command line equivalent.
- **[#758](https://github.com/xmrig/xmrig/issues/758) Added SSL/TLS support for secure connections to pools.**
  - Added per pool options `"tls"` and `"tls-fingerprint"` and command line equivalents.
- [#767](https://github.com/xmrig/xmrig/issues/767) Added config autosave feature, same with GPU miners.  
- [#245](https://github.com/xmrig/xmrig-proxy/issues/245) Fixed API ID collision when run multiple miners on same machine.
- [#757](https://github.com/xmrig/xmrig/issues/757) Fixed send buffer overflow.

# v2.6.4
- [#700](https://github.com/xmrig/xmrig/issues/700) `cryptonight-lite/ipbc` replaced to `cryptonight-heavy/tube` for **Bittube (TUBE)**.
- Added `cryptonight/rto` (cryptonight variant 1 with IPBC/TUBE mod) variant for **Arto (RTO)** coin.
- Added `cryptonight/xao` (original cryptonight with bigger iteration count) variant for **Alloy (XAO)** coin.
- Better variant detection for **nicehash.com** and **minergate.com**.
- [#692](https://github.com/xmrig/xmrig/issues/692) Added support for specify both algorithm and variant via single `algo` option.

# v2.6.3
- **Added support for new cryptonight-heavy variant xhv** (`cn-heavy/xhv`) for upcoming Haven Protocol fork.
- **Added support for new cryptonight variant msr** (`cn/msr`) also known as `cryptonight-fast` for upcoming Masari fork.
- Added new detailed hashrate report.
- [#446](https://github.com/xmrig/xmrig/issues/446) Likely fixed SIGBUS error on 32 bit ARM CPUs.
- [#551](https://github.com/xmrig/xmrig/issues/551) Fixed `cn-heavy` algorithm on ARMv8.
- [#614](https://github.com/xmrig/xmrig/issues/614) Fixed display issue with huge pages percentage when colors disabled.
- [#615](https://github.com/xmrig/xmrig/issues/615) Fixed build without libcpuid.
- [#629](https://github.com/xmrig/xmrig/pull/629) Fixed file logging with non-seekable files.
- [#672](https://github.com/xmrig/xmrig/pull/672) Reverted back `cryptonight-light` and exit if no valid algorithm specified.

# v2.6.2
 - [#607](https://github.com/xmrig/xmrig/issues/607) Fixed donation bug.
 - [#610](https://github.com/xmrig/xmrig/issues/610) Fixed ARM build.

# v2.6.1
 - [#168](https://github.com/xmrig/xmrig-proxy/issues/168) Added support for [mining algorithm negotiation](https://github.com/xmrig/xmrig-proxy/blob/dev/doc/STRATUM_EXT.md#1-mining-algorithm-negotiation).
 - Added IPBC coin support, base algorithm `cn-lite` variant `ipbc`.
 - [#581](https://github.com/xmrig/xmrig/issues/581) Added support for upcoming Stellite (XTL) fork, base algorithm `cn` variant `xtl`, variant can set now, no need do it after fork.
 - Added support for **rig-id** stratum protocol extensions, compatible with xmr-stak.
 - Changed behavior for option `variant=-1` for `cryptonight`, now variant is `1` by default, if you mine old coins need change `variant` to `0`.
 - A lot of small fixes and better unification with proxy code.

# v2.6.0-beta3
- [#563](https://github.com/xmrig/xmrig/issues/563) **Added [advanced threads mode](https://github.com/xmrig/xmrig/issues/563), now possible configure each thread individually.**
- [#255](https://github.com/xmrig/xmrig/issues/563) Low power mode extended to **triple**, **quard** and **penta** modes.
- [#519](https://github.com/xmrig/xmrig/issues/519) Fixed high donation levels, improved donation start time randomization.
- [#554](https://github.com/xmrig/xmrig/issues/554) Fixed regression with `print-time` option.

# v2.6.0-beta2
- Improved performance for `cryptonight v7` especially in double hash mode.
- [#499](https://github.com/xmrig/xmrig/issues/499) IPv6 disabled for internal HTTP API by default, was causing issues on some systems.
- Added short aliases for algorithm names: `cn`, `cn-lite` and `cn-heavy`.
- Fixed regressions (v2.6.0-beta1 affected)
  - [#494](https://github.com/xmrig/xmrig/issues/494) Command line option `--donate-level` was broken.
  - [#502](https://github.com/xmrig/xmrig/issues/502) Build without libmicrohttpd was broken.
  - Fixed nonce calculation for `--av 4` (software AES, double hash) was causing reduction of effective hashrate and rejected shares on nicehash.

# v2.6.0-beta1
 - [#476](https://github.com/xmrig/xmrig/issues/476) **Added Cryptonight-Heavy support for Sumokoin ASIC resistance fork.**
 - HTTP server now runs in main loop, it make possible easy extend API without worry about thread synchronization.
 - Added initial graceful reload support, miner will reload configuration if config file changed, disabled by default until it will be fully implemented and tested.
 - Added API endpoint `PUT /1/config` to update current config.
 - Added API endpoint `GET /1/config` to get current active config.
 - Added API endpoint `GET /1/threads` to get current active threads configuration.
 - API endpoint `GET /` now deprecated, use `GET /1/summary` instead.
 - Added `--api-no-ipv6` and similar config option to disable IPv6 support for HTTP API.
 - Added `--api-no-restricted` to enable full access to api, this option has no effect if `--api-access-token` not specified.

# v2.5.3
- Fixed critical bug, in some cases miner was can't recovery connection and switch to failover pool, version 2.5.2 affected. If you use v2.6.0-beta3 this issue doesn't concern you.
- [#499](https://github.com/xmrig/xmrig/issues/499) IPv6 support disabled for internal HTTP API.
- Added workaround for nicehash.com if you use `cryptonightv7.<region>.nicehash.com` option `variant=1` will be set automatically.

# v2.5.2
- [#448](https://github.com/xmrig/xmrig/issues/478) Fixed broken reconnect.

# v2.5.1
- [#454](https://github.com/xmrig/xmrig/issues/454) Fixed build with libmicrohttpd version below v0.9.35.
- [#456](https://github.com/xmrig/xmrig/issues/459) Verbose errors related to donation pool was not fully silenced.
- [#459](https://github.com/xmrig/xmrig/issues/459) Fixed regression (version 2.5.0 affected) with connection to **xmr.f2pool.com**.

# v2.5.0
- [#434](https://github.com/xmrig/xmrig/issues/434) **Added support for Monero v7 PoW, scheduled on April 6.**
- Added full IPv6 support.
- Added protocol extension, when use the miner with xmrig-proxy 2.5+ no more need manually specify `nicehash` option.
- [#123](https://github.com/xmrig/xmrig-proxy/issues/123) Fixed regression (all versions since 2.4 affected) fragmented responses from pool/proxy was parsed incorrectly.
- [#428](https://github.com/xmrig/xmrig/issues/428) Fixed regression (version 2.4.5 affected) with CPU cache size detection.

# v2.4.5
- [#324](https://github.com/xmrig/xmrig/pull/324) Fixed build without libmicrohttpd (CMake cache issue).
- [#341](https://github.com/xmrig/xmrig/issues/341) Fixed wrong exit code and added command line option `--dry-run`.
- [#385](https://github.com/xmrig/xmrig/pull/385) Up to 20% performance increase for non-AES CPU and fixed Intel Core 2 cache detection.

# v2.4.4
 - Added libmicrohttpd version to --version output.
 - Fixed bug in singal handler, in some cases miner wasn't shutdown properly.
 - Fixed recent MSVC 2017 version detection.
 - [#279](https://github.com/xmrig/xmrig/pull/279) Fixed build on some macOS versions.

# v2.4.3
 - [#94](https://github.com/xmrig/xmrig/issues/94#issuecomment-342019257) [#216](https://github.com/xmrig/xmrig/issues/216) Added **ARMv8** and **ARMv7** support. Hardware AES supported, thanks [Imran Yusuff](https://github.com/imranyusuff).
 - [#157](https://github.com/xmrig/xmrig/issues/157) [#196](https://github.com/xmrig/xmrig/issues/196) Fixed Linux compile issues.
 - [#184](https://github.com/xmrig/xmrig/issues/184) Fixed cache size detection for CPUs with disabled Hyper-Threading.
 - [#200](https://github.com/xmrig/xmrig/issues/200) In some cases miner was doesn't write log to stdout.

# v2.4.2
 - [#60](https://github.com/xmrig/xmrig/issues/60) Added FreeBSD support, thanks [vcambur](https://github.com/vcambur).
 - [#153](https://github.com/xmrig/xmrig/issues/153) Fixed issues with dwarfpool.com.
 
# v2.4.1
  - [#147](https://github.com/xmrig/xmrig/issues/147) Fixed comparability with monero-stratum.

# v2.4.0
 - Added [HTTP API](https://github.com/xmrig/xmrig/wiki/API).
 - Added comments support in config file.
 - libjansson replaced to rapidjson.
 - [#98](https://github.com/xmrig/xmrig/issues/98) Ignore `keepalive` option with minergate.com and nicehash.com.
 - [#101](https://github.com/xmrig/xmrig/issues/101) Fixed MSVC 2017 (15.3) compile time version detection.
 - [#108](https://github.com/xmrig/xmrig/issues/108) Silently ignore invalid values for `donate-level` option.
 - [#111](https://github.com/xmrig/xmrig/issues/111) Fixed build without AEON support.
 
# v2.3.1
- [#68](https://github.com/xmrig/xmrig/issues/68) Fixed compatibility with Docker containers, was nothing print on console.

# v2.3.0
- Added `--cpu-priority` option (0 idle, 2 normal to 5 highest).
- Added `--user-agent` option, to set custom user-agent string for pool. For example `cpuminer-multi/0.1`.
- Added `--no-huge-pages` option, to disable huge pages support.
- [#62](https://github.com/xmrig/xmrig/issues/62) Don't send the login to the dev pool.
- Force reconnect if pool block miner IP address. helps switch to backup pool.
- Fixed: failed open default config file if path contains non English characters.
- Fixed: error occurred if try use unavailable stdin or stdout, regression since version 2.2.0.
- Fixed: message about huge pages support successfully enabled on Windows was not shown in release builds.

# v2.2.1
- Fixed [terminal issues](https://github.com/xmrig/xmrig-proxy/issues/2#issuecomment-319914085) after exit on Linux and OS X.

# v2.2.0
- [#46](https://github.com/xmrig/xmrig/issues/46) Restored config file support. Now possible use multiple config files and combine with command line options also added support for default config.
- Improved colors support on Windows, now used uv_tty, legacy code removed.
- QuickEdit Mode now disabled on Windows.
- Added interactive commands in console window:: **h**ashrate, **p**ause, **r**esume.
- Fixed autoconf mode for AMD FX CPUs.

# v2.1.0
- [#40](https://github.com/xmrig/xmrig/issues/40)
Improved miner shutdown, fixed crash on exit for Linux and OS X.
- Fixed, login request was contain malformed JSON if username or password has some special characters for example `\`. 
- [#220](https://github.com/fireice-uk/xmr-stak-cpu/pull/220) Better support for Round Robin DNS, IP address now always chosen randomly instead of stuck on first one.
- Changed donation address, new [xmrig-proxy](https://github.com/xmrig/xmrig-proxy) is coming soon.

# v2.0.2
- Better deal with possible duplicate jobs from pool, show warning and ignore duplicates.
- For Windows builds libuv updated to version 1.13.1 and gcc to 7.1.0.

# v2.0.1
 - [#27](https://github.com/xmrig/xmrig/issues/27) Fixed possibility crash on 32bit systems.

# v2.0.0
 - Option `--backup-url` removed, instead now possibility specify multiple pools for example: `-o example1.com:3333 -u user1 -p password1 -k -o example2.com:5555 -u user2 -o example3.com:4444 -u user3`
 - [#15](https://github.com/xmrig/xmrig/issues/15) Added option `-l, --log-file=FILE` to write log to file.
 - [#15](https://github.com/xmrig/xmrig/issues/15) Added option `-S, --syslog` to use syslog for logging, Linux only.
 - [#18](https://github.com/xmrig/xmrig/issues/18) Added nice messages for accepted/rejected shares with diff and network latency.
 - [#20](https://github.com/xmrig/xmrig/issues/20) Fixed `--cpu-affinity` for more than 32 threads.
 - Fixed Windows XP support.
 - Fixed regression, option `--no-color` was not fully disable colored output.
 - Show resolved pool IP address in miner output.
 
# v1.0.1
- Fix broken software AES implementation, app has crashed if CPU not support AES-NI, only version 1.0.0 affected.

# v1.0.0
- Miner complete rewritten in C++ with libuv.
- This version should be fully compatible (except config file) with previos versions, many new nice features will come in next versions.
- This is still beta. If you found regression, stability or perfomance issues or have an idea for new feature please fell free to open new [issue](https://github.com/xmrig/xmrig/issues/new).
- Added new option `--print-time=N`, print hashrate report every N seconds.
- New hashrate reports, by default every 60 secons.
- Added Microsoft Visual C++ 2015 and 2017 support.
- Removed dependency on libcurl.
- To compile this version from source please switch to [dev](https://github.com/xmrig/xmrig/tree/dev) branch.

# v0.8.2
- Fixed L2 cache size detection for AMD CPUs (Bulldozer/Piledriver/Steamroller/Excavator architecture).

# v0.8.2
- Fixed L2 cache size detection for AMD CPUs (Bulldozer/Piledriver/Steamroller/Excavator architecture).
- Fixed gcc 7.1 support.

# v0.8.1
- Added nicehash support, detects automaticaly by pool URL, for example `cryptonight.eu.nicehash.com:3355` or manually via option `--nicehash`.

# v0.8.0
- Added double hash mode, also known as lower power mode. `--av=2` and `--av=4`.
- Added smart automatic CPU configuration. Default threads count now depends on size of the L3 cache of CPU.
- Added CryptoNight-Lite support for AEON `-a cryptonight-lite`.
- Added `--max-cpu-usage` option for auto CPU configuration mode.
- Added `--safe` option for adjust threads and algorithm variations to current CPU.
- No more manual steps to enable huge pages on Windows. XMRig will do it automatically.
- Removed BMI2 algorithm variation.
- Removed default pool URL.

# v0.6.0
- Added automatic cryptonight self test.
- New software AES algorithm variation. Will be automatically selected if cpu not support AES-NI.
- Added 32 bit builds.
- Documented [algorithm variations](https://github.com/xmrig/xmrig#algorithm-variations).

# v0.5.0
- Initial public release.
