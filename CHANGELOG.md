# v4.3.0-beta
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

# Previous versions
[doc/CHANGELOG_OLD.md](doc/CHANGELOG_OLD.md)
