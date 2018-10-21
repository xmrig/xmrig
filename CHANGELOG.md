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
  - Added global and per thread option `"asm"` and and command line equivalent.
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
