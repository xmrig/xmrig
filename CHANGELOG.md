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
