# v1.0.0
- Miner complete rewritten in C++ with libuv.
- This version should be fully compatible (except config file) with previos versions, many new nice features will come in next versions.
- This is still beta. If you found regression, stability or perfomance issues or have an idea for new feature please fell free to open new [issue](https://github.com/xmrig/xmrig/issues/new).
- Added new option `--print-time=N`, print hashrate report every N seconds.
- New hashrate reports, by default every 60 secons.
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
