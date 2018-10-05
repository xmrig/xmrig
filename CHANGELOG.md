# v0.9.0
- **[#753](https://github.com/xmrig/xmrig/issues/753) Added new algorithm [CryptoNight variant 2](https://github.com/xmrig/xmrig/issues/753) for Monero fork, thanks [@SChernykh](https://github.com/SChernykh).**
  - Added option `--asm`, possible values `--asm auto`, `--asm none`, `--asm intel` and `--asm ryzen`.
- Added support for new style long and short algorithm names, possible values: `cryptonight`, `cryptonight/0`, `cryptonight/1`, `cryptonight/2`, `cryptonight-lite`, `cryptonight-lite/0`, `cryptonight-lite/1` and short equvalents `cn/2` etc. 
- Added `--variant`, example `--algo cn --variant 2`, by default miner automaticaly detect proper variant for Monero by block version.  
- Added CryptoNight-Lite variant 1.
- Added xmrig-proxy autodetection, nicehash will be enabled automaticaly. 
- Added workaround for xmrig-proxy [bug](https://github.com/xmrig/xmrig-proxy/commit/dfa1960fe3eeb13f80717b7dbfcc7c6e9f222d89).

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
