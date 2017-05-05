# v0.7.0
- Added double hash mode, also known as lower power mode. `--av=2` and `--av=5`.
- Default threads count now depends on size of the L3 cache of CPU.
- No more manual steps to enable huge pages on Windows. XMRig will do it automatically.

# v0.6.0
- Added automatic cryptonight self test.
- New software AES algorithm variation `--av=4`. Will be automatically selected if cpu not support AES-NI.
- Added 32 bit builds.
- Documented [algorithm variations](https://github.com/xmrig/xmrig#algorithm-variations).

# v0.5.0
- Initial public release.
