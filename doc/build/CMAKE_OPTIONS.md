# CMake options
This document contains list of useful cmake options.

## Algorithms

* **`-DWITH_CN_LITE=OFF`** disable all CryptoNight-Lite algorithms (`cn-lite/0`, `cn-lite/1`).
* **`-DWITH_CN_HEAVY=OFF`** disable all CryptoNight-Heavy algorithms (`cn-heavy/0`, `cn-heavy/xhv`, `cn-heavy/tube`).
* **`-DWITH_CN_PICO=OFF`** disable CryptoNight-Pico algorithm (`cn-pico`).
* **`-DWITH_CN_GPU=OFF`** disable CryptoNight-GPU algorithm (`cn/gpu`).
* **`-DWITH_RANDOMX=OFF`** disable RandomX algorithms (`rx/loki`, `rx/wow`).
* **`-DWITH_ARGON2=OFF`** disable Argon2 algorithms (`argon2/chukwa`, `argon2/wrkz`).

## Features

* **`-DWITH_HWLOC=OFF`**
disable [hwloc](https://github.com/xmrig/xmrig/issues/1077) support.
Disabling this feature is not recommended in most cases.
This feature add external dependency to libhwloc (1.10.0+) (except MSVC builds).
* **`-DWITH_LIBCPUID=OFF`** disable built in libcpuid support, this feature always disabled if hwloc enabled, if both hwloc and libcpuid disabled auto configuration for CPU will very limited.
* **`-DWITH_HTTP=OFF`** disable built in HTTP support, this feature used for HTTP API and daemon (solo mining) support.
* **`-DWITH_TLS=OFF`** disable SSL/TLS support (secure connections to pool). This feature add external dependency to OpenSSL.
* **`-DWITH_ASM=OFF`** disable assembly optimizations for modern CryptoNight algorithms.
* **`-DWITH_EMBEDDED_CONFIG=ON`** Enable [embedded](https://github.com/xmrig/xmrig/issues/957) config support.
