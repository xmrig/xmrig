# XMRig

:warning: **If you mine Monero, Aeon, Sumokoin, Turtlecoin, Stellite, GRAFT, Haven Protocol, IPBC, [PLEASE READ](https://github.com/xmrig/xmrig/issues/482)!** :warning:

[![Github All Releases](https://img.shields.io/github/downloads/xmrig/xmrig/total.svg)](https://github.com/xmrig/xmrig/releases)
[![GitHub release](https://img.shields.io/github/release/xmrig/xmrig/all.svg)](https://github.com/xmrig/xmrig/releases)
[![GitHub Release Date](https://img.shields.io/github/release-date-pre/xmrig/xmrig.svg)](https://github.com/xmrig/xmrig/releases)
[![GitHub license](https://img.shields.io/github/license/xmrig/xmrig.svg)](https://github.com/xmrig/xmrig/blob/master/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/xmrig/xmrig.svg)](https://github.com/xmrig/xmrig/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/xmrig/xmrig.svg)](https://github.com/xmrig/xmrig/network)

XMRig is a high performance Monero (XMR) CPU miner, with official support for Windows.
Originally based on cpuminer-multi with heavy optimizations/rewrites and removing a lot of legacy code, since version 1.0.0 completely rewritten from scratch on C++.

* This is the **CPU-mining** version, there is also a [NVIDIA GPU version](https://github.com/xmrig/xmrig-nvidia) and [AMD GPU version]( https://github.com/xmrig/xmrig-amd).
* [Roadmap](https://github.com/xmrig/xmrig/issues/106) for next releases.

<img src="http://i.imgur.com/OKZRVDh.png" width="619" >

#### Table of contents
* [Features](#features)
* [Download](#download)
* [Usage](#usage)
* [Algorithm variations](#algorithm-variations)
* [Build](https://github.com/xmrig/xmrig/wiki/Build)
* [Common Issues](#common-issues)
* [Other information](#other-information)
* [Donations](#donations)
* [Release checksums](#release-checksums)
* [Contacts](#contacts)

## Features
* High performance.
* Official Windows support.
* Small Windows executable, without dependencies.
* x86/x64 support.
* Support for backup (failover) mining server.
* keepalived support.
* Command line options compatible with cpuminer.
* CryptoNight-Lite support for AEON.
* Smart automatic [CPU configuration](https://github.com/xmrig/xmrig/wiki/Threads).
* Nicehash support
* It's open source software.

## Download
* Binary releases: https://github.com/xmrig/xmrig/releases
* Git tree: https://github.com/xmrig/xmrig.git
  * Clone with `git clone https://github.com/xmrig/xmrig.git` :hammer: [Build instructions](https://github.com/xmrig/xmrig/wiki/Build).

## Usage
Use [config.xmrig.com](https://config.xmrig.com/xmrig) to generate, edit or share configurations.

### Options
```
  -a, --algo=ALGO          cryptonight (default) or cryptonight-lite
  -o, --url=URL            URL of mining server
  -O, --userpass=U:P       username:password pair for mining server
  -u, --user=USERNAME      username for mining server
  -p, --pass=PASSWORD      password for mining server
  -t, --threads=N          number of miner threads
  -v, --av=N               algorithm variation, 0 auto select
  -k, --keepalive          send keepalived for prevent timeout (need pool support)
  -r, --retries=N          number of times to retry before switch to backup server (default: 5)
  -R, --retry-pause=N      time to pause between retries (default: 5)
      --cpu-affinity       set process affinity to CPU core(s), mask 0x3 for cores 0 and 1
      --cpu-priority       set process priority (0 idle, 2 normal to 5 highest)
      --no-huge-pages      disable huge pages support
      --no-color           disable colored output
      --variant            algorithm PoW variant
      --donate-level=N     donate level, default 5% (5 minutes in 100 minutes)
      --user-agent         set custom user-agent string for pool
  -B, --background         run the miner in the background
  -c, --config=FILE        load a JSON-format configuration file
  -l, --log-file=FILE      log all output to a file
  -S, --syslog             use system log for output messages
      --max-cpu-usage=N    maximum CPU usage for automatic threads mode (default 75)
      --safe               safe adjust threads and av settings for current CPU
      --nicehash           enable nicehash/xmrig-proxy support
      --print-time=N       print hashrate report every N seconds
      --api-port=N         port for the miner API
      --api-access-token=T access token for API
      --api-worker-id=ID   custom worker-id for API
  -h, --help               display this help and exit
  -V, --version            output version information and exit
```

Also you can use configuration via config file, default **config.json**. You can load multiple config files and combine it with command line options.

## Algorithm variations
Since version 0.8.0.
* `--av=1` For CPUs with hardware AES.
* `--av=2` Lower power mode (double hash) of `1`.
* `--av=3` Software AES implementation.
* `--av=4` Lower power mode (double hash) of `3`.

## Common Issues
### HUGE PAGES unavailable
* Run XMRig as Administrator.
* Since version 0.8.0 XMRig automatically enables SeLockMemoryPrivilege for current user, but reboot or sign out still required. [Manual instruction](https://msdn.microsoft.com/en-gb/library/ms190730.aspx).

## Other information
* No HTTP support, only stratum protocol support.
* No TLS support.
* Default donation 5% (5 minutes in 100 minutes) can be reduced to 1% via command line option `--donate-level`.


### CPU mining performance
* **Intel i7-7700** - 307 H/s (4 threads)
* **AMD Ryzen 7 1700X** - 560 H/s (8 threads)

Please note performance is highly dependent on system load. The numbers above are obtained on an idle system. Tasks heavily using a processor cache, such as video playback, can greatly degrade hashrate. Optimal number of threads depends on the size of the L3 cache of a processor, 1 thread requires 2 MB of cache.

### Maximum performance checklist
* Idle operating system.
* Do not exceed optimal thread count.
* Use modern CPUs with AES-NI instruction set.
* Try setup optimal cpu affinity.
* Enable fast memory (Large/Huge pages).

## Donations
* XMR: `48edfHu7V9Z84YzzMa6fUueoELZ9ZRXq9VetWzYGzKt52XU5xvqgzYnDK9URnRoJMk1j8nLwEVsaSWJ4fhdUyZijBGUicoD`
* BTC: `1P7ujsXeX7GxQwHNnJsRMgAdNkFZmNVqJT`

## Release checksums
### SHA-256
```
f8e1957e8bfd7f281a76d1e42694049c67f39dea90ac36e9d589c14cdf8924bc xmrig-2.6.1-xenial-amd64.tar.gz/xmrig-2.6.1/xmrig
472c7aaf5aacc1212bfd3f2f96daca4f42d64e2d0db0872891328e7d8503d0c8 xmrig-2.6.1-gcc-win32.zip/xmrig.exe
d53154cef24c884b2be539ac13bfb6e7dba6bbc53b62e91f2877637d43fa4b15 xmrig-2.6.1-gcc-win64.zip/xmrig.exe
a253381b617463e6e1193d49b8afbf720a1c376621da7429d97f192668cd59ad xmrig-2.6.1-msvc-win64.zip/xmrig.exe
```

## Contacts
* support@xmrig.com
* [reddit](https://www.reddit.com/user/XMRig/)
