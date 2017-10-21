# XMRigCC

[Build instructions / Notes / Howto](https://raw.githubusercontent.com/Bendr0id/xmrigCC/feature_cc/howto_and_notes.txt)

Documentation will be added here soon.

##### About

XMRig is high performance Monero (XMR) / Aeon CPU miner, with the official full Windows support.
Originally based on cpuminer-multi with heavy optimizations/rewrites and removing a lot of legacy code, since version 1.0.0 complete rewritten from scratch on C++.

* This is the CPU-mining version, there is also a [NVIDIA GPU version](https://github.com/xmrig/xmrig-nvidia).

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
* [Contacts](#contacts)

## Additional features of xmrigCC
* Command and control server
* CC Dashboard with:
    * statistics 
    * remote control (start/stop/restart/shutdown) miner
    * remote configuration changes of miner
* Added daemon to restart miner

\+ All features of the latest xmrig

## Features of xmrig
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
* Binary releases: soon
* Git tree: https://github.com/Bendr0id/xmrigCC.git
  * Clone with `git clone https://github.com/Bendr0id/xmrigCC.git` :hammer: [Build instructions / Notes / Howto](https://raw.githubusercontent.com/Bendr0id/xmrigCC/feature_cc/howto_and_notes.txt).

## Usage
### Basic example
```
xmrig.exe -o pool.minemonero.pro:5555 -u YOUR_WALLET -p x -k
```

### Failover
```
xmrig.exe -o pool.minemonero.pro:5555 -u YOUR_WALLET1 -p x -k -o pool.supportxmr.com:5555 -u YOUR_WALLET2 -p x -k
```
For failover you can add multiple pools, maximum count not limited.

### Options
```
  -a, --algo=ALGO       cryptonight (default) or cryptonight-lite
  -o, --url=URL         URL of mining server
  -O, --userpass=U:P    username:password pair for mining server
  -u, --user=USERNAME   username for mining server
  -p, --pass=PASSWORD   password for mining server
  -t, --threads=N       number of miner threads
  -v, --av=N            algorithm variation, 0 auto select
  -k, --keepalive       send keepalived for prevent timeout (need pool support)
  -r, --retries=N       number of times to retry before switch to backup server (default: 5)
  -R, --retry-pause=N   time to pause between retries (default: 5)
      --cpu-affinity    set process affinity to CPU core(s), mask 0x3 for cores 0 and 1
      --cpu-priority    set process priority (0 idle, 2 normal to 5 highest)
      --no-huge-pages   disable huge pages support
      --no-color        disable colored output
      --donate-level=N  donate level, default 5% (5 minutes in 100 minutes)
      --user-agent      set custom user-agent string for pool
  -B, --background      run the miner in the background
  -c, --config=FILE     load a JSON-format configuration file
  -l, --log-file=FILE   log all output to a file
      --max-cpu-usage=N maximum CPU usage for automatic threads mode (default 75)
      --safe            safe adjust threads and av settings for current CPU
      --nicehash        enable nicehash support
      --print-time=N    print hashrate report every N seconds
  -h, --help            display this help and exit
  -V, --version         output version information and exit
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
* Since version 0.8.0 XMRig automatically enable SeLockMemoryPrivilege for current user, but reboot or sign out still required. [Manual instruction](https://msdn.microsoft.com/en-gb/library/ms190730.aspx).

## Other information
* No HTTP support, only stratum protocol support.
* No TLS support.


### CPU mining performance
* **Intel i7-7700** - 307 H/s (4 threads)
* **AMD Ryzen 7 1700X** - 560 H/s (8 threads)

Please note performance is highly dependent on system load. The numbers above are obtained on an idle system. Tasks heavily using a processor cache, such as video playback, can greatly degrade hashrate. Optimal number of threads depends on the size of the L3 cache of a processor, 1 thread requires 2 MB of cache.

### Maximum performance checklist
* Idle operating system.
* Do not exceed optimal thread count.
* Use modern CPUs with AES-NI instructuon set.
* Try setup optimal cpu affinity.
* Enable fast memory (Large/Huge pages).

## Donations
* Default donation 5% (5 minutes in 100 minutes) can be reduced to 1% via command line option `--donate-level`. 
* The donations will be automatically splitted (50/50) on my pool between xmrig and myself.

##### BenDroid (xmrigCC):
* XMR: `46FkYo7x6LqYjLQo4Jd84UTGBybW7tsWqJaQVLPhbUSK19ajSTMY9T2Sa2LH6CfWhSingjvQARtfeM4Feekpp2yFR1wsFNT`
* AEON: `Wmtm4S2cQ8uEBBAVjvbiaVAPv2d6gA1mAUmBmjna4VF7VixLxLRUYag5cvsym3WnuzdJ9zvhQ3Xwa8gWxPDPRfcQ3AUkYra3W`
* BTC: `128qLZCaGdoWhBTfaS7rytpbvG4mNTyAQm`

##### xmrig:
* XMR: `48edfHu7V9Z84YzzMa6fUueoELZ9ZRXq9VetWzYGzKt52XU5xvqgzYnDK9URnRoJMk1j8nLwEVsaSWJ4fhdUyZijBGUicoD`
* BTC: `1P7ujsXeX7GxQwHNnJsRMgAdNkFZmNVqJT`

## Contacts xmrigCC
* ben [at] graef.in
* [reddit](https://www.reddit.com/user/BenDr0id/)

## Contacts xmrig
* support@xmrig.com
* [reddit](https://www.reddit.com/user/XMRig/)
