# NinjaRig v1.0
### Argon2 miner for CPU and GPU

## Dev Fee
In order to support development, this miner has 1-5% configurable dev fee - 1-5 minutes from 100 minutes it will mine for developer. Mining settings are downloaded from http://coinfee.changeling.biz/index.json at startup.

## Features
- optimized argon2 hashing library - both in speed and in memory usage; everything not related to actual mining was stripped down, indexing calculation for argon2i and argon2id sequence was replaced with precalculated versions
- support for both CPU and GPU mining using multiple engines perfectly adapted to your hardware
- support for autodetecting the best version of the CPU hasher for your machine (SSE2/SSSE3/AVX/AVX2/AVX512F)

## Releases
There are binaries compiled for Windows 10 and Linux/HiveOS. Just pick the one matching your OS and skip to usage information. If for some reason the binaries don't work for you or you want the cutting edge version of this software you can try building it yourself using below instructions (build instructions are only provided for Ubuntu, you will need to adapt them accordingly for other distribution).
You can get the binaries from here:
https://github.com/bogdanadnan/ninjarig/releases

## Build it yourself
What you need:
- Recent Linux distribution (recommended - Ubuntu 16.04 or higher)
- Git client
- CMake 3
- GCC & G++ version 7 or higher or LLVM/Clang 7 or higher. Provided binaries are compiled with Clang 8, it seems to give a slightly higher hashrate for CPU mining.
- CUDA developer toolkit 9 or higher. Provided binaries are compiled with CUDA 10.1. Follow instructions from NVidia site to get the latest version up and running: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html (be careful that CUDA might have specific requirements for compiler version as well)
- OpenCL libraries and headers
- OpenSSL, libuv and microhttpd libraries and headers

Instructions:
- run the following snippet:
```sh
$ git clone http://github.com/bogdanadnan/ninjarig.git
$ cd ninjarig
$ mkdir build
$ cd build
$ cmake .. -DCMAKE_BUILD_TYPE=Release
$ make
```

## Basic usage:
**!!! In some cases (mostly on Windows) the miner doesn't properly detect AVX2 optimization for CPU. If AVX2 doesn't appear in optimization features list for CPU at miner startup, please verify on google if your CPU model has it. If it does have AVX2 support, please run it with "--cpu-optimization AVX2" option. This will give a serious boost to hash rate speed so it does worth the effort to check. !!!**

```sh
       ninjarig -a chukwa -o stratum+tcp://<pool_address>:<pool_port> -u <username> -p <password> -t <cpu_threads> --use-gpu <OPENCL,CUDA> --gpu-filter <filters like: OPENCL:AMD,CUDA:1070> --gpu-intensity <intensity from 1-100>
```

### Options
```
  -a, --algo=ALGO          specify the algorithm to use
                             chukwa
                             chukwa/wrkz
  -o, --url=URL            URL of mining server
  -O, --userpass=U:P       username:password pair for mining server
  -u, --user=USERNAME      username for mining server
  -p, --pass=PASSWORD      password for mining server
      --rig-id=ID          rig identifier for pool-side statistics (needs pool support)
  -t, --cpu-threads=N      number of cpu mining threads
      --cpu-affinity       set process affinity to CPU core(s), mask 0x3 for cores 0 and 1
      --cpu-optimization=REF|SSE2|SSSE3|AVX|AVX2|AVX512F|NEON force specific optimization for cpu mining
      --use-gpu=CUDA,OPENCL       gpu engine to use, ignore this param to disable gpu support
      --gpu-intensity=v1,v2...    percent of gpu memory to use - you can have different values for each card (default 50)
      --gpu-filter=<filter1>,CUDA:<filter2>,OPENCL:<filter3>  gpu filters to select cards
   -k, --keepalive          send keepalived packet for prevent timeout (needs pool support)
      --nicehash           enable nicehash.com support
      --tls                enable SSL/TLS support (needs pool support)
      --tls-fingerprint=F  pool TLS certificate fingerprint, if set enable strict certificate pinning
  -r, --retries=N          number of times to retry before switch to backup server (default: 5)
  -R, --retry-pause=N      time to pause between retries (default: 5)
      --priority           set process priority (0 idle, 2 normal to 5 highest)
      --no-color           disable colored output
      --variant            algorithm PoW variant
      --donate-level=N     donate level, default 5% (5 minutes in 100 minutes)
      --user-agent         set custom user-agent string for pool
  -B, --background         run the miner in the background
  -c, --config=FILE        load a JSON-format configuration file
  -l, --log-file=FILE      log all output to a file
  -S, --syslog             use system log for output messages
      --print-time=N       print hashrate report every N seconds
      --api-port=N         port for the miner API
      --api-access-token=T access token for API
      --api-worker-id=ID   custom worker-id for API
      --api-id=ID          custom instance ID for API
      --api-ipv6           enable IPv6 support for API
      --api-no-restricted  enable full remote access (only if API token set)
      --dry-run            test configuration and exit
  -h, --help               display this help and exit
  -V, --version            output version information and exit
```

Also you can use configuration via config file, default name **config.json**.

## Other information
* No HTTP support, only stratum protocol support.
* Default donation 5% (5 minutes in 100 minutes) can be reduced to 1% via option `donate-level`.
