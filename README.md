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
https://github.com/turtlecoin/ninjarig/releases

## Build it yourself - Linux
What you need:
- Recent Linux distribution (recommended - Ubuntu 16.04 or higher)
- Git client
- CMake 3
- GCC & G++ version 7 or higher or LLVM/Clang 7 or higher. Provided binaries are compiled with Clang 8, it seems to give a slightly higher hashrate for CPU mining.
- CUDA developer toolkit 9 or higher. Provided binaries are compiled with CUDA 10.1. Follow instructions from NVidia site to get the latest version up and running: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html (be careful that CUDA might have specific requirements for compiler version as well)
- OpenCL libraries and headers (package ocl-icd-opencl-dev in Ubuntu/Debian)
- OpenSSL, libuv and microhttpd libraries and headers

Instructions:
- run the following snippet taking the parts specific for your system (Ubuntu 16.04 or 18.04):
```sh
# Install dependences 16.04/ Stretch
$ sudo apt-get install git build-essential cmake libuv1-dev libmicrohttpd-dev libssl-dev

# 18.04/ Buster
$ sudo apt-get install git build-essential cmake libuv1-dev libmicrohttpd-dev libssl-dev gcc-8 g++-8
$ export CC=gcc-8
$ export CXX=g++-8

# Clone Repository
$ git clone https://github.com/turtlecoin/ninjarig.git && cd ninjarig

# Make Build Repository
$ mkdir build && cd build

# For CPU Only
$ cmake -DWITH_CUDA=OFF -DWITH_OPENCL=OFF .. -DCMAKE_BUILD_TYPE=RELEASE

# For CPU and OpenCL
$ cmake -DWITH_CUDA=OFF .. -DCMAKE_BUILD_TYPE=RELEASE

# For CPU and CUDA
$ cmake -DWITH_OPENCL=OFF .. -DCMAKE_BUILD_TYPE=RELEASE

# For CPU, OpenCL, and CUDA
$ cmake .. -DCMAKE_BUILD_TYPE=RELEASE
$ make
```

## Build it yourself - Windows
Compiling NinjaRig for Windows is not for the faint-hearted. It needs a mixed compilation environment, becuase most of the code only compiles with clang/gcc while CUDA is MSVC specific (Visual Studio C/C++ compiler). The binaries made by those 2 compilers needs to be compatible as well, which only leaves clang-cl and msvc combination as a valid one. There are a lot of issues also finding proper binaries for dependencies or compiling them yourself.
So, if you really really really want to do that, these are the basic steps you will have to do:
- install CLang environment for Windows - x64 version (http://releases.llvm.org/download.html)
- install a Git client (https://git-scm.com/downloads)
- install CMake 3 for Windows (https://cmake.org/download/)
- install Make for Windows (http://gnuwin32.sourceforge.net/packages/make.htm) or install MinGW. In both cases add the binary to PATH env variable.
- install Visual Studio 2017 - Community edition should suffice, be sure to install C/C++ support and Windows SDK (https://visualstudio.microsoft.com/vs/community/) 
- install NVidia CUDA Toolkit (https://developer.nvidia.com/cuda-downloads)
- search for binaries or compile yourself the following libraries: OpenSSL (https://wiki.openssl.org/index.php/Binaries), libuv (https://github.com/libuv/libuv & https://github.com/WenbinHou/libuv-prebuilt) and microhttpd (https://www.gnu.org/software/libmicrohttpd/). If you don't need TLS and API support, you can ignore OpenSSL and microhttpd dependencies but libuv is mandatory. For each you will need the DLL file, the .LIB - for Visual Studio (not the .A! - for MinGW/gcc) file and the headers. Find the place where Visual Studio has installed the SDK and copy those files in the appropriate lib/include folders. The DLLs will be needed in the final compilation folder.
- create a folder named ninja somewhere on your drive and add 2 subfolders in it, one named build_clang and another build_vc
- now comes the fun part. In the Windows Run menu type "x64 Native Tools Command Prompt" (https://docs.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=vs-2019) - and open x64 build console.
- test that all dependencies work. You should be able to run in console clang-cl --version, make --version, cmake --version and git --version and get results. 
- in the CMD window opened, navigate to ninja folder.
- type the following commands:
```sh
$ git clone https://github.com/turtlecoin/ninjarig.git
$ cd build_clang
$ set CC=clang-cl
$ set CXX=clang-cl
$ cmake ../ninjarig -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=OFF -G "MinGW Makefiles"
$ cd ../build_vc
$ cmake ../ninjarig -DCMAKE_BUILD_TYPE=Release -G "Visual Studio 15 2017 Win64"
$ cd ../build_clang
$ make
```
- this will do 3 things. First cmake call will create a Makefile for building NinjaRig with clang-cl without CUDA which is not compatible with Clang (!!clang-cl is a LLVM driver compatible with MSVC; don't use clang without cl suffix as the results won't be compatible with visual studio compiled CUDA module!!). Second cmake call will create Visual Studio project files for compiling NinjaRig with Visual Studio - only parts of the code will actually be compilable - CUDA module is fortunately one of them. And the last make call will compile the clang version of the miner. This last step will take a while and spill out LOTS of warnings. As long as there is no error you can ignore the warnings. At this step if you don't need TLS or API you can add -DWITH_TLS=OFF and -DWITH_HTTPD=OFF to both cmake commands.
- copy OpenSSL, libuv and microhttpd DLLs to build_clang folder.
- at this point you should have a working NinjaRig executable without CUDA support.
- open Visual Studio and from there NinjaRig.sln file from build_vc folder. Go to Batch Build menu, search for cuda_hasher/Release in the list and check it, leaving everything else unchecked. Press Build button. This should build the cuda_hasher module. In build_vc folder, go to modules/Release folder and copy the cuda_hasher.hsh file to build_clang/modules folder.
- open the champagne bottle you bought for this specific moment. You have a windows compiled version of NinjaRig :)

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
