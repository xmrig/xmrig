# XMRigCC 

:warning: **Confused by all the forks? Check the [Coin Configuration](https://github.com/Bendr0id/xmrigCC/wiki/Coin-configurations) guide.**


[![Windows Build status](https://ci.appveyor.com/api/projects/status/l8v7cuuy320a4tpd?svg=true)](https://ci.appveyor.com/project/Bendr0id/xmrigcc)
[![Docker Build status](https://img.shields.io/docker/build/bendr0id/xmrigcc.svg)](https://hub.docker.com/r/bendr0id/xmrigcc/)
[![GitHub release](https://img.shields.io/github/release/bendr0id/xmrigCC/all.svg)](https://github.com/bendr0id/xmrigCC/releases)
[![Github downloads latest](https://img.shields.io/github/downloads/bendr0id/xmrigCC/latest/total.svg)](https://github.com/bendr0id/xmrigCC/releases)
[![Github downloads total](https://img.shields.io/github/downloads/bendr0id/xmrigCC/total.svg)](https://github.com/bendr0id/xmrigCC/releases)
[![GitHub stars](https://img.shields.io/github/stars/bendr0id/xmrigCC.svg)](https://github.com/bendr0id/xmrigCC/stargazers)

![XMRigCC Logo](https://i.imgur.com/7mi0WCe.png)

### About XMRigCC

XMRigCC is a fork of [XMRig](https://github.com/xmrig/xmrig) which adds the ability to remote control your XMRig instances via a Webfrontend and REST api.
This fork is based on XMRig and adds a "Command and Control" (C&amp;C) server, a daemon to reload XMRigCCMiner on config changes and modifications in XMRig to send the current status to the C&amp;C Server.
The modified version can also handle commands like "update config", "start/stop mining" or "restart/shutdown" which can be send from the C&amp;C-Server.

**AND MANY MORE**

Full Windows/Linux compatible, and you can mix Linux and Windows miner on one XMRigCCServer.

## Additional features of XMRigCC (on top of XMRig)

Check the [Coin Configuration](https://github.com/Bendr0id/xmrigCC/wiki/Coin-configurations) guide
* **NEW: Support of Crytptonight Masari (MSR) v7 variant (use variant "msr" to be ready for the fork, with autodetect)**
* **NEW: Support of Crytptonight-Heavy Haven Protocol (XHV) v3 variant (use variant "xhv")**
* **Support of Crytptonight Stellite (XTL) v4 variant**
* **Support of Crytptonight Alloy (XAO) variant**
* **Support of Crytptonight-Lite IPBC/TUBE variant**
* **Support of Crytptonight-Heavy (Loki, Ryo, ...)**
* **Support of Crytptonight v7 PoW changes**
* **Support of Crytptonight-Lite v7 PoW changes**
* Full SSL/TLS support for the whole communication: [Howto](https://github.com/Bendr0id/xmrigCC/wiki/tls)
    - XMRigCCServer Dashboard <-> Browser
    - XMRigCCServer <-> XMRigMiner
    - XMRigMiner <-> Pool
* Command and control server
* CC Dashboard with:
    * statistics of all connected miners
    * remote control miners (start/stop/restart/shutdown) 
    * remote configuration changes of miners
    * simple config editor for miner / mass editor for multiple miners 
    * monitoring / offline notification
* Daemon around the miner to restart and apply config changes
* High optimized mining code ([Benchmarks](#benchmarks))
* Working CPU affinity for NUMA Cores or CPU's with lots of cores
* Multihash support (Double, Triple, Quadruple, Quituple)
* Configuration of multihash per thread
* Smarter automatic CPU configuration
* It's still open source software :D


**[Find Help/Howto](https://github.com/Bendr0id/xmrigCC/wiki/)**


**XMRigCC Daemon(miner)**

![Screenshot of XMRig Daemon (miner)](https://i.imgur.com/gYq1QSP.png)

**XMRigCC Server**

![Screenshot of XMRigCC Server](https://i.imgur.com/0Ke9gIg.png)

**XMRigCC Dashboard**

![Screenshot of XMRigCC Dashboard](https://i.imgur.com/VwJaf26.png)


#### Table of contents
* [Download](#download)
* [Wiki/Building/Howto](https://github.com/Bendr0id/xmrigCC/wiki/)
* [Usage](#usage)
* [Multihash factor](#multihash-multihash-factor)
* [Multihash thread Mask](#multihash-thread-mask-only-for-multihash-factor--1)
* [Common Issues](#common-issues)
* [Optimizations](#cpu-mining-performance)
* [Benchmarks](#benchmarks)
* [Donations](#donations)
* [Contacts](#contact)

## Download
* Binary releases: https://github.com/Bendr0id/xmrigCC/releases
* Git tree: https://github.com/Bendr0id/xmrigCC.git
  * Clone with `git clone https://github.com/Bendr0id/xmrigCC.git` :hammer: [Build instructions](https://github.com/Bendr0id/xmrigCC/wiki/Build-Debian%5CUbuntu).

## Usage
### Basic example xmrigCCServer
```
xmrigCCServer --cc-port=3344 --cc-user=admin --cc-pass=pass --cc-access-token=SECRET_TOKEN_TO_ACCESS_CC_SERVER
```

### Options xmrigCCServer
```
        --cc-user=USERNAME                CC Server admin user
        --cc-pass=PASSWORD                CC Server admin pass
        --cc-access-token=T               CC Server access token for CC Client
        --cc-port=N                       CC Server
        --cc-use-tls                      enable tls encryption for CC communication
        --cc-cert-file=FILE               when tls is turned on, use this to point to the right cert file (default: server.pem) 
        --cc-key-file=FILE                when tls is turned on, use this to point to the right key file (default: server.key) 
        --cc-client-config-folder=FOLDER  Folder contains the client config files
        --cc-custom-dashboard=FILE        loads a custom dashboard and serve it to '/'
        --no-color                        disable colored output
    -S, --syslog                          use system log for output messages
    -B, --background                      run the miner in the background
    -c, --config=FILE                     load a JSON-format configuration file
    -l, --log-file=FILE                   log all output to a file
    -h, --help                            display this help and exit
    -V, --version                         output version information and exit
```

Also you can use configuration via config file, default **[config_cc.json](https://github.com/Bendr0id/xmrigCC/wiki/Config-XMRigCCServer)**. You can load multiple config files and combine it with command line options.


### Basic example xmrigDaemon
```
xmrigDaemon -o pool.minemonero.pro:5555 -u YOUR_WALLET -p x -k --cc-url=IP_OF_CC_SERVER:PORT --cc-access-token=SECRET_TOKEN_TO_ACCESS_CC_SERVER --cc-worker-id=OPTIONAL_WORKER_NAME
```

### Options xmrigDaemon
```
  -a, --algo=ALGO                       cryptonight (default), cryptonight-lite or cryptonight-heavy
  -o, --url=URL                         URL of mining server
  -O, --userpass=U:P                    username:password pair for mining server
  -u, --user=USERNAME                   username for mining server
  -p, --pass=PASSWORD                   password for mining server
  -t, --threads=N                       number of miner threads
  -A, --aesni=N                         selection of AES-NI mode (0 auto, 1 on, 2 off)
  -k, --keepalive                       send keepalived for prevent timeout (need pool support)
  -r, --retries=N                       number of times to retry before switch to backup server (default: 5)
  -R, --retry-pause=N                   time to pause between retries (default: 5)
      --pow-variant=V                   specificy the PoW variat to use: -> auto (default), 0 (v0), 1 (v1, aka monerov7, aeonv7), ipbc (tube), alloy, xtl (including autodetect for v5)
                                        for further help see: https://github.com/Bendr0id/xmrigCC/wiki/Coin-configurations
      --multihash-factor=N              number of hash blocks to process at a time (don't set or 0 enables automatic selection of optimal number of hash blocks)
      --multihash-thread-mask=MASK      limits multihash to given threads (mask), (default: all threads)
      --cpu-affinity                    set process affinity to CPU core(s), mask 0x3 for cores 0 and 1
      --cpu-priority                    set process priority (0 idle, 2 normal to 5 highest)
      --no-huge-pages                   disable huge pages support
      --donate-level=N                  donate level, default 5% (5 minutes in 100 minutes)
      --user-agent                      set custom user-agent string for pool
      --max-cpu-usage=N                 maximum CPU usage for automatic threads mode (default 75)
      --safe                            safe adjust threads and av settings for current CPU
      --nicehash                        enable nicehash/xmrig-proxy support
      --use-tls                         enable tls on pool communication
      --print-time=N                    print hashrate report every N seconds
      --api-port=N                      port for the miner API
      --api-access-token=T              access token for API
      --api-worker-id=ID                custom worker-id for API
      --cc-url=URL                      url of the CC Server
      --cc-use-tls                      enable tls encryption for CC communication
      --cc-access-token=T               access token for CC Server
      --cc-worker-id=ID                 custom worker-id for CC Server
      --cc-update-interval-s=N          status update interval in seconds (default: 10 min: 1)
      --no-color                        disable colored output
  -S, --syslog                          use system log for output messages
  -B, --background                      run the miner in the background
  -c, --config=FILE                     load a JSON-format configuration file
  -l, --log-file=FILE                   log all output to a file
  -h, --help                            display this help and exit
  -V, --version                         output version information and exit
```

Also you can use configuration via config file, default **[config.json](https://github.com/Bendr0id/xmrigCC/wiki/Config-XMRigDaemon)**. You can load multiple config files and combine it with command line options.

## Multihash (multihash-factor)
With this option it is possible to increase the number of hashblocks calculated by a single thread in each round.
Selecting multihash-factors greater than 1 increases the L3 cache demands relative to the multihash-factor.
E.g. at multihash-factor 2, each Cryptonight thread requires 4MB and each Cryptonight-lite thread requires 2 MB of L3 cache.
With multihash-factor 3, they need 6MB or 3MB respectively.

Setting multihash-factor to 0 will allow automatic detection of the optimal value.
Xmrig will then try to utilize as much of the L3 cache as possible for the selected number of threads.
If the threads parameter has been set to auto, Xmrig will detect the optimal number of threads first.
After that it finds the greatest possible multihash-factor.

### Multihash for low power operation
Depending the CPU and its L3 caches, it can make sense to replace multiple single hash threads with single multi-hash counterparts.
This change might come at the price of a minor drop in effective hash-rate, yet it will also reduce heat production and power consumption of the used CPU.

### Multihash for optimal CPU exploitation
In certain environments (e.g. vServer) the system running xmrig can have access to relatively large amounts of L3 cache, but may has access to only a few CPU cores.
In such cases, running xmrig with higher multihash-factors can lead to improvements.


## Multihash thread Mask (only for multihash-factor > 1)
With this option you can limit multihash to the given threads (mask).
This can significantly improve your hashrate by using unused l3 cache.
The default is to run the configured multihash-factor on all threads.


```
{
...

"multihash-factor":2,
"multihash-thread-mask":"0x5", // in binary -> 0101
"threads": 4,

...
}
``` 
This will limit multihash mode (multihash-factor = 2) to thread 0 and thread 2, thread 1 and thread 3 will run in single hashmode.


## Common Issues
### XMRigMiner
* XMRigMiner is just the worker, it is not designed to work standalone. Please start **XMRigDaemon** instead.

### Windows only: DLL error on starting
* Make sure that you installed latest Visual C++ Redistributable for Visual Studio 2015. Can be downloaded here: [microsoft.com](https://www.microsoft.com/de-de/download/details.aspx?id=48145)

### Linux only: Background mode
* The `--background` option will only work properly for the XMRigServer. But there is a simple workaround for the XMRigDaemon process. Just append an `&` to the command and it will run smoothly in the background.

    `./xmrigDaemon --config=my_config_cc.json &` or you just use `screen`


### HUGE PAGES unavailable (Windows)
* Run XMRig as Administrator.
* Since version 0.8.0 XMRig automatically enables SeLockMemoryPrivilege for current user, but reboot or sign out still required. [Manual instruction](https://msdn.microsoft.com/en-gb/library/ms190730.aspx).

### HUGE PAGES unavailable (Linux)
* Before starting XMRigDaemon set huge pages

    `sudo sysctl -w vm.nr_hugepages=128`


## Other information
* No HTTP support, only stratum protocol support.


### CPU mining performance
Please note performance is highly dependent on system load.
The numbers above are obtained on an idle system.
Tasks heavily using a processor cache, such as video playback, can greatly degrade hashrate.
Optimal number of threads depends on the size of the L3 cache of a processor, 1 thread requires 4 MB (Cryptonight-Heavy), 2 MB (Cryptonight) or 1MB (Cryptonigh-Lite) of cache.

### Maximum performance checklist
* Idle operating system.
* Do not exceed optimal thread count.
* Use modern CPUs with AES-NI instruction set.
* Try setup optimal cpu affinity.
* Try decreasing number of threads while increasing multihash-factor.
  Allocate unused cores and L3 cache with the help of multihash-thread-mask.
* Enable fast memory (Large/Huge pages).

## Benchmarks

Here are some result reported by users. Feel free to share your results, i'll add them.

### XMRigCC with max optimizations:

  * AMD Ryzen 1950x
        
        AEON: ~5300 h/s     (XMRig Stock: ~4900 h/s)
        XMR: ~1320 h/s      (XMRig Stock: ~1200 h/s)

  * AMD Ryzen 1600
  
        AEON: ~2065 h/s     (XMRig Stock: ~1800 h/s)
        XMR: ~565 h/s       (XMRig Stock: ~460 h/s)
  
  * 4x Intel XEON e7-4820
  
        AEON: ~2500 h/s     (XMRig Stock ~2200h/s)
        
  * 2x Intel XEON 2x e5-2670
        
        AEON: ~3300 h/s     (XMRig Stock ~2500h/s)
 
## Donations
* Default donation 5% (5 minutes in 100 minutes) can be reduced to 1% via command line option `--donate-level`. 

##### BenDroid (xmrigCC):
XMR:  `4BEn3sSa2SsHBcwa9dNdKnGvvbyHPABr2JzoY7omn7DA2hPv84pVFvwDrcwMCWgz3dQVcrkw3gE9aTC9Mi5HxzkfF9ev1eH`
AEON: `Wmtm4S2cQ8uEBBAVjvbiaVAPv2d6gA1mAUmBmjna4VF7VixLxLRUYag5cvsym3WnuzdJ9zvhQ3Xwa8gWxPDPRfcQ3AUkYra3W`
BTC:  `128qLZCaGdoWhBTfaS7rytpbvG4mNTyAQm`

##### xmrig:
XMR:  `48edfHu7V9Z84YzzMa6fUueoELZ9ZRXq9VetWzYGzKt52XU5xvqgzYnDK9URnRoJMk1j8nLwEVsaSWJ4fhdUyZijBGUicoD`
BTC:  `1P7ujsXeX7GxQwHNnJsRMgAdNkFZmNVqJT`

## Contact
* ben [at] graef.in
* Telegram: @BenDr0id
* [discord](https://discord.gg/r3rCKTB)
* [reddit](https://www.reddit.com/user/BenDr0id/)
