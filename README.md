# XMRigCC

[![Build status](https://ci.appveyor.com/api/projects/status/l8v7cuuy320a4tpd?svg=true)](https://ci.appveyor.com/project/Bendr0id/xmrigcc)

### About XMRigCC

XMRigCC is a fork of [XMRig](https://github.com/xmrig/xmrig) which adds the ability to remote control your XMRig instances via a Webfrontend and REST api.
This fork is based on XMRig (2.4.1) and adds a "Command and Control" (C&amp;C) server, a daemon to reload XMRig on config changes and modifications in XMRig to send the current status to the C&amp;C Server.
The modified version can also handle commands like "update config", "start/stop mining" or "restart/shutdown" which can be send from the C&amp;C-Server. 


[Wiki](https://github.com/Bendr0id/xmrigCC/wiki/)

[Notes / Howto](https://raw.githubusercontent.com/Bendr0id/xmrigCC/master/howto_and_notes.txt)

##### About XMRig

XMRig is high performance Monero (XMR) / Aeon CPU miner, with the official full Windows support.
Originally based on cpuminer-multi with heavy optimizations/rewrites and removing a lot of legacy code, since version 1.0.0 complete rewritten from scratch on C++.

#### Table of contents
* [Features](#features-of-xmrigcc)
* [Download](#download)
* [Usage](#usage)
* [Algorithm variations](#algorithm-variations)
* [Build on Debian/Ubuntu](https://github.com/Bendr0id/xmrigCC/wiki/Build-Debian%5CUbuntu)
* [Common Issues](#common-issues)
* [Other information](#other-information)
* [Donations](#donations)
* [Contacts](#contacts)
* [Wiki](https://github.com/Bendr0id/xmrigCC/wiki/)

## Features of xmrigCC
* Command and control server
* CC Dashboard with:
    * statistics of all connected miners
    * remote control miners (start/stop/restart/shutdown)
    * remote configuration changes of miners
* Daemon around the miner to restart and apply config changes
* High performance.
* Official Windows support.
* Small Windows executable, without dependencies.
* Support for backup (failover) mining server.
* keepalived support.
* CryptoNight-Lite support for AEON.
* Smart automatic [CPU configuration](https://github.com/xmrig/xmrig/wiki/Threads).
* Nicehash support
* It's open source software.

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
  -a, --algo=ALGO                       cryptonight (default) or cryptonight-lite
  -o, --url=URL                         URL of mining server
  -O, --userpass=U:P                    username:password pair for mining server
  -u, --user=USERNAME                   username for mining server
  -p, --pass=PASSWORD                   password for mining server
  -t, --threads=N                       number of miner threads
  -v, --av=N                            algorithm variation, 0 auto select
  -k, --keepalive                       send keepalived for prevent timeout (need pool support)
  -r, --retries=N                       number of times to retry before switch to backup server (default: 5)
  -R, --retry-pause=N                   time to pause between retries (default: 5)
      --cpu-affinity                    set process affinity to CPU core(s), mask 0x3 for cores 0 and 1
      --cpu-priority                    set process priority (0 idle, 2 normal to 5 highest)
      --no-huge-pages                   disable huge pages support
      --donate-level=N                  donate level, default 5% (5 minutes in 100 minutes)
      --user-agent                      set custom user-agent string for pool
      --max-cpu-usage=N                 maximum CPU usage for automatic threads mode (default 75)
      --safe                            safe adjust threads and av settings for current CPU
      --nicehash                        enable nicehash/xmrig-proxy support
      --print-time=N                    print hashrate report every N seconds
      --api-port=N                      port for the miner API
      --api-access-token=T              access token for API
      --api-worker-id=ID                custom worker-id for API
      --cc-url=URL                      url of the CC Server
      --cc-access-token=T               access token for CC Server
      --cc-worker-id=ID                 custom worker-id for CC Server
      --no-color                        disable colored output
  -S, --syslog                          use system log for output messages
  -B, --background                      run the miner in the background
  -c, --config=FILE                     load a JSON-format configuration file
  -l, --log-file=FILE                   log all output to a file
  -h, --help                            display this help and exit
  -V, --version                         output version information and exit

```

Also you can use configuration via config file, default **[config.json](https://github.com/Bendr0id/xmrigCC/wiki/Config-XMRigMiner)**. You can load multiple config files and combine it with command line options.

## Algorithm variations
Since version 0.8.0.
* `--av=1` For CPUs with hardware AES.
* `--av=2` Lower power mode (double hash) of `1`.
* `--av=3` Software AES implementation.
* `--av=4` Lower power mode (double hash) of `3`.

## Common Issues
### Windows only: DLL error on starting
* Make sure that you installed latest Visual C++ Redistributable für Visual Studio 2015. Can be downloaded here: [microsoft.com](https://www.microsoft.com/de-de/download/details.aspx?id=48145)

### Linux only: Background mode
* The `--background` option will only work properly for the XMRigServer. But there is a simple workaround for the XMRigDaemon process. Just append an `&amp;` to the command and it will run smoothly in the background.

    `./xmrigDaemon --config=my_config_cc.json &amp;`


### HUGE PAGES unavailable (Windows)
* Run XMRig as Administrator.
* Since version 0.8.0 XMRig automatically enable SeLockMemoryPrivilege for current user, but reboot or sign out still required. [Manual instruction](https://msdn.microsoft.com/en-gb/library/ms190730.aspx).

### HUGE PAGES unavailable (Linux)
* Before starting XMRigDaemon set huge pages

    `sudo sudo sysctl -w vm.nr_hugepages=256`


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
* Use modern CPUs with AES-NI instruction set.
* Try setup optimal cpu affinity.
* Enable fast memory (Large/Huge pages).

## Donations
* Default donation 5% (5 minutes in 100 minutes) can be reduced to 1% via command line option `--donate-level`. 
* The donations will be automatically splitted (50/50) on my pool to xmrig's and my wallet.

##### BenDroid (xmrigCC):
XMR:  `46FkYo7x6LqYjLQo4Jd84UTGBybW7tsWqJaQVLPhbUSK19ajSTMY9T2Sa2LH6CfWhSingjvQARtfeM4Feekpp2yFR1wsFNT`
AEON: `Wmtm4S2cQ8uEBBAVjvbiaVAPv2d6gA1mAUmBmjna4VF7VixLxLRUYag5cvsym3WnuzdJ9zvhQ3Xwa8gWxPDPRfcQ3AUkYra3W`
BTC:  `128qLZCaGdoWhBTfaS7rytpbvG4mNTyAQm`

##### xmrig:
XMR:  `48edfHu7V9Z84YzzMa6fUueoELZ9ZRXq9VetWzYGzKt52XU5xvqgzYnDK9URnRoJMk1j8nLwEVsaSWJ4fhdUyZijBGUicoD`
BTC:  `1P7ujsXeX7GxQwHNnJsRMgAdNkFZmNVqJT`

## Contacts xmrigCC
* ben [at] graef.in
* [reddit](https://www.reddit.com/user/BenDr0id/)

## Contacts xmrig
* support@xmrig.com
* [reddit](https://www.reddit.com/user/XMRig/)
