# XMRig

[![Github All Releases](https://img.shields.io/github/downloads/xmrig/xmrig/total.svg)](https://github.com/xmrig/xmrig/releases)
[![GitHub release](https://img.shields.io/github/release/xmrig/xmrig/all.svg)](https://github.com/xmrig/xmrig/releases)
[![GitHub Release Date](https://img.shields.io/github/release-date/xmrig/xmrig.svg)](https://github.com/xmrig/xmrig/releases)
[![GitHub license](https://img.shields.io/github/license/xmrig/xmrig.svg)](https://github.com/xmrig/xmrig/blob/master/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/xmrig/xmrig.svg)](https://github.com/xmrig/xmrig/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/xmrig/xmrig.svg)](https://github.com/xmrig/xmrig/network)

XMRig is a high performance, open source, cross platform RandomX, KawPow, CryptoNight and [GhostRider](https://github.com/xmrig/xmrig/tree/master/src/crypto/ghostrider#readme) unified CPU/GPU miner and [RandomX benchmark](https://xmrig.com/benchmark). Official binaries are available for Windows, Linux, macOS and FreeBSD.

## Mining backends
- **CPU** (x86/x64/ARMv7/ARMv8/RISC-V)
- **OpenCL** for AMD GPUs.
- **CUDA** for NVIDIA GPUs via external [CUDA plugin](https://github.com/xmrig/xmrig-cuda).

## Download
* **[Binary releases](https://github.com/xmrig/xmrig/releases)**
* **[Build from source](https://xmrig.com/docs/miner/build)**

## Usage
The preferred way to configure the miner is the [JSON config file](https://xmrig.com/docs/miner/config) as it is more flexible and human friendly. The [command line interface](https://xmrig.com/docs/miner/command-line-options) does not cover all features, such as mining profiles for different algorithms. Important options can be changed during runtime without miner restart by editing the config file or executing [API](https://xmrig.com/docs/miner/api) calls.

* **[Wizard](https://xmrig.com/wizard)** helps you create initial configuration for the miner.
* **[Workers](http://workers.xmrig.info)** helps manage your miners via HTTP API.

## Mine TKM

The release archive includes a TKM-ready `config.json`. Before starting,
replace `YOUR_TKM_WALLET_ADDRESS` with the wallet address where you want pool
payments sent. The included pool is the Tor onion service
`4aof7abdduh4vftejgdpdfqeosvxxco3xmpu4uqypnpdbi7wjuzfqhqd.onion:33330` and
uses the TKM RandomX algorithm `rx/tkm`. Start Tor with a local SOCKS5 proxy
on `127.0.0.1:9050` before starting XMRig. The pool entry must contain
`"socks5": "socks5://127.0.0.1:9050"`; XMRig cannot resolve `.onion` names
through ordinary DNS.

Use this configuration as `config.json` beside the miner executable:

```json
{
    "autosave": true,
    "background": false,
    "cpu": {
        "enabled": true,
        "huge-pages": true,
        "yield": true,
        "max-threads-hint": 100
    },
    "opencl": { "enabled": false },
    "cuda": { "enabled": false },
    "donate-level": 1,
    "pools": [
        {
            "algo": "rx/tkm",
            "coin": "TKM",
            "url": "4aof7abdduh4vftejgdpdfqeosvxxco3xmpu4uqypnpdbi7wjuzfqhqd.onion:33330",
            "user": "YOUR_TKM_WALLET_ADDRESS",
            "pass": "x",
            "rig-id": "worker-1",
            "keepalive": true,
            "tls": false,
            "socks5": "socks5://127.0.0.1:9050"
        }
    ],
    "print-time": 60,
    "retries": 5,
    "retry-pause": 5,
    "watch": true
}
```

`user` is normally the TKM payout address, not a password or a private key. A
full `tkmshield3.<...>` payment code is also accepted when Shield4 payouts are
enabled; it is public recipient data (never a seed or passphrase) and is about
28 KiB, so use a current TKM XMRig build with the expanded Stratum send limit.
For a legacy build, use the 0x payout address and attach the payment code in the
pool's authenticated recipient-code endpoint. The `socks5` setting is required:
without it the miner will try ordinary DNS and fail with `unknown node or
service` for the onion hostname.

If an older binary prints `max send buffer size exceeded` when logging in with a
`tkmshield3` code, replace it with the current TKM release or use the compact
0x-address login. The pool job itself is small; this message is the miner's
local 16 KiB login limit, not a failed RandomX share.

Run the miner from the directory containing the executable and config:

```sh
./xmrig --config=config.json
```

On Windows, open Command Prompt in the extracted directory and run:

```bat
xmrig.exe --config=config.json
```

The miner prints accepted and rejected shares in the terminal. Keep the
terminal open while mining. Set a worker name by changing `rig-id` in the
pool entry. The pool dashboard is available at `https://pool.tkmchain.site`.

For Android, copy the native binary and `config.json` into an executable
directory, replace the wallet address, then run `./xmrig --config=config.json`.

### Other miners

The TKM pool is Tor-only. Every miner must run Tor locally and connect through
a SOCKS5 proxy at `127.0.0.1:9050`; direct IP connections and ordinary DNS
lookups cannot reach the `.onion` pool. Configure miners with:

```text
Pool:  4aof7abdduh4vftejgdpdfqeosvxxco3xmpu4uqypnpdbi7wjuzfqhqd.onion
Port:  33330
SOCKS5: 127.0.0.1:9050
Algorithm: rx/tkm
```

For a miner that has no SOCKS5 setting, Linux users can try the Tor wrapper:

```sh
torsocks ./miner --pool \
  4aof7abdduh4vftejgdpdfqeosvxxco3xmpu4uqypnpdbi7wjuzfqhqd.onion:33330
```

The miner must implement the `rx/tkm` algorithm. Standard Monero XMRig
binaries may not include this TKM algorithm; use a TKM build. On Windows,
macOS, and Android, use a miner with native SOCKS5 support or a platform Tor
wrapper.

## Donations
* Default donation 1% (1 minute in 100 minutes) can be increased via option `donate-level` or disabled in source code.
* XMR: `48edfHu7V9Z84YzzMa6fUueoELZ9ZRXq9VetWzYGzKt52XU5xvqgzYnDK9URnRoJMk1j8nLwEVsaSWJ4fhdUyZijBGUicoD`

## Developers
* **[xmrig](https://github.com/xmrig)**
* **[sech1](https://github.com/SChernykh)**

## Contacts
* support@xmrig.com
* [reddit](https://www.reddit.com/user/XMRig/)
* [twitter](https://twitter.com/xmrig_dev)
