# XMRig w/ Salvium Support

Note about this project.  This is specifically for people that are trying to mine Salvium.  I am not rigourously testing this against other coins.  They *should* work, but I am only worried about XMR and SAL.

If Salvium is not a coin you are interested in or this bothers you, I suggest you head to the official [XMRIG repo](https://github.com/xmrig/xmrighttps:/).  It works great for everything else.

## Salvium Coin Configuration

Set `"coin": "SAL"` in your pool configuration to enable Salvium support:

```json
{
    "pools": [
        {
            "coin": "SAL",
            "url": "your-pool:port",
            "user": "your-wallet-address",
            "pass": "x"
        }
    ]
}
```

### What `"coin": "SAL"` does

The coin setting controls how the miner identifies and interacts with the Salvium network. Its behavior differs depending on the mining mode.

#### Pool / Stratum Mode

In pool mode, the coin setting serves one purpose: **algorithm selection**. When the pool sends a job without an explicit `algo` field, the miner uses the coin identity to determine that Salvium uses `rx/0` (RandomX). This is the only effect in pool mode — the pool handles block template parsing and validation on its side.

#### Daemon / Solo Mode

In daemon mode (`"daemon": true`), the coin identity activates the full Salvium protocol support:

- **Block template parsing** — Salvium blocks contain a `protocol_tx` between the miner transaction and the regular transaction hashes. The parser uses the coin identity to detect and correctly parse this extra transaction, which does not exist in Monero or other CryptoNote coins.
- **Output type handling** — Salvium supports output types `txout_to_key` (2), `txout_to_tagged_key` (3), and `txout_to_carrot_v1` (4), and allows multiple outputs per miner transaction. The parser enables these when the coin is SAL.
- **Transaction type and burn fields** — Salvium miner transactions include a `tx_type` field and an `amount_burnt` field after the extra data. These are parsed only for Salvium.
- **Merkle root computation** — The block hash tree includes both the miner transaction and the protocol transaction as base entries. The coin identity controls whether the root hash is computed over 1 base transaction (standard) or 2 (Salvium with protocol_tx).
- **Hardfork-aware protocol versioning** — The parser validates the protocol transaction version against the block's major version, supporting legacy (HF 2+), Carrot (HF 10+), and Tokens (HF 11+) eras.
- **Wallet address decoding** — Salvium legacy (`SaLv*`) and Carrot (`SC1*`) address prefixes are recognized for mainnet, testnet, and stagenet, along with their corresponding RPC (19081/29081/39081) and ZMQ (19082/29082/39082) ports.

#### Coin Metadata

| Property | Value |
|---|---|
| Code | `SAL` |
| Name | `Salvium` |
| Algorithm | `rx/0` (RandomX) |
| Block target | 120 seconds |
| Coin units | 10^8 (8 decimal places) |

For More information on Salvium:
- Salvium project: [https://salvium.io/](https://salvium.io/ "https://salvium.io/")
- Salvium: [https://github.com/salvium/salvium](https://github.com/salvium/salvium "https://github.com/salvium/salvium")
- P2Pool Salvium fork info: [https://whiskymine.io/p2pool-setup.html](https://whiskymine.io/p2pool-setup.html "https://whiskymine.io/p2pool-setup.html")
- P2Pool Salvium fork repo: [https://gitlab.com/whiskyrelaxing-group/p2pool-salvium-releases](https://gitlab.com/whiskyrelaxing-group/p2pool-salvium-releases "https://gitlab.com/whiskyrelaxing-group/p2pool-salvium-releases")


## Original Readme

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
