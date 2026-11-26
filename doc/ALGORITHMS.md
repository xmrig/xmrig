# Algorithms

Algorithm can be defined in 3 ways:

1. By pool, using algorithm negotiation, in this case no need specify algorithm on miner side.
2. Per pool `coin` option, see [coin names](#coin-names) for possible values.
3. Per pool `algo` option.

Option `coin` useful for pools without [algorithm negotiation](https://xmrig.com/docs/extensions/algorithm-negotiation) support or daemon to allow automatically switch algorithm in next hard fork. If you use xmrig-proxy don't need specify algorithm on miner side.

## Algorithm names

| Name | Memory | Version | Description | Notes |
|------|--------|---------|-------------|-------|
| `rx/2` | 2 MB | 6.26.0+ | RandomX v2 (Monero after the FCMP++ hard fork). |  |
| `rx/yada` | 2 MB | 6.22.0+ | RandomYada (RandomX variant for YadaCoin). |  |
| `ghostrider` | 2 MB | 6.16.0+ | GhostRider (Raptoreum). | CPU only |
| `cn/upx2` | 128 KB | 6.12.0+ | CryptoNight-UPX2 (uPlexa). |  |
| `cn/ccx` | 2 MB | 6.2.0+ | CryptoNight variant 0 (modified, Conceal). |  |
| `kawpow` | - | 6.0.0+ | KawPow (Ravencoin) | GPU only |
| `rx/sfx` | 2 MB | 5.4.0+ | RandomSFX (RandomX variant for Safex). |  |
| `rx/arq` | 256 KB | 4.3.0+ | RandomARQ (RandomX variant for ArQmA). |  |
| `rx/0` | 2 MB | 3.2.0+ | RandomX (Monero). |  |
| `argon2/chukwa` | 512 KB | 3.1.0+ | Argon2id (Chukwa). | CPU only |
| `rx/wow` | 1 MB | 3.0.0+ | RandomWOW (RandomX variant for Wownero). |  |
| `cn-pico` | 256 KB | 2.10.0+ | CryptoNight-Pico. |  |

## Deprecated algorithms

No live coin uses these algorithms anymore. They still work, but will be removed in v7.0. The miner prints a warning when a pool or benchmark uses one of them, and marks it as deprecated in the pool list at startup.

| Name | Memory | Version | Description | Notes |
|------|--------|---------|-------------|-------|
| `rx/graft` | 2 MB | 6.15.0+ | RandomGRAFT (RandomX variant for Graft). |  |
| `argon2/chukwav2` | 1 MB | 6.4.0+ | Argon2id (Chukwa v2). | CPU only |
| `cn-pico/tlo` | 256 KB | 5.5.0+ | CryptoNight-Pico (Talleo). |  |
| `argon2/ninja` | 256 KB | 3.1.0+ | Argon2id (WRKZ), alias `argon2/wrkz`. | CPU only |
| `cn/fast` | 2 MB | 3.0.0+ | CryptoNight variant 1 with half iterations. |  |
| `cn/rwz` | 2 MB | 2.14.0+ | CryptoNight variant 2 with 3/4 iterations and reversed shuffle operation. |  |
| `cn/zls` | 2 MB | 2.14.0+ | CryptoNight variant 2 with 3/4 iterations. |  |
| `cn/double` | 2 MB | 2.14.0+ | CryptoNight variant 2 with double iterations. |  |
| `cn/r` | 2 MB | 2.13.0+ | CryptoNightR (Monero's variant 4). |  |
| `cn/half` | 2 MB | 2.9.0+ | CryptoNight variant 2 with half iterations. |  |
| `cn/2` | 2 MB | 2.8.0+ | CryptoNight variant 2. |  |
| `cn/xao` | 2 MB | 2.6.4+ | CryptoNight variant 0 (modified). |  |
| `cn/rto` | 2 MB | 2.6.4+ | CryptoNight variant 1 (modified). |  |
| `cn-heavy/tube` | 4 MB | 2.6.4+ | CryptoNight-Heavy (modified). |  |
| `cn-heavy/xhv` | 4 MB | 2.6.3+ | CryptoNight-Heavy (modified). |  |
| `cn-heavy/0` | 4 MB | 2.6.0+ | CryptoNight-Heavy. |  |
| `cn/1` | 2 MB | 2.5.0+ | CryptoNight variant 1. |  |
| `cn-lite/1` | 1 MB | 2.5.0+ | CryptoNight-Lite variant 1. |  |
| `cn-lite/0` | 1 MB | 0.8.0+ | CryptoNight-Lite variant 0. |  |
| `cn/0` | 2 MB | 0.5.0+ | CryptoNight (original). |  |

## Coin names

Values for the per pool `coin` option, case insensitive.

| Coin | Values | Algorithm | Notes |
|------|--------|-----------|-------|
| Monero | `monero`, `xmr` | `rx/0`, `rx/2` after the FCMP++ hard fork |  |
| Wownero | `wownero`, `wow` | `rx/wow` |  |
| ArQmA | `arqma`, `arq` | `rx/arq` |  |
| Ravencoin | `ravencoin`, `rvn` | `kawpow` |  |
| Zephyr | `zephyr`, `zeph` | `rx/0` |  |
| YadaCoin | `yadacoin`, `yda` | `rx/yada` |  |
| Sumokoin | `sumokoin`, `sumo` | `cn/r` | Deprecated, will be removed in v7.0 |
| Graft | `graft`, `grft` | `rx/graft` | Deprecated, will be removed in v7.0 |
| Townforge | `townforge` | `rx/0` | Deprecated, will be removed in v7.0 |

## Migration to v3
Since version 3 mining [algorithm](#algorithm-names) should specified for each pool separately (`algo` option), earlier versions was use one global `algo` option and per pool `variant` option (this option was removed in v3). If your pool support [mining algorithm negotiation](https://github.com/xmrig/xmrig-proxy/issues/168) you may not specify this option at all.
 
#### Example
```json
{
  "pools": [
    {
      "url": "...",
      "algo": "rx/0",
      "coin": null
      ...
    }
 ],
 ...
}
```
