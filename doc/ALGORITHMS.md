# Algorithms

Since version 3 mining algorithm should specified for each pool separately (`algo` option), earlier versions was use one global `algo` option and per pool `variant` option (this option was removed in v3). If your pool support [mining algorithm negotiation](https://github.com/xmrig/xmrig-proxy/issues/168) you may not specify this option at all.
 
#### Example
```json
{
  "pools": [
    {
      "url": "...",
      "algo": "cn/r",
      ...
    }
 ],
 ...
}
```

#### Pools with mining algorithm negotiation support.

 * [www.hashvault.pro](https://www.hashvault.pro/)
 * [moneroocean.stream](https://moneroocean.stream)
 
 ## Algorithm names

| Name            | Memory | Notes                                                                                |
|-----------------|--------|--------------------------------------------------------------------------------------|
| `cn/0`          |   2 MB | CryptoNight (original)                                                               |
| `cn/1`          |   2 MB | CryptoNight variant 1 also known as `Monero7` and `CryptoNightV7`.                   |
| `cn/2`          |   2 MB | CryptoNight variant 2.                                                               |
| `cn/r`          |   2 MB | CryptoNightR (Monero's variant 4).                                                   |
| `cn/wow`        |   2 MB | CryptoNightR (Wownero).                                                              |
| `cn/fast`       |   2 MB | CryptoNight variant 1 with half iterations.                                          |
| `cn/half`       |   2 MB | CryptoNight variant 2 with half iterations (Masari/Torque)                           |
| `cn/xao`        |   2 MB | CryptoNight variant 0 (modified, Alloy only)                                         |
| `cn/rto`        |   2 MB | CryptoNight variant 1 (modified, Arto only)                                          |
| `cn/rwz`        |   2 MB | CryptoNight variant 2 with 3/4 iterations and reversed shuffle operation (Graft).    |
| `cn/zls`        |   2 MB | CryptoNight variant 2 with 3/4 iterations (Zelerius).                                |
| `cn/double`     |   2 MB | CryptoNight variant 2 with double iterations (X-CASH).                               |
| `cn/gpu`        |   2 MB | CryptoNight-GPU (RYO).                                                               |
| `cn-lite/0`     |   1 MB | CryptoNight-Lite variant 0.                                                          |
| `cn-lite/1`     |   1 MB | CryptoNight-Lite variant 1.                                                          |
| `cn-heavy/0`    |   4 MB | CryptoNight-Heavy       .                                                            |
| `cn-heavy/xhv`  |   4 MB | CryptoNight-Heavy (modified, TUBE only).                                             |
| `cn-heavy/tube` |   4 MB | CryptoNight-Heavy (modified, Haven Protocol only).                                   |
| `cn-pico`       | 256 KB | TurtleCoin (TRTL)                                                                    |
| `rx/0`          |   2 MB | RandomX (reference configuration), reserved for future use.                          |
| `rx/wow`        |   1 MB | RandomWOW (Wownero).                                                                 |
| `rx/loki`       |   2 MB | RandomXL (Loki).                                                                     |