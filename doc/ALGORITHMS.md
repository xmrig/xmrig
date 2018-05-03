# Algorithms

XMRig uses a different way to specify algorithms, compared to other miners.

Algorithm selection splitted to 2 parts:

 * Global base algorithm per miner or proxy instance, `algo` option. Possible values: `cryptonight`, `cryptonight-lite`, `cryptonight-heavy`.
 * Algorithm variant specified separately for each pool, `variant` option.

Possible variants for `cryptonight`:

 * `0` Original cryptonight.
 * `1` cryptonight variant 1, also known as cryptonight v7 or monero7.
 * `"xtl"` Stellite coin variant.

Possible variants for `cryptonight-lite`:

 * `0` Original cryptonight-lite.
 * `1` cryptonight-lite variant 1, also known as cryptonight-lite v7 or aeon7.
 * `"ipbc"` IPBC coin variant.

For `cryptonight-heavy` currently no variants.


### Cheatsheet

You mine **Sumokoin** or **Haven Protocol**?
Your algorithm is `cryptonight-heavy` no variant option need.

You mine **Aeon**, **TurtleCoin** or **IPBC**?.
Your base algorithm is `cryptonight-lite`:
Variants:
 * Aeon: `-1` autodetect. `0` right now, `1` after fork.
 * TurtleCoin: `1`.
 * IPBC: `"ipbc"`.

In all other cases base algorithm is `cryptonight`.

### Mining algorithm negotiation
If your pool support [mining algorithm negotiation](https://github.com/xmrig/xmrig-proxy/issues/168) miner will choice proper variant automaticaly and if you choice wrong base algorithm you will see error message.

Pools with mining algorithm negotiation support.
 * [www.hashvault.pro](https://www.hashvault.pro/)