# CPU backend

All CPU related settings contains in one `cpu` object in config file, CPU backend allow specify multiple profiles and allow switch between them without restrictions by pool request or config change. Default auto-configuration create reasonable minimum of profiles which cover all supported algorithms.

### Example

Example below demonstrate all primary ideas of flexible profiles configuration:

* `"rx/wow"` Exact match to algorithm `rx/wow`, defined 4 threads without CPU affinity.
* `"cn"` Default failback profile for all `cn/*` algorithms, defined 2 threads with CPU affinity, another failback profiles is `cn-lite`, `cn-heavy` and `rx`.
* `"cn-lite"` Default failback profile for all `cn-lite/*` algorithms, defined 2 double threads with CPU affinity.
* `"cn-pico"` Alternative short object format.
* `"custom-profile"` Custom user defined profile.
* `"*"` Failback profile for all unhandled by other profiles algorithms.
* `"cn/r"` Exact match, alias to profile `custom-profile`.
* `"cn/0"` Exact match, disabled algorithm.

```json
{
    "cpu": {
        "enabled": true,
        "huge-pages": true,
        "hw-aes": null,
        "priority": null,
        "asm": true,
        "rx/wow": [-1, -1, -1, -1],
        "cn": [
            [1, 0],
            [1, 2]
        ],
        "cn-lite": [
            [2, 0],
            [2, 2]
        ],
        "cn-pico": {
            "intensity": 2,
            "threads": 8,
            "affinity": -1
        },
        "custom-profile": [0, 2],
        "*": [-1],
        "cn/r": "custom-profile",
        "cn/0": false
    }
}
```

## Threads definition
Threads can be defined in 3 formats.

#### Array format
```json
[
    [1, 0],
    [1, 2],
    [1, -1],
    [2, -1]
]
```
Each line represent one thread, first element is intensity, this option was known as `low_power_mode`, possible values is range from 1 to 5, second element is CPU affinity, special value `-1` means no affinity.

#### Short array format
```json
[-1, -1, -1, -1]
```
Each number represent one thread and means CPU affinity, this is default format for algorithm with maximum intensity 1, currently it all RandomX variants and cryptonight-gpu.

#### Short object format
```json
{
    "intensity": 2,
    "threads": 8,
    "affinity": -1
}
```
Internal format, but can be user defined.

## Shared options

#### `enabled`
Enable (`true`) or disable (`false`) CPU backend, by default `true`.

#### `huge-pages`
Enable (`true`) or disable (`false`) huge pages support, by default `true`.

#### `hw-aes`
Force enable (`true`) or disable (`false`) hardware AES support. Default value `null` means miner autodetect this feature. Usually don't need change this option, this option useful for some rare cases when miner can't detect hardware AES, but it available. If you force enable this option, but your hardware not support it, miner will crash.

#### `priority`
Mining threads priority, value from `1` (lowest priority) to `5` (highest possible priority). Default value `null` means miner don't change threads priority at all.

#### `asm`
Enable/configure or disable ASM optimizations. Possible values: `true`, `false`, `"intel"`, `"ryzen"`, `"bulldozer"`.

#### `argon2-impl` (since v3.1.0)
Allow override automatically detected Argon2 implementation, this option added mostly for debug purposes, default value `null` means autodetect. Other possible values: `"x86_64"`, `"SSE2"`, `"SSSE3"`, `"XOP"`, `"AVX2"`, `"AVX-512F"`. Manual selection has no safe guards, if you CPU not support required instuctions, miner will crash.

#### `max-threads-hint` (since v4.2.0)
Maximum CPU threads count (in percentage) hint for autoconfig. [CPU_MAX_USAGE.md](CPU_MAX_USAGE.md)

#### `memory-pool` (since v4.3.0)
Use continuous, persistent memory block for mining threads, useful for preserve huge pages allocation while algorithm swithing. Possible values `false` (feature disabled, by default) or `true` or specific count of 2 MB huge pages.

#### `yield` (since v5.1.1)
Prefer system better system response/stability `true` (default value) or maximum hashrate `false`.
