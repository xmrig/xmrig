# CPU backend

All CPU related settings contains in one `cpu` object in config file, CPU backend allow specify multiple profiles and allow switch between them without restrictions by pool request or config change. Default auto-configuration create reasonable minimum of profiles which cover all supported algorithms.

### Example

Example below demonstrate all primary ideas of flexible profiles configuration:

* `"rx/wow"` Exact match to algorithm `rx/wow`, defined 4 threads without CPU affinity.
* `"cn"` Default failback profile for all `cn/*` algorithms, defined 2 threads with CPU affinity, another failback profiles is `cn-lite`, `cn-heavy` and `rx`.
* `"cn-lite"` Default failback profile for all `cn-lite/*` algorithms, defined 2 double threads with CPU affinity.
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
        "rx/wow": [
            -1,
            -1,
            -1,
            -1,
        ],
        "cn": [
            0,
            2
        ],
        "cn-lite": [
            {
                "intensity": 2,
                "affinity": 0
            },
            {
                "intensity": 2,
                "affinity": 2
            }
        ],
        "custom-profile": [
            0,
            2,
        ],
        "*": [
            -1
        ],
        "cn/r": "custom-profile",
        "cn/0": false
    }
}
```

### Intensity
This option was known as `low_power_mode`, possible values is range from 1 to 5, for convinient if value 1 used, possible omit this option and specify CPU thread config by only one number: CPU affinity, instead of object.

