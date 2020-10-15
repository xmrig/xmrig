# Embedded benchmark

You can run with XMRig with the following commands:
```
xmrig --bench=1M
xmrig --bench=10M
xmrig --bench=1M -a rx/wow
xmrig --bench=10M -a rx/wow
```
This will run between 1 and 10 million RandomX hashes, depending on `bench` parameter, and print the time it took. First two commands use Monero variant (2 MB per thread, best for Zen2/Zen3 CPUs), second two commands use Wownero variant (1 MB per thread, useful for Intel and 1st gen Zen/Zen+ CPUs).

Checksum of all the hashes will be also printed to check stability of your hardware: if it's green then it's correct, if it's red then there was hardware error during computation. No Internet connection is required for the benchmark.

Double check that you see `Huge pages 100%` both for dataset and for all threads, and also check for `msr register values ... has been set successfully` - without this result will be far from the best. Running as administrator is required for MSR and huge pages to be set up properly.

![Benchmark example](https://i.imgur.com/PST3BYc.png)

### Benchmark with custom config

You can run benchmark with any configuration you want. Just start without command line parameteres, use regular config.json and add `"benchmark":"1M",` on the next line after pool url. 

# Stress test

You can also run continuous stress-test that is as close to the real RandomX mining as possible and doesn't require any configuration:
```
xmrig --stress
xmrig --stress -a rx/wow
```
This will require Internet connection and will run indefinitely.