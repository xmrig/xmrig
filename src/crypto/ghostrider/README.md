# GhostRider (Raptoreum) release notes

**XMRig** supports GhostRider algorithm starting from version **v6.16.0**.

No tuning is required - auto-config works well on most CPUs!

### Sample command line (non-SSL port)
```
xmrig -a gr -o raptoreumemporium.com:3008 -u WALLET_ADDRESS -p x
```

### Sample command line (SSL port)
```
xmrig -a gr -o rtm.suprnova.cc:4273 --tls -u WALLET_ADDRESS -p x
```

You can use **rtm_ghostrider_example.cmd** as a template and put pool URL and your wallet address there. The general XMRig documentation is available [here](https://xmrig.com/docs/miner).

**Using `--threads` or `-t` option is NOT recommended because it turns off advanced built-in config.** If you want to tweak the nubmer of threads used for GhostRider, it's recommended to start using config.json instead of command line. The best suitable command line option for this is `--cpu-max-threads-hint=N` where N can be between 0 and 100.

## Performance

While individual algorithm implementations are a bit unoptimized, XMRig achieves higher hashrates by employing better auto-config and more fine-grained thread scheduling: it can calculate a single batch of hashes using 2 threads for parts that don't require much cache. For example, on a typical Intel CPU (2 MB cache per core) it will use 1 thread per core for cn/fast, and 2 threads per core for other Cryptonight variants while calculating the same batch of hashes, always achieving more than 50% CPU load.

For the same reason, XMRig can sometimes use less than 100% CPU on Ryzen 3000/5000 CPUs if it finds that running 1 thread per core is faster for some Cryptonight variants on your system.

**Windows** (detailed results [here](https://imgur.com/a/0njIVVW))
CPU|cpuminer-gr-avx2 1.2.4.1 (tuned), h/s|XMRig v6.16.2 (MSVC build), h/s|Speedup
-|-|-|-
AMD Ryzen 7 4700U|632.6|733.1|+15.89%
Intel Core i7-2600|496.4|554.6|+11.72%
AMD Ryzen 7 3700X @ 4.1 GHz|2453.0|2496.5|+1.77%
AMD Ryzen 5 5600X @ 4.65 GHz|2112.6|2337.5|+10.65%

**Linux (outdated)** (tested by **Delgon**, detailed results [here](https://cdn.discordapp.com/attachments/604375870236524574/913167614749048872/unknown.png))
CPU|cpuminer-gr-avx2 1.2.4.1 (tuned), h/s|XMRig v6.16.0 (GCC build), h/s|Speedup
-|-|-|-
AMD Ryzen 9 3900X|3746.51|3604.89|-3.78%
2xIntel Xeon E5-2698v3|2563.4|2638.38|+2.925%
