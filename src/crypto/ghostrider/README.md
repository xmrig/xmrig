# GhostRider (Raptoreum) release notes

**XMRig** supports GhostRider algorithm starting from version **v6.16.0**.

No tuning is required - auto-config works well on most CPUs!

**Note for Windows users: MSVC binary is ~5% faster than GCC binary!**

### Sample command line (non-SSL port)
```
xmrig -a gr -o raptoreumemporium.com:3008 -u WALLET_ADDRESS
```

### Sample command line (SSL port)
```
xmrig -a gr -o us.flockpool.com:5555 --tls -u WALLET_ADDRESS
```

You can use **rtm_ghostrider_example.cmd** as a template and put pool URL and your wallet address there. The general XMRig documentation is available [here](https://xmrig.com/docs/miner).

## Performance

While individual algorithm implementations are a bit unoptimized, XMRig achieves higher hashrates by employing better auto-config and more fine-grained thread scheduling: it can calculate a single batch of hashes using 2 threads for parts that don't require much cache. For example, on a typical Intel CPU (2 MB cache per core) it will use 1 thread per core for cn/fast, and 2 threads per core for other Cryptonight variants while calculating the same batch of hashes, always achieving more than 50% CPU load.

For the same reason, XMRig can sometimes use less than 100% CPU on Ryzen 3000/5000 CPUs if it finds that running 1 thread per core is faster for some Cryptonight variants on your system. Also, this is why it reports using only half the threads at startup - it's actually 2 threads per each reported thread.

**Windows** (detailed results [here](https://imgur.com/a/GCjEWpl))
CPU|cpuminer-gr-avx2 (tuned), h/s|XMRig (MSVC build), h/s|Speedup
-|-|-|-
AMD Ryzen 7 4700U|632.6|731|+15.5%
Intel Core i7-2600|496.4|533.6|+7.5%
AMD Ryzen 7 3700X @ 4.1 GHz|2453.0|2469.1|+0.65%
AMD Ryzen 5 5600X @ 4.65 GHz|2112.6|2221.2|+5.1%

**Linux** (tested by **Delgon**, detailed results [here](https://cdn.discordapp.com/attachments/604375870236524574/913167614749048872/unknown.png))
CPU|cpuminer-gr-avx2 (tuned), h/s|XMRig (GCC build), h/s|Speedup
-|-|-|-
AMD Ryzen 9 3900X|3746.51|3604.89|-3.78%
2xIntel Xeon E5-2698v3|2563.4|2638.38|+2.925%
