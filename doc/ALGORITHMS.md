# Algorithms

Since version 3 mining [algorithm](#algorithm-names) should specified for each pool separately (`algo` option), earlier versions was use one global `algo` option and per pool `variant` option (this option was removed in v3). If your pool support [mining algorithm negotiation](https://github.com/xmrig/xmrig-proxy/issues/168) you may not specify this option at all.
 
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

* https://xmrig.com/docs/algorithms
