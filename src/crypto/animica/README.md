# Animica hashshare PoW (`--algo animica`)

Adds support for mining **Animica** — a Layer-1 blockchain that pairs hashshare PoW with optional AICF (AI Compute Fund) inference jobs — to the xmrig CPU backend.

## Algorithm

Animica's PoW is intentionally simple: **NIST SHA3-256** over `prefix || nonce_le8`, where
- `prefix` is the per-job header built from the Stratum job (`version || prev_hash || merkle_root || ntime || nbits || …`), and
- `nonce` is a 64-bit counter the miner sweeps, serialized little-endian.

A digest `d` is a valid share iff `int256_be(d) <= target`. The target is derived on the pool side from a µ-nats share threshold (`d_ratio · Θ`) — the miner just compares 32 BE bytes.

No large memory dataset (cf. RandomX), no per-block program (cf. KawPow). The binary grows by ~6 KB.

## Build

```bash
cmake -B build -DWITH_ANIMICA=ON
cmake --build build -j
```

`WITH_ANIMICA` defaults to `ON`. To exclude the family, pass `-DWITH_ANIMICA=OFF`; this strips the algorithm enum, the SHA3 module, and the CPU worker.

## Mining

```bash
./xmrig --algo animica --url pool.animica.org:3333 \
        --user anim1zqpye0muk7etljd2fh7wxsh9y9027cq7dykj3de8u80s2mcnfp6qxecpunkth.rig01 \
        --keepalive --donate-level 1
```

Aliases accepted by `--algo`: `animica`, `animica/sha3`, `animica-sha3`, `anm`, `anm/sha3`.

The `--user` field follows the Animica Stratum convention: `<bech32-address>.<worker-tag>`. The worker tag is optional; the address goes into the µ-nats payout split.

## Useful-work (AICF) jobs

Animica's Stratum extension lets miners optionally serve AICF inference jobs alongside hash shares — that is, run LLM inference for paying clients and receive ANIMICA per turn. The protocol uses `mining.aicf.notify` / `mining.aicf.submit` on top of standard v1.

xmrig's role here is delegation: when the pool pushes an AICF job, xmrig spawns the configured runner binary (default `animica miner aicf-worker` from the [`animica`](https://pypi.org/project/animica/) PyPI package), pipes the prompt in, reads the generated text from stdout, and submits via `mining.aicf.submit`. xmrig itself does not embed an LLM; the LLM lives in the runner subprocess.

To enable, install the Python helper and point `--aicf-runner` at it:

```bash
pip install animica
./xmrig --algo animica --url pool.animica.org:3333 \
        --user anim1... \
        --aicf-tiers tiny,small \
        --aicf-runner /root/.venv/bin/aicf-chat-once
```

Status: the algorithm + CPU hot loop are complete. AICF delegation is **scaffolded** behind the same `XMRIG_ALGO_ANIMICA` flag — see the tracking notes in `src/crypto/animica/AnimicaWorker.h` for the CPU backend wiring still pending in `src/backend/cpu/CpuWorker.cpp`. Live test against `pool.animica.org:3333` will follow the backend wiring.

## License

GPL-3.0, same as upstream xmrig. Animica copyright held by `ercmine` and contributors.
