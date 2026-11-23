/* XMRig
 * Copyright 2026      ercmine     <https://github.com/ercmine>
 * Copyright 2016-2024 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 */

#ifndef XMRIG_ANIMICAAICFDELEGATE_H
#define XMRIG_ANIMICAAICFDELEGATE_H


#include <chrono>
#include <string>


namespace xmrig {


/**
 * AICF (useful-work) delegate.
 *
 * The Animica Stratum extension lets miners serve LLM inference jobs
 * alongside hash shares. The protocol adds two methods on top of
 * standard v1:
 *
 *   - `mining.aicf.notify` (server → miner) pushes a JobSpec:
 *       params = {
 *           "jobId": "0x…",
 *           "spec":  { "prompt": "…", "tier_preferred": "tiny" |
 *                                     "small" | "flagship" | "large",
 *                       "max_output_tokens": int, … },
 *           "tier":  "tiny",
 *           "expected_payout_animica": float,
 *       }
 *
 *   - `mining.aicf.submit` (miner → server) submits the generated text:
 *       params = {
 *           "worker": "<bech32>.tag",
 *           "jobId":  "0x…",
 *           "text":   "<generated text>",
 *           "latencyMs": int,
 *           "attestation": { "backend": "…", "model": "…", … },
 *       }
 *
 * xmrig itself does not embed an LLM (it's a hash miner, not an
 * inference engine). The delegate shells out to a configurable runner
 * binary that takes a one-line JSON spec on stdin and writes a
 * one-line JSON receipt on stdout. The default runner is the
 * `aicf-chat-once` helper shipped by the `animica` PyPI package; any
 * other binary with the same stdin/stdout contract works
 * (--aicf-runner CLI flag, ANIMICA_AICF_RUNNER env var).
 *
 * Runner stdin shape (one line of JSON):
 *   {
 *     "prompt":            "<full prompt text>",
 *     "tier":              "tiny" | "small" | "flagship" | "large",
 *     "max_output_tokens": int,
 *     "temperature":       float,
 *     "wallet_path":       "/root/.animica/wallets.json",
 *     "wallet_label":      "pool",
 *     "yolo":              true,
 *     "rpc_url":           "http://127.0.0.1:8545/rpc"
 *   }
 *
 * Runner stdout shape on success (one line of JSON):
 *   {
 *     "ok":            true,
 *     "content":       "<generated text>",
 *     "provider":      "distributed-aicf",
 *     "miner_id":      "anim1…",
 *     "tier":          "standard",
 *     "cost_animica":  0.001,
 *     "latency_ms":    32567,
 *     "source":        "aicf"
 *   }
 *
 * Runner stdout on failure (one line of JSON, exit code 1):
 *   {"ok": false, "error": "<reason>", "code": "TIMEOUT|SPAWN|…"}
 *
 * Status: this header is the wire-contract anchor. The CPP file wraps
 * the subprocess spawn with libuv (the same async runtime xmrig uses
 * for network I/O), so AICF jobs don't block the hash-share path.
 * Wiring this into EthStratumClient is the next commit.
 */
class AnimicaAicfDelegate
{
public:
    struct Result {
        bool        ok            = false;
        std::string content;
        std::string minerId;
        std::string error;
        std::string code;
        double      costAnimica   = 0.0;
        uint64_t    latencyMs     = 0;
    };

    AnimicaAicfDelegate();
    explicit AnimicaAicfDelegate(const std::string &runner_path);

    // Synchronously run one inference. Spawns the runner, writes the
    // spec to stdin, reads one JSON line from stdout, terminates the
    // subprocess. Blocks the calling thread for up to `timeout` — use
    // a dedicated worker thread or libuv async wrapper to avoid
    // blocking the Stratum I/O loop.
    Result run(const std::string &prompt,
               const std::string &tier,
               uint32_t max_output_tokens,
               double temperature,
               std::chrono::milliseconds timeout) const;

    // Best-effort: does the runner binary exist and look executable?
    // Used at startup to decide whether to advertise AICF features in
    // mining.subscribe (no point claiming tiers we can't serve).
    bool runnerAvailable() const;

    const std::string &runnerPath() const { return m_runnerPath; }

private:
    std::string m_runnerPath;
};


} // namespace xmrig


#endif // XMRIG_ANIMICAAICFDELEGATE_H
