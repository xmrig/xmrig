/* XMRig
 * Copyright 2026      ercmine     <https://github.com/ercmine>
 * Copyright 2016-2024 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 */

#ifndef XMRIG_ANIMICAWORKER_H
#define XMRIG_ANIMICAWORKER_H


#include <atomic>
#include <cstddef>
#include <cstdint>


namespace xmrig {


/**
 * Per-thread hot loop for Animica hashshare PoW.
 *
 * The Animica job consists of:
 *   - prefix: a per-job header derived from the Stratum job
 *     (version || prev_hash || merkle_root || ntime || nbits || …) that
 *     stays constant for every nonce trial within the job.
 *   - target: a 256-bit integer derived from the µ-nats share threshold;
 *     a digest is a valid share iff int256(digest) <= target.
 *
 * The worker is constructed once with the (prefix, target, startNonce,
 * step) tuple. `run()` blocks the calling thread, sweeping nonces and
 * invoking the share-submission callback for every digest <= target,
 * stopping cleanly when stop() is called (e.g. on new job notify).
 *
 * Status: this is the scaffold. The CPU backend integration in
 * src/backend/cpu/CpuWorker.cpp still needs to dispatch here when the
 * job algorithm == ANIMICA_SHA3. Tracking issue: TODO(ercmine).
 */
class AnimicaWorker
{
public:
    using ShareCallback = void (*)(void *ctx, uint64_t nonce, const uint8_t digest[32]);

    AnimicaWorker();

    // (Re)configure the worker for a fresh job. `prefix` is copied;
    // `target` is the 32-byte big-endian compare value.
    void setJob(const uint8_t *prefix, size_t prefix_len,
                const uint8_t target[32],
                uint64_t startNonce, uint64_t step);

    // Sweep nonces in [startNonce, ∞) with stride = step, calling
    // `onShare` whenever a digest is <= target. Returns the number of
    // shares submitted before stop().
    uint64_t run(ShareCallback onShare, void *callbackCtx);

    // Ask run() to return at the next loop iteration.
    void stop();

    // Hashes attempted since the last reset (cleared on setJob).
    uint64_t hashes() const { return m_hashes.load(std::memory_order_relaxed); }

private:
    uint8_t                  m_prefix[256];
    size_t                   m_prefixLen   = 0;
    uint8_t                  m_target[32]  = {};
    uint64_t                 m_startNonce  = 0;
    uint64_t                 m_step        = 1;
    std::atomic<bool>        m_stop;
    std::atomic<uint64_t>    m_hashes;
};


} // namespace xmrig


#endif // XMRIG_ANIMICAWORKER_H
