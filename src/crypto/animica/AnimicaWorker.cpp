/* XMRig
 * Copyright 2026      ercmine     <https://github.com/ercmine>
 * Copyright 2016-2024 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 */

#include "crypto/animica/AnimicaWorker.h"
#include "crypto/animica/AnimicaHash.h"

#include <cstring>


namespace xmrig {


// Compare a 32-byte big-endian digest against a 32-byte big-endian
// target. Returns true iff digest <= target. Walks byte-by-byte from
// the MSB; first differing byte decides. Compiles to ~6 instructions on
// x86 in practice (no SIMD needed — share rate is dominated by the
// keccakf permutation, not the compare).
static inline bool digest_le_target(const uint8_t digest[32], const uint8_t target[32])
{
    for (size_t i = 0; i < 32; ++i) {
        if (digest[i] < target[i]) return true;
        if (digest[i] > target[i]) return false;
    }
    return true;     // equal counts as a share
}


AnimicaWorker::AnimicaWorker() :
    m_stop(false),
    m_hashes(0)
{
}


void AnimicaWorker::setJob(const uint8_t *prefix, size_t prefix_len,
                            const uint8_t target[32],
                            uint64_t startNonce, uint64_t step)
{
    if (prefix_len > sizeof(m_prefix)) {
        prefix_len = sizeof(m_prefix);    // truncate defensively; real jobs are <80 bytes
    }
    std::memcpy(m_prefix, prefix, prefix_len);
    m_prefixLen   = prefix_len;
    std::memcpy(m_target, target, 32);
    m_startNonce  = startNonce;
    m_step        = step == 0 ? 1 : step;
    m_stop.store(false, std::memory_order_relaxed);
    m_hashes.store(0,   std::memory_order_relaxed);
}


uint64_t AnimicaWorker::run(ShareCallback onShare, void *callbackCtx)
{
    AnimicaSha3Prefix engine;
    engine.absorbPrefix(m_prefix, m_prefixLen);

    uint64_t shares = 0;
    uint64_t nonce  = m_startNonce;
    uint8_t  digest[32];
    uint64_t local_hashes = 0;
    // Flush the local counter to the atomic every kFlush trials. Reduces
    // atomic-op pressure on the hot path while keeping the
    // worker-summary hashrate within ~1 second of reality.
    constexpr uint64_t kFlush = 1 << 14;

    while (!m_stop.load(std::memory_order_relaxed)) {
        engine.hashNonce(nonce, digest);
        if (digest_le_target(digest, m_target)) {
            if (onShare) {
                onShare(callbackCtx, nonce, digest);
            }
            ++shares;
        }
        nonce += m_step;
        if (++local_hashes >= kFlush) {
            m_hashes.fetch_add(local_hashes, std::memory_order_relaxed);
            local_hashes = 0;
        }
    }
    m_hashes.fetch_add(local_hashes, std::memory_order_relaxed);
    return shares;
}


void AnimicaWorker::stop()
{
    m_stop.store(true, std::memory_order_release);
}


} // namespace xmrig
