/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2024 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2024 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
 * Copyright 2026      ercmine     <https://github.com/ercmine>
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef XMRIG_ANIMICAHASH_H
#define XMRIG_ANIMICAHASH_H


#include <cstddef>
#include <cstdint>


namespace xmrig {


/**
 * NIST FIPS-202 SHA3-256.
 *
 * Animica's hashshare PoW hashes `prefix || nonce_le8` with NIST SHA3-256
 * (NOT legacy Keccak — the only difference is the padding byte: 0x06
 * for SHA3 vs 0x01 for the pre-standard Keccak that xmrig already
 * carries in src/base/crypto/keccak.cpp). Reusing the existing
 * `keccakf` permutation would be tempting but breaks for the very first
 * absorb block, so we keep this implementation self-contained.
 *
 * Implementation notes:
 *   - rate r = 1088 bits (136 bytes), capacity c = 512 bits
 *   - output md is exactly 32 bytes
 *   - input length is unbounded (16 EiB ceiling from uint64_t), but the
 *     mining hot path only ever hashes ~80–256 bytes
 *
 * The implementation lives in AnimicaHash.cpp; consumers include only
 * this header. Thread-safe: every call uses stack-local state.
 */
void sha3_256(const uint8_t *in, size_t inlen, uint8_t out[32]);


/**
 * Hot-loop variant: `sha3_256(prefix || nonce_le8, ...)` where the
 * prefix is already absorbed into a precomputed state. The miner
 * absorbs the per-job `prefix` once into the returned context, then
 * each nonce trial only re-runs the final block + squeeze (~7-10x
 * faster than re-hashing from scratch when prefix is large).
 *
 * The miner is expected to:
 *   1. AnimicaSha3Prefix p; p.absorbPrefix(prefix, prefix_len);
 *   2. for (nonce in [0, 2^64)): p.hashNonce(nonce, out32);
 *
 * `prefix` MUST be a whole-block multiple in length to keep the final
 * block at exactly 8 bytes (nonce) + padding. If the pool ever sends a
 * non-aligned prefix, AnimicaSha3Prefix transparently splits the
 * remainder into the per-nonce path.
 */
class AnimicaSha3Prefix
{
public:
    AnimicaSha3Prefix();

    // Absorb the per-job prefix. Safe to call multiple times to reset
    // for a new job — internal state is cleared first.
    void absorbPrefix(const uint8_t *prefix, size_t prefix_len);

    // Hash (prefix || nonce_le8) and write a 32-byte digest to `out`.
    // `nonce` is little-endian-serialized into 8 bytes per Animica spec.
    void hashNonce(uint64_t nonce, uint8_t out[32]) const;

private:
    // 5x5 lanes of 64 bits = 1600-bit state, post-prefix absorb.
    uint64_t m_state[25];
    // Bytes of the prefix that didn't fall on a 136-byte block boundary
    // and therefore have to be re-absorbed in the per-nonce path.
    uint8_t  m_tail[136];
    size_t   m_tailLen;
};


} // namespace xmrig


#endif // XMRIG_ANIMICAHASH_H
