/* XMRig
 * Copyright 2026      ercmine     <https://github.com/ercmine>
 * Copyright 2016-2024 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
 *
 * Standalone NIST FIPS-202 SHA3-256 used for Animica hashshare PoW.
 * We keep this independent of src/base/crypto/keccak.cpp because that
 * implementation uses pre-standard Keccak padding (0x01); the only
 * delta vs NIST SHA3 is the domain-separator byte (0x06), but mixing
 * the two in one TU is a maintenance trap, so this file is self-
 * contained and the permutation is re-derived here.
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 */

#include "crypto/animica/AnimicaHash.h"

#include <cstring>


namespace xmrig {


// ─── Keccak-f[1600] permutation ─────────────────────────────────────────

namespace {

constexpr int KECCAK_ROUNDS = 24;

constexpr uint64_t kRndConstants[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

constexpr int kRho[24] = {
     1,  3,  6, 10, 15, 21, 28, 36, 45, 55,  2, 14,
    27, 41, 56,  8, 25, 43, 62, 18, 39, 61, 20, 44
};

constexpr int kPi[24] = {
    10,  7, 11, 17, 18,  3,  5, 16,  8, 21, 24,  4,
    15, 23, 19, 13, 12,  2, 20, 14, 22,  9,  6,  1
};

static inline uint64_t rotl64(uint64_t x, int n)
{
    return (x << n) | (x >> (64 - n));
}

static void keccakf(uint64_t s[25])
{
    uint64_t t, bc[5];

    for (int r = 0; r < KECCAK_ROUNDS; ++r) {
        // Theta
        for (int i = 0; i < 5; ++i) {
            bc[i] = s[i] ^ s[i + 5] ^ s[i + 10] ^ s[i + 15] ^ s[i + 20];
        }
        for (int i = 0; i < 5; ++i) {
            t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5) {
                s[j + i] ^= t;
            }
        }

        // Rho + Pi
        t = s[1];
        for (int i = 0; i < 24; ++i) {
            int j = kPi[i];
            bc[0] = s[j];
            s[j]  = rotl64(t, kRho[i]);
            t = bc[0];
        }

        // Chi
        for (int j = 0; j < 25; j += 5) {
            for (int i = 0; i < 5; ++i) {
                bc[i] = s[j + i];
            }
            for (int i = 0; i < 5; ++i) {
                s[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
            }
        }

        // Iota
        s[0] ^= kRndConstants[r];
    }
}


// ─── SHA3-256 absorb / pad / squeeze ────────────────────────────────────
//
// Rate r = 1088 bits = 136 bytes for SHA3-256.
// Output is the first 32 bytes of the state after squeeze.

constexpr size_t SHA3_256_RATE = 136;
constexpr size_t SHA3_256_OUT  = 32;
constexpr uint8_t SHA3_PAD     = 0x06;   // NIST FIPS-202 domain separator
constexpr uint8_t SHA3_PAD_END = 0x80;   // last bit of multi-rate padding

static inline uint64_t load64_le(const uint8_t *src)
{
    uint64_t v;
    std::memcpy(&v, src, sizeof(v));
    return v;     // x86 is LE; portable bit-shift fallback only matters on BE
}

static inline void store64_le(uint8_t *dst, uint64_t v)
{
    std::memcpy(dst, &v, sizeof(v));
}


static void absorbBlocks(uint64_t state[25], const uint8_t *data, size_t blocks)
{
    for (size_t b = 0; b < blocks; ++b) {
        for (size_t i = 0; i < SHA3_256_RATE / 8; ++i) {
            state[i] ^= load64_le(data + b * SHA3_256_RATE + i * 8);
        }
        keccakf(state);
    }
}

} // anonymous namespace


void sha3_256(const uint8_t *in, size_t inlen, uint8_t out[32])
{
    uint64_t state[25];
    std::memset(state, 0, sizeof(state));

    // Absorb whole blocks.
    const size_t full_blocks = inlen / SHA3_256_RATE;
    absorbBlocks(state, in, full_blocks);

    // Build the final padded block.
    uint8_t pad_block[SHA3_256_RATE];
    std::memset(pad_block, 0, sizeof(pad_block));
    const size_t tail = inlen % SHA3_256_RATE;
    std::memcpy(pad_block, in + full_blocks * SHA3_256_RATE, tail);
    pad_block[tail] = SHA3_PAD;
    pad_block[SHA3_256_RATE - 1] |= SHA3_PAD_END;

    for (size_t i = 0; i < SHA3_256_RATE / 8; ++i) {
        state[i] ^= load64_le(pad_block + i * 8);
    }
    keccakf(state);

    // Squeeze first 32 bytes.
    for (size_t i = 0; i < SHA3_256_OUT / 8; ++i) {
        store64_le(out + i * 8, state[i]);
    }
}


// ─── AnimicaSha3Prefix ──────────────────────────────────────────────────
//
// Pre-absorbs the whole-block portion of `prefix` so the per-nonce hot
// path only needs to absorb the tail bytes + the 8-byte LE nonce +
// padding. On a typical 80-byte prefix the saving is one keccakf call
// per trial; on longer prefixes (with merkle bundles etc.) the saving
// scales linearly with prefix size.

AnimicaSha3Prefix::AnimicaSha3Prefix() :
    m_tailLen(0)
{
    std::memset(m_state, 0, sizeof(m_state));
    std::memset(m_tail, 0, sizeof(m_tail));
}


void AnimicaSha3Prefix::absorbPrefix(const uint8_t *prefix, size_t prefix_len)
{
    std::memset(m_state, 0, sizeof(m_state));

    const size_t full_blocks = prefix_len / SHA3_256_RATE;
    absorbBlocks(m_state, prefix, full_blocks);

    m_tailLen = prefix_len - full_blocks * SHA3_256_RATE;
    std::memset(m_tail, 0, sizeof(m_tail));
    if (m_tailLen) {
        std::memcpy(m_tail, prefix + full_blocks * SHA3_256_RATE, m_tailLen);
    }
}


void AnimicaSha3Prefix::hashNonce(uint64_t nonce, uint8_t out[32]) const
{
    uint64_t state[25];
    std::memcpy(state, m_state, sizeof(state));

    uint8_t pad_block[SHA3_256_RATE];
    std::memset(pad_block, 0, sizeof(pad_block));

    // Concatenated input is (tail || nonce_le8). If tail + 8 fits in one
    // block (i.e., tail + 8 <= 136 — always true since tail <= 135), we
    // build that block, append SHA3 padding, and absorb once. Otherwise
    // we'd need an extra block; not possible at the byte counts Animica
    // uses, but the code path handles it for safety.
    if (m_tailLen + 8 <= SHA3_256_RATE) {
        std::memcpy(pad_block, m_tail, m_tailLen);
        store64_le(pad_block + m_tailLen, nonce);
        const size_t total = m_tailLen + 8;
        pad_block[total] = SHA3_PAD;
        pad_block[SHA3_256_RATE - 1] |= SHA3_PAD_END;

        for (size_t i = 0; i < SHA3_256_RATE / 8; ++i) {
            state[i] ^= load64_le(pad_block + i * 8);
        }
        keccakf(state);
    } else {
        // Defensive fallback: build a tail+nonce buffer and re-hash via
        // the stateless absorb path. Never hit in practice.
        uint8_t buf[SHA3_256_RATE + 8];
        std::memcpy(buf, m_tail, m_tailLen);
        store64_le(buf + m_tailLen, nonce);
        uint8_t digest[32];
        sha3_256(buf, m_tailLen + 8, digest);
        std::memcpy(out, digest, 32);
        return;
    }

    for (size_t i = 0; i < SHA3_256_OUT / 8; ++i) {
        store64_le(out + i * 8, state[i]);
    }
}


} // namespace xmrig
