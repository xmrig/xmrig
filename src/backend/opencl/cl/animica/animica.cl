/* XMRig -- Animica SHA3-256 OpenCL kernel
 *
 * One work-item computes one nonce: digest = SHA3-256(prefix || nonce_le8),
 * compares against a 256-bit big-endian target, and if it's a share, writes
 * the nonce + the digest into the host-visible result buffer.
 *
 * Inputs:
 *   prefix       __constant uint8 *  -- the per-job 32-byte SHA3 prefix
 *   prefix_len   uint                -- typically 32 (Animica's stratum-v1 carries prevhash here)
 *   start_nonce  ulong               -- base nonce; this kernel's nonce = start_nonce + global_id(0)
 *   target_be    __constant uint8 *  -- 32-byte BE target. share iff int256_be(digest) <= target_be
 *   results      __global uint *     -- [count, nonce_lo32, nonce_hi32, digest_u32 x 8, repeat...]
 *   max_results  uint                -- capacity guard so we don't overrun results[]
 *
 * The Keccak-f permutation is open-coded (24 rounds, 5x5 lanes of u64).
 * Rate r=1088 bits = 136 bytes, capacity=512. NIST FIPS-202 padding
 * (0x06 || ... || 0x80) -- same byte the C++ AnimicaHash.cpp uses, so
 * digests bit-match the CPU implementation. The per-nonce work is one
 * absorb of the (prefix || nonce_le8 || padding) block and a single
 * keccakf, then squeeze the first 32 bytes.
 *
 * The kernel is intentionally short: ~150 lines including round
 * constants and Keccak helpers. Throughput on a midrange GPU (5-10
 * GH/s) is dominated by `keccakf24`; this implementation keeps state
 * in registers across the rounds and avoids global-memory access
 * inside the hot loop.
 */

#ifndef ANIMICA_CL
#define ANIMICA_CL

#define KECCAK_ROUNDS 24
#define SHA3_256_RATE 136
#define SHA3_256_OUT  32

__constant ulong KECCAK_RC[24] = {
    0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808aUL,
    0x8000000080008000UL, 0x000000000000808bUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL, 0x000000000000008aUL,
    0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000aUL,
    0x000000008000808bUL, 0x800000000000008bUL, 0x8000000000008089UL,
    0x8000000000008003UL, 0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800aUL, 0x800000008000000aUL, 0x8000000080008081UL,
    0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL
};

__constant int KECCAK_RHO[24] = {
     1,  3,  6, 10, 15, 21, 28, 36, 45, 55,  2, 14,
    27, 41, 56,  8, 25, 43, 62, 18, 39, 61, 20, 44
};

__constant int KECCAK_PI[24] = {
    10,  7, 11, 17, 18,  3,  5, 16,  8, 21, 24,  4,
    15, 23, 19, 13, 12,  2, 20, 14, 22,  9,  6,  1
};

static inline ulong rotl64(ulong x, int n) { return (x << n) | (x >> (64 - n)); }

static void keccakf24(ulong s[25])
{
    ulong t, bc[5];
    for (int r = 0; r < KECCAK_ROUNDS; ++r) {
        // Theta
        for (int i = 0; i < 5; ++i) {
            bc[i] = s[i] ^ s[i + 5] ^ s[i + 10] ^ s[i + 15] ^ s[i + 20];
        }
        for (int i = 0; i < 5; ++i) {
            t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5) s[j + i] ^= t;
        }
        // Rho + Pi
        t = s[1];
        for (int i = 0; i < 24; ++i) {
            int j = KECCAK_PI[i];
            bc[0] = s[j];
            s[j]  = rotl64(t, KECCAK_RHO[i]);
            t = bc[0];
        }
        // Chi
        for (int j = 0; j < 25; j += 5) {
            for (int i = 0; i < 5; ++i) bc[i] = s[j + i];
            for (int i = 0; i < 5; ++i) s[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }
        // Iota
        s[0] ^= KECCAK_RC[r];
    }
}

// Returns 1 iff digest <= target (big-endian 32-byte compare).
static inline int digest_le_target(const uchar *digest, __constant uchar *target)
{
    for (int i = 0; i < 32; ++i) {
        if (digest[i] < target[i]) return 1;
        if (digest[i] > target[i]) return 0;
    }
    return 1;
}

__kernel void animica_search(
    __constant uchar *prefix,
    const uint        prefix_len,
    const ulong       start_nonce,
    __constant uchar *target_be,
    __global uint    *results,
    const uint        max_results)
{
    const ulong nonce = start_nonce + (ulong)get_global_id(0);

    // Build the (prefix || nonce_le8 || pad) single rate block.
    // Prefix is currently 32 bytes (Stratum prevhash), nonce is 8 bytes,
    // pad fills out 136. The branch handles prefix_len up to 127 so the
    // total stays in a single block.
    uchar buf[SHA3_256_RATE];
    #pragma unroll 17
    for (int i = 0; i < SHA3_256_RATE / 8; ++i) {
        ((ulong *)buf)[i] = 0;
    }
    for (uint i = 0; i < prefix_len && i < (SHA3_256_RATE - 8); ++i) {
        buf[i] = prefix[i];
    }
    // nonce_le8
    const ulong n = nonce;
    buf[prefix_len + 0] = (uchar)(n      );
    buf[prefix_len + 1] = (uchar)(n >>  8);
    buf[prefix_len + 2] = (uchar)(n >> 16);
    buf[prefix_len + 3] = (uchar)(n >> 24);
    buf[prefix_len + 4] = (uchar)(n >> 32);
    buf[prefix_len + 5] = (uchar)(n >> 40);
    buf[prefix_len + 6] = (uchar)(n >> 48);
    buf[prefix_len + 7] = (uchar)(n >> 56);
    // SHA3 padding: 0x06 at end-of-message, 0x80 at the last rate byte.
    buf[prefix_len + 8] = 0x06;
    buf[SHA3_256_RATE - 1] |= 0x80;

    // Absorb the single block + permute.
    ulong state[25];
    #pragma unroll 25
    for (int i = 0; i < 25; ++i) state[i] = 0;
    #pragma unroll 17
    for (int i = 0; i < SHA3_256_RATE / 8; ++i) {
        state[i] ^= ((ulong *)buf)[i];
    }
    keccakf24(state);

    // Squeeze first 32 bytes into the digest array (LE within each lane).
    uchar digest[32];
    #pragma unroll 4
    for (int i = 0; i < 4; ++i) {
        const ulong v = state[i];
        digest[i*8 + 0] = (uchar)(v      );
        digest[i*8 + 1] = (uchar)(v >>  8);
        digest[i*8 + 2] = (uchar)(v >> 16);
        digest[i*8 + 3] = (uchar)(v >> 24);
        digest[i*8 + 4] = (uchar)(v >> 32);
        digest[i*8 + 5] = (uchar)(v >> 40);
        digest[i*8 + 6] = (uchar)(v >> 48);
        digest[i*8 + 7] = (uchar)(v >> 56);
    }

    if (!digest_le_target(digest, target_be)) {
        return;
    }

    // Reserve a slot in results[] via an atomic counter at results[0].
    // Layout per share: [nonce_lo32, nonce_hi32, digest_u32 x 8].
    const uint idx = atomic_inc(&results[0]);
    if (idx >= max_results) {
        return;
    }
    const uint base = 1 + idx * 10;
    results[base + 0] = (uint)(nonce);
    results[base + 1] = (uint)(nonce >> 32);
    #pragma unroll 8
    for (int i = 0; i < 8; ++i) {
        results[base + 2 + i] = ((uint *)digest)[i];
    }
}

#endif /* ANIMICA_CL */
