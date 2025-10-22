/* XMRig
 * Copyright (c) 2025 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * RISC-V Crypto Extensions (Zbk*) Support
 * 
 * Supports detection and usage of RISC-V crypto extensions:
 * - Zkn: NIST approved cryptographic extensions (AES, SHA2, SHA3)
 * - Zknd/Zkne: AES decryption/encryption
 * - Zknh: SHA2/SHA3 hash extensions
 * - Zkb: Bit manipulation extensions (Zba, Zbb, Zbc, Zbs)
 * 
 * Falls back gracefully to software implementations on systems without support.
 */

#ifndef XMRIG_RISCV_CRYPTO_H
#define XMRIG_RISCV_CRYPTO_H

#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(XMRIG_RISCV)

/* Check if RISC-V crypto extensions are available at compile time */
#if defined(__riscv_zkne) || defined(__riscv_zknd)
#define HAVE_RISCV_AES 1
#else
#define HAVE_RISCV_AES 0
#endif

#if defined(__riscv_zknh)
#define HAVE_RISCV_SHA 1
#else
#define HAVE_RISCV_SHA 0
#endif

#if defined(__riscv_zba) && defined(__riscv_zbb) && defined(__riscv_zbc)
#define HAVE_RISCV_BIT_MANIP 1
#else
#define HAVE_RISCV_BIT_MANIP 0
#endif

/* Detect CPU support at runtime via /proc/cpuinfo */
extern bool riscv_cpu_has_aes_support(void);
extern bool riscv_cpu_has_sha_support(void);
extern bool riscv_cpu_has_bitmanip_support(void);

/* Software fallback AES utilities optimized for RISC-V */

/* AES S-box lookup - cache-friendly implementation */
typedef struct {
    uint32_t sbox_enc[256];
    uint32_t sbox_dec[256];
} riscv_aes_sbox_t;

extern const riscv_aes_sbox_t riscv_aes_tables;

/* Software AES encryption round optimized for RISC-V */
static inline uint32_t riscv_aes_enc_round(uint32_t input, const uint32_t *round_key)
{
    uint32_t result = 0;
    
    /* Unroll byte-by-byte lookups for better instruction-level parallelism */
    uint32_t b0 = (input >> 0) & 0xFF;
    uint32_t b1 = (input >> 8) & 0xFF;
    uint32_t b2 = (input >> 16) & 0xFF;
    uint32_t b3 = (input >> 24) & 0xFF;
    
    result = riscv_aes_tables.sbox_enc[b0] ^
             riscv_aes_tables.sbox_enc[b1] ^
             riscv_aes_tables.sbox_enc[b2] ^
             riscv_aes_tables.sbox_enc[b3];
    
    return result ^ (*round_key);
}

/* Bit rotation optimized for RISC-V */
static inline uint32_t riscv_rotr32(uint32_t x, int r)
{
#if defined(__riscv_zbb)
    /* Use RISC-V bit rotation if available */
    uint32_t result;
    asm volatile ("ror %0, %1, %2" : "=r"(result) : "r"(x), "r"(r) : );
    return result;
#else
    /* Scalar fallback */
    return (x >> r) | (x << (32 - r));
#endif
}

static inline uint64_t riscv_rotr64(uint64_t x, int r)
{
#if defined(__riscv_zbb)
    /* Use RISC-V bit rotation if available */
    uint64_t result;
    asm volatile ("ror %0, %1, %2" : "=r"(result) : "r"(x), "r"(r) : );
    return result;
#else
    /* Scalar fallback */
    return (x >> r) | (x << (64 - r));
#endif
}

/* Bit count operations optimized for RISC-V */
static inline int riscv_popcount(uint64_t x)
{
#if defined(__riscv_zbb)
    /* Use hardware popcount if available */
    int result;
    asm volatile ("cpop %0, %1" : "=r"(result) : "r"(x) : );
    return result;
#else
    /* Scalar fallback */
    return __builtin_popcountll(x);
#endif
}

static inline int riscv_ctz(uint64_t x)
{
#if defined(__riscv_zbb)
    /* Use hardware count trailing zeros if available */
    int result;
    asm volatile ("ctz %0, %1" : "=r"(result) : "r"(x) : );
    return result;
#else
    /* Scalar fallback */
    return __builtin_ctzll(x);
#endif
}

/* Bit manipulation operations from Zba */
static inline uint64_t riscv_add_uw(uint64_t a, uint64_t b)
{
#if defined(__riscv_zba)
    /* Add unsigned word (add.uw) - zero extends 32-bit addition */
    uint64_t result;
    asm volatile ("add.uw %0, %1, %2" : "=r"(result) : "r"(a), "r"(b) : );
    return result;
#else
    return ((a & 0xFFFFFFFF) + (b & 0xFFFFFFFF)) & 0xFFFFFFFF;
#endif
}

#else /* !XMRIG_RISCV */

/* Non-RISC-V fallbacks */
#define HAVE_RISCV_AES 0
#define HAVE_RISCV_SHA 0
#define HAVE_RISCV_BIT_MANIP 0

static inline bool riscv_cpu_has_aes_support(void) { return false; }
static inline bool riscv_cpu_has_sha_support(void) { return false; }
static inline bool riscv_cpu_has_bitmanip_support(void) { return false; }

static inline uint32_t riscv_rotr32(uint32_t x, int r) { return (x >> r) | (x << (32 - r)); }
static inline uint64_t riscv_rotr64(uint64_t x, int r) { return (x >> r) | (x << (64 - r)); }
static inline int riscv_popcount(uint64_t x) { return __builtin_popcountll(x); }
static inline int riscv_ctz(uint64_t x) { return __builtin_ctzll(x); }
static inline uint64_t riscv_add_uw(uint64_t a, uint64_t b) { return (a & 0xFFFFFFFF) + (b & 0xFFFFFFFF); }

#endif

#ifdef __cplusplus
}
#endif

#endif // XMRIG_RISCV_CRYPTO_H
