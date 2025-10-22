/* XMRig
 * Copyright (c) 2025      Slayingripper <https://github.com/Slayingripper>
 * Copyright (c) 2018-2025 SChernykh     <https://github.com/SChernykh>
 * Copyright (c) 2016-2025 XMRig         <support@xmrig.com>
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
 * RISC-V Vector Extension (RVV) Optimizations for XMRig
 * 
 * Leverages RVV for parallel cryptographic operations
 * Automatically falls back to scalar if RVV unavailable
 */

#ifndef XMRIG_RISCV_RVV_H
#define XMRIG_RISCV_RVV_H

#include <riscv_vector.h>
#include <stdint.h>
#include <string.h>

#ifdef __riscv_v_elen
    #define XMRIG_RVV_ENABLED 1
    #define XMRIG_RVV_ELEN __riscv_v_elen
#else
    #define XMRIG_RVV_ENABLED 0
    #define XMRIG_RVV_ELEN 64
#endif

/* Vector length in bits */
#define RVV_VLEN __riscv_v_max_vlen

/* Detect VLEN at runtime if available */
static inline uint32_t riscv_rvv_vlen(void) {
#ifdef __riscv_v_max_vlen
    return __riscv_v_max_vlen;
#else
    /* Fallback: typical VLEN is 128, 256, or 512 bits */
    return 128;
#endif
}

/* Detect if RVV is available at runtime */
static inline int riscv_has_rvv(void) {
#ifdef __riscv_v
    return 1;
#else
    return 0;
#endif
}

#if XMRIG_RVV_ENABLED

/* Vectorized 64-bit memory copy using RVV
 * Copies 'size' bytes from src to dst using vector operations
 * Assumes size is multiple of vector element width
 */
static inline void riscv_memcpy_rvv(void *dst, const void *src, size_t size) {
    const uint8_t *s = (const uint8_t *)src;
    uint8_t *d = (uint8_t *)dst;
    
    /* Process in 64-byte chunks with RVV */
    size_t vl;
    uint64_t *d64 = (uint64_t *)dst;
    const uint64_t *s64 = (const uint64_t *)src;
    size_t count = size / 8;
    
    size_t i = 0;
    while (i < count) {
        vl = __riscv_vsetvl_e64m1(count - i);
        vfloat64m1_t vs = __riscv_vle64_v_f64m1((double *)(s64 + i), vl);
        __riscv_vse64_v_f64m1((double *)(d64 + i), vs, vl);
        i += vl;
    }
    
    /* Handle remainder */
    size_t remainder = size % 8;
    if (remainder) {
        memcpy((uint8_t *)dst + size - remainder, 
               (uint8_t *)src + size - remainder, 
               remainder);
    }
}

/* Vectorized memset using RVV - fill memory with pattern */
static inline void riscv_memset_rvv(void *dst, uint32_t pattern, size_t size) {
    uint32_t *d32 = (uint32_t *)dst;
    size_t count = size / 4;
    size_t vl, i = 0;
    
    while (i < count) {
        vl = __riscv_vsetvl_e32m1(count - i);
        vuint32m1_t vp = __riscv_vmv_v_x_u32m1(pattern, vl);
        __riscv_vse32_v_u32m1(d32 + i, vp, vl);
        i += vl;
    }
    
    /* Handle remainder */
    size_t remainder = size % 4;
    if (remainder) {
        memset((uint8_t *)dst + size - remainder, 
               pattern & 0xFF, 
               remainder);
    }
}

/* Vectorized XOR operation - a ^= b for size bytes */
static inline void riscv_xor_rvv(void *a, const void *b, size_t size) {
    uint64_t *a64 = (uint64_t *)a;
    const uint64_t *b64 = (const uint64_t *)b;
    size_t count = size / 8;
    size_t vl, i = 0;
    
    while (i < count) {
        vl = __riscv_vsetvl_e64m1(count - i);
        vuint64m1_t va = __riscv_vle64_v_u64m1(a64 + i, vl);
        vuint64m1_t vb = __riscv_vle64_v_u64m1(b64 + i, vl);
        vuint64m1_t vc = __riscv_vxor_vv_u64m1(va, vb, vl);
        __riscv_vse64_v_u64m1(a64 + i, vc, vl);
        i += vl;
    }
    
    /* Handle remainder */
    size_t remainder = size % 8;
    if (remainder) {
        uint8_t *a8 = (uint8_t *)a;
        const uint8_t *b8 = (const uint8_t *)b;
        for (size_t j = 0; j < remainder; j++) {
            a8[size - remainder + j] ^= b8[size - remainder + j];
        }
    }
}

/* Vectorized memory comparison - returns 0 if equal, first differing byte difference otherwise */
static inline int riscv_memcmp_rvv(const void *a, const void *b, size_t size) {
    const uint64_t *a64 = (const uint64_t *)a;
    const uint64_t *b64 = (const uint64_t *)b;
    size_t count = size / 8;
    size_t vl, i = 0;
    
    while (i < count) {
        vl = __riscv_vsetvl_e64m1(count - i);
        vuint64m1_t va = __riscv_vle64_v_u64m1(a64 + i, vl);
        vuint64m1_t vb = __riscv_vle64_v_u64m1(b64 + i, vl);
        vbool64_t cmp = __riscv_vmsne_vv_u64m1_b64(va, vb, vl);
        
        if (__riscv_vcpop_m_b64(cmp, vl) > 0) {
            /* Found difference, fall back to scalar for exact position */
            goto scalar_fallback;
        }
        i += vl;
    }
    
    /* Check remainder */
    size_t remainder = size % 8;
    if (remainder) {
        const uint8_t *a8 = (const uint8_t *)a;
        const uint8_t *b8 = (const uint8_t *)b;
        for (size_t j = 0; j < remainder; j++) {
            if (a8[size - remainder + j] != b8[size - remainder + j]) {
                return a8[size - remainder + j] - b8[size - remainder + j];
            }
        }
    }
    return 0;
    
scalar_fallback:
    return memcmp(a, b, size);
}

/* Vectorized 256-bit rotation for RandomX AES operations */
static inline void riscv_aes_rotate_rvv(uint32_t *data, size_t count) {
    /* Rotate 32-bit elements by 8 bits within 256-bit vectors */
    size_t vl, i = 0;
    
    while (i < count) {
        vl = __riscv_vsetvl_e32m1(count - i);
        vuint32m1_t v = __riscv_vle32_v_u32m1(data + i, vl);
        
        /* Rotate left by 8: (x << 8) | (x >> 24) */
        vuint32m1_t shifted_left = __riscv_vsll_vx_u32m1(v, 8, vl);
        vuint32m1_t shifted_right = __riscv_vsrl_vx_u32m1(v, 24, vl);
        vuint32m1_t result = __riscv_vor_vv_u32m1(shifted_left, shifted_right, vl);
        
        __riscv_vse32_v_u32m1(data + i, result, vl);
        i += vl;
    }
}

/* Parallel AES SubBytes operation using RVV */
static inline void riscv_aes_subbytes_rvv(uint8_t *state, size_t size) {
    /* This is a simplified version - real AES SubBytes uses lookup tables */
    size_t vl, i = 0;
    
    while (i < size) {
        vl = __riscv_vsetvl_e8m1(size - i);
        vuint8m1_t v = __riscv_vle8_v_u8m1(state + i, vl);
        
        /* Placeholder: in real implementation, use AES SBOX lookup */
        /* For now, just apply a simple transformation */
        vuint8m1_t result = __riscv_vxor_vx_u8m1(v, 0x63, vl);
        
        __riscv_vse8_v_u8m1(state + i, result, vl);
        i += vl;
    }
}

#else /* Scalar fallback when RVV unavailable */

static inline void riscv_memcpy_rvv(void *dst, const void *src, size_t size) {
    memcpy(dst, src, size);
}

static inline void riscv_memset_rvv(void *dst, uint32_t pattern, size_t size) {
    memset(dst, pattern & 0xFF, size);
}

static inline void riscv_xor_rvv(void *a, const void *b, size_t size) {
    uint8_t *a8 = (uint8_t *)a;
    const uint8_t *b8 = (const uint8_t *)b;
    for (size_t i = 0; i < size; i++) {
        a8[i] ^= b8[i];
    }
}

static inline int riscv_memcmp_rvv(const void *a, const void *b, size_t size) {
    return memcmp(a, b, size);
}

static inline void riscv_aes_rotate_rvv(uint32_t *data, size_t count) {
    for (size_t i = 0; i < count; i++) {
        data[i] = (data[i] << 8) | (data[i] >> 24);
    }
}

static inline void riscv_aes_subbytes_rvv(uint8_t *state, size_t size) {
    for (size_t i = 0; i < size; i++) {
        state[i] ^= 0x63;
    }
}

#endif /* XMRIG_RVV_ENABLED */

#endif /* XMRIG_RISCV_RVV_H */
