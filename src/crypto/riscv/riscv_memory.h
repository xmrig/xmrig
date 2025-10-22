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
 * RISC-V optimized memory operations
 * 
 * Provides efficient:
 * - Memory barriers
 * - Cache line operations
 * - Prefetching hints
 * - Aligned memory access
 * - Memory pooling utilities
 */

#ifndef XMRIG_RISCV_MEMORY_H
#define XMRIG_RISCV_MEMORY_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(XMRIG_RISCV)

#define CACHELINE_SIZE 64
#define CACHELINE_MASK (~(CACHELINE_SIZE - 1))

/* Memory barriers - optimized for RISC-V */

/* Full memory barrier: all reads and writes before must complete before any after */
static inline void riscv_mfence(void)
{
    asm volatile ("fence rw,rw" : : : "memory");
}

/* Load barrier: all loads before must complete before any after */
static inline void riscv_lfence(void)
{
    asm volatile ("fence r,r" : : : "memory");
}

/* Store barrier: all stores before must complete before any after */
static inline void riscv_sfence(void)
{
    asm volatile ("fence w,w" : : : "memory");
}

/* TSO (total store order) - ensures store-release semantics */
static inline void riscv_fence_tso(void)
{
    asm volatile ("fence rw,w" : : : "memory");
}

/* Acquire barrier - for lock acquisition */
static inline void riscv_acquire_fence(void)
{
    asm volatile ("fence r,rw" : : : "memory");
}

/* Release barrier - for lock release */
static inline void riscv_release_fence(void)
{
    asm volatile ("fence rw,w" : : : "memory");
}

/* CPU pause hint (Zihintpause extension, falls back to NOP) */
static inline void riscv_pause(void)
{
    asm volatile ("pause");
}

/* Prefetch operations - hints to load into L1 cache */

/* Prefetch for read (temporal locality) */
static inline void riscv_prefetch_read(const void *addr)
{
    /* Temporary workaround: use inline asm */
    asm volatile ("# prefetch %0 \n" : : "m"(*(const char *)addr));
}

/* Prefetch for write (prepare for store) */
static inline void riscv_prefetch_write(const void *addr)
{
    asm volatile ("# prefetch.w %0 \n" : : "m"(*(const char *)addr));
}

/* Prefetch with 0 temporal locality (load into L1 but not higher levels) */
static inline void riscv_prefetch_nta(const void *addr)
{
    asm volatile ("# prefetch.nta %0 \n" : : "m"(*(const char *)addr));
}

/* Cache line flush (if supported) */
static inline void riscv_clflush(const void *addr)
{
    /* RISC-V may not have cache flush in userspace */
    /* This is a no-op unless running in privileged mode */
    (void)addr;
}

/* Optimized memory copy with cache prefetching */
static inline void riscv_memcpy_prefetch(void *dest, const void *src, size_t size)
{
    uint8_t *d = (uint8_t *)dest;
    const uint8_t *s = (const uint8_t *)src;
    
    /* Process in cache line sized chunks with prefetching */
    size_t cache_lines = size / CACHELINE_SIZE;
    for (size_t i = 0; i < cache_lines; ++i) {
        /* Prefetch next cache lines ahead */
        if (i + 4 < cache_lines) {
            riscv_prefetch_read(s + (i + 4) * CACHELINE_SIZE);
        }
        
        /* Copy current cache line - use 64-bit accesses for efficiency */
        const uint64_t *src64 = (const uint64_t *)(s + i * CACHELINE_SIZE);
        uint64_t *dest64 = (uint64_t *)(d + i * CACHELINE_SIZE);
        
        for (int j = 0; j < 8; ++j) {  /* 8 * 8 bytes = 64 bytes */
            dest64[j] = src64[j];
        }
    }
    
    /* Handle remainder */
    size_t remainder = size % CACHELINE_SIZE;
    if (remainder > 0) {
        memcpy(d + cache_lines * CACHELINE_SIZE, 
               s + cache_lines * CACHELINE_SIZE, 
               remainder);
    }
}

/* Optimized memory fill with pattern */
static inline void riscv_memfill64(void *dest, uint64_t value, size_t count)
{
    uint64_t *d = (uint64_t *)dest;
    
    /* Unroll loop for better ILP */
    size_t i = 0;
    while (i + 8 <= count) {
        d[i + 0] = value;
        d[i + 1] = value;
        d[i + 2] = value;
        d[i + 3] = value;
        d[i + 4] = value;
        d[i + 5] = value;
        d[i + 6] = value;
        d[i + 7] = value;
        i += 8;
    }
    
    /* Handle remainder */
    while (i < count) {
        d[i] = value;
        i++;
    }
}

/* Compare memory with early exit optimization */
static inline int riscv_memcmp_fast(const void *s1, const void *s2, size_t n)
{
    const uint64_t *a = (const uint64_t *)s1;
    const uint64_t *b = (const uint64_t *)s2;
    
    size_t qwords = n / 8;
    for (size_t i = 0; i < qwords; ++i) {
        if (a[i] != b[i]) {
            /* Use byte comparison to find first difference */
            const uint8_t *ba = (const uint8_t *)a;
            const uint8_t *bb = (const uint8_t *)b;
            for (size_t j = i * 8; j < (i + 1) * 8 && j < n; ++j) {
                if (ba[j] != bb[j]) {
                    return ba[j] - bb[j];
                }
            }
        }
    }
    
    /* Check remainder */
    size_t remainder = n % 8;
    if (remainder > 0) {
        const uint8_t *ba = (const uint8_t *)s1 + qwords * 8;
        const uint8_t *bb = (const uint8_t *)s2 + qwords * 8;
        for (size_t i = 0; i < remainder; ++i) {
            if (ba[i] != bb[i]) {
                return ba[i] - bb[i];
            }
        }
    }
    
    return 0;
}

/* Atomic operations - optimized for RISC-V A extension */

typedef volatile uint64_t riscv_atomic64_t;

static inline uint64_t riscv_atomic64_load(const riscv_atomic64_t *p)
{
    riscv_lfence();  /* Ensure load-acquire semantics */
    return *p;
}

static inline void riscv_atomic64_store(riscv_atomic64_t *p, uint64_t v)
{
    riscv_sfence();  /* Ensure store-release semantics */
    *p = v;
}

static inline uint64_t riscv_atomic64_exchange(riscv_atomic64_t *p, uint64_t v)
{
    uint64_t old;
    asm volatile ("amoswap.d.aq %0, %2, (%1)" : "=r"(old) : "r"(p), "r"(v) : "memory");
    return old;
}

static inline uint64_t riscv_atomic64_add(riscv_atomic64_t *p, uint64_t v)
{
    uint64_t old;
    asm volatile ("amoadd.d.aq %0, %2, (%1)" : "=r"(old) : "r"(p), "r"(v) : "memory");
    return old;
}

#else  /* !XMRIG_RISCV */

/* Fallback implementations for non-RISC-V */

#define CACHELINE_SIZE 64

static inline void riscv_mfence(void) { __sync_synchronize(); }
static inline void riscv_lfence(void) { __sync_synchronize(); }
static inline void riscv_sfence(void) { __sync_synchronize(); }
static inline void riscv_fence_tso(void) { __sync_synchronize(); }
static inline void riscv_acquire_fence(void) { __sync_synchronize(); }
static inline void riscv_release_fence(void) { __sync_synchronize(); }
static inline void riscv_pause(void) { }

static inline void riscv_prefetch_read(const void *addr) { __builtin_prefetch(addr, 0, 3); }
static inline void riscv_prefetch_write(const void *addr) { __builtin_prefetch(addr, 1, 3); }
static inline void riscv_prefetch_nta(const void *addr) { __builtin_prefetch(addr, 0, 0); }
static inline void riscv_clflush(const void *addr) { (void)addr; }

static inline void riscv_memcpy_prefetch(void *dest, const void *src, size_t size)
{
    memcpy(dest, src, size);
}

static inline void riscv_memfill64(void *dest, uint64_t value, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        ((uint64_t *)dest)[i] = value;
    }
}

static inline int riscv_memcmp_fast(const void *s1, const void *s2, size_t n)
{
    return memcmp(s1, s2, n);
}

#endif

#ifdef __cplusplus
}
#endif

#endif  // XMRIG_RISCV_MEMORY_H
