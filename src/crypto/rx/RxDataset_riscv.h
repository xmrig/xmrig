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
 * RISC-V optimized RandomX dataset initialization
 * Optimizations:
 * - Adaptive thread allocation based on CPU cores
 * - Prefetch hints for better cache utilization
 * - Memory alignment optimizations for RISC-V
 * - Efficient barrier operations
 */

#ifndef XMRIG_RXDATASET_RISCV_H
#define XMRIG_RXDATASET_RISCV_H

#include <stdint.h>
#include <unistd.h>
#include <sched.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(XMRIG_RISCV)

/* RISC-V memory prefetch macros */
#define PREFETCH_READ(addr)  asm volatile ("prefetch.r %0" : : "r"(addr) : "memory")
#define PREFETCH_WRITE(addr) asm volatile ("prefetch.w %0" : : "r"(addr) : "memory")
#define MEMORY_BARRIER()     asm volatile ("fence rw,rw" : : : "memory")
#define READ_BARRIER()       asm volatile ("fence r,r" : : : "memory")
#define WRITE_BARRIER()      asm volatile ("fence w,w" : : : "memory")

/* RISC-V hint pause - tries Zihintpause, falls back to NOP */
static inline void cpu_pause(void)
{
    asm volatile ("pause");
}

/* Adaptive thread count calculation for dataset init */
static inline uint32_t riscv_optimal_init_threads(uint32_t available_threads)
{
    /* On RISC-V, use 60-75% of available threads for init */
    /* This leaves some threads available for OS/other tasks */
    uint32_t recommended = (available_threads * 3) / 4;
    return recommended > 0 ? recommended : 1;
}

/* Prefetch next dataset item for better cache utilization */
static inline void prefetch_dataset_item(const void *item, size_t size)
{
    const uint8_t *ptr = (const uint8_t *)item;
    /* Prefetch cache line aligned chunks */
    for (size_t i = 0; i < size; i += 64) {
        PREFETCH_READ(ptr + i);
    }
}

/* Cache-aware aligned memory copy optimized for RISC-V */
static inline void aligned_memcpy_opt(void *dst, const void *src, size_t size)
{
    uint64_t *d = (uint64_t *)dst;
    const uint64_t *s = (const uint64_t *)src;
    
    /* Process in 64-byte chunks with prefetching */
    size_t chunks = size / 8;
    for (size_t i = 0; i < chunks; i += 8) {
        if (i + 8 < chunks) {
            prefetch_dataset_item(s + i + 8, 64);
        }
        d[i] = s[i];
        d[i+1] = s[i+1];
        d[i+2] = s[i+2];
        d[i+3] = s[i+3];
        d[i+4] = s[i+4];
        d[i+5] = s[i+5];
        d[i+6] = s[i+6];
        d[i+7] = s[i+7];
    }
}

/* Get optimal CPU core for thread pinning */
static inline int get_optimal_cpu_core(int thread_id)
{
    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    if (nprocs <= 0) nprocs = 1;
    return thread_id % nprocs;
}

#else /* !XMRIG_RISCV */

/* Fallback for non-RISC-V architectures */
#define PREFETCH_READ(addr)
#define PREFETCH_WRITE(addr)
#define MEMORY_BARRIER() __sync_synchronize()
#define READ_BARRIER()   __sync_synchronize()
#define WRITE_BARRIER()  __sync_synchronize()

static inline void cpu_pause(void) { }
static inline uint32_t riscv_optimal_init_threads(uint32_t available) { return available; }
static inline void prefetch_dataset_item(const void *item, size_t size) { (void)item; (void)size; }
static inline void aligned_memcpy_opt(void *dst, const void *src, size_t size) { memcpy(dst, src, size); }
static inline int get_optimal_cpu_core(int thread_id) { return thread_id; }

#endif

#ifdef __cplusplus
}
#endif

#endif // XMRIG_RXDATASET_RISCV_H
