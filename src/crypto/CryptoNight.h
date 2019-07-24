/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
 *
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

#ifndef __CRYPTONIGHT_H__
#define __CRYPTONIGHT_H__


#include <stddef.h>
#include <stdint.h>

#define ONE_MB       1048576

#define MEMORY       2097152 /* 2 MiB in bytes*/
#define MEMORY_LITE  1048576 /* 1 MiB in bytes */
#define MEMORY_SUPER_LITE  524288 /* 512 KiB in bytes */
#define MEMORY_ULTRA_LITE  262144 /* 256 KiB in bytes */
#define MEMORY_EXTREME_LITE  131072 /* 128 KiB in bytes */
#define MEMORY_HEAVY 4194304 /* 4 MiB in bytes */

#define POW_DEFAULT_INDEX_SHIFT 3
#define POW_XLT_V4_INDEX_SHIFT 4

#if defined _MSC_VER || defined XMRIG_ARM
#define ABI_ATTRIBUTE
#else
#define ABI_ATTRIBUTE __attribute__((ms_abi))
#endif

struct ScratchPad;
typedef void(*cn_mainloop_fun_ms_abi)(ScratchPad*) ABI_ATTRIBUTE;
typedef void(*cn_mainloop_double_fun_ms_abi)(ScratchPad*, ScratchPad*) ABI_ATTRIBUTE;

struct cryptonight_r_data {
    int variant;
    uint64_t height;

    bool match(const int v, const uint64_t h) const { return (v == variant) && (h == height); }
};

struct ScratchPad {
    alignas(16) uint8_t state[224];
    alignas(16) uint8_t* memory;

    // Additional stuff for asm impl
    uint8_t ctx_info[24];
    const void* input;
    uint8_t* variant_table;
    const uint32_t* t_fn;

    cn_mainloop_fun_ms_abi generated_code;
    cn_mainloop_double_fun_ms_abi generated_code_double;
    cryptonight_r_data generated_code_data;
    cryptonight_r_data generated_code_double_data;
};

alignas(64) static uint8_t variant1_table[256];
alignas(64) static uint8_t variant_xtl_table[256];


#endif /* __CRYPTONIGHT_H__ */
