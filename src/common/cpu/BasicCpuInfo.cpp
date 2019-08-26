/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2019 XMRig       <support@xmrig.com>
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

#include <string.h>
#include <thread>


#ifdef _MSC_VER
#   include <intrin.h>
#else
#   include <cpuid.h>
#endif

#ifndef bit_AES
#   define bit_AES (1 << 25)
#endif

#ifndef bit_OSXSAVE
#   define bit_OSXSAVE (1 << 27)
#endif

#ifndef bit_AVX2
#   define bit_AVX2 (1 << 5)
#endif


#include "common/cpu/BasicCpuInfo.h"


#define VENDOR_ID                  (0)
#define PROCESSOR_INFO             (1)
#define CACHE_TLB_DESCRIPTOR       (2)
#define EXTENDED_FEATURES          (7)
#define PROCESSOR_BRAND_STRING_1   (0x80000002)
#define PROCESSOR_BRAND_STRING_2   (0x80000003)
#define PROCESSOR_BRAND_STRING_3   (0x80000004)

#define EAX_Reg  (0)
#define EBX_Reg  (1)
#define ECX_Reg  (2)
#define EDX_Reg  (3)


#ifdef _MSC_VER
static inline void cpuid(int level, int output[4]) {
    __cpuid(output, level);
}
#else
static inline void cpuid(int level, int output[4]) {
    int a, b, c, d;
    __cpuid_count(level, 0, a, b, c, d);

    output[0] = a;
    output[1] = b;
    output[2] = c;
    output[3] = d;
}
#endif


static inline void cpu_brand_string(char* s) {
    int32_t cpu_info[4] = { 0 };
    cpuid(VENDOR_ID, cpu_info);

    if (cpu_info[EAX_Reg] >= 4) {
        for (int i = 0; i < 4; i++) {
            cpuid(0x80000002 + i, cpu_info);
            memcpy(s, cpu_info, sizeof(cpu_info));
            s += 16;
        }
    }
}


static inline bool has_aes_ni()
{
    int32_t cpu_info[4] = { 0 };
    cpuid(PROCESSOR_INFO, cpu_info);

    return (cpu_info[ECX_Reg] & bit_AES) != 0;
}


static inline bool has_avx2()
{
    int32_t cpu_info[4] = { 0 };
    cpuid(EXTENDED_FEATURES, cpu_info);

    return (cpu_info[EBX_Reg] & bit_AVX2) != 0;
}


static inline bool has_ossave()
{
    int32_t cpu_info[4] = { 0 };
    cpuid(PROCESSOR_INFO, cpu_info);

    return (cpu_info[ECX_Reg] & bit_OSXSAVE) != 0;
}


xmrig::BasicCpuInfo::BasicCpuInfo() :
    m_aes(has_aes_ni()),
    m_avx2(has_avx2() && has_ossave()),
    m_brand(),
    m_threads(std::thread::hardware_concurrency())
{
    cpu_brand_string(m_brand);

    if (hasAES()) {
        char vendor[13] = { 0 };
        int32_t data[4] = { 0 };

        cpuid(0, data);

        memcpy(vendor + 0, &data[1], 4);
        memcpy(vendor + 4, &data[3], 4);
        memcpy(vendor + 8, &data[2], 4);
    }
}


size_t xmrig::BasicCpuInfo::optimalThreadsCount(size_t memSize) const
{
    const size_t count = threads() / 2;

    return count < 1 ? 1 : count;
}
