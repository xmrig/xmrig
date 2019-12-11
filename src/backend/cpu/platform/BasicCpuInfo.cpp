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

#include <algorithm>
#include <cstring>
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

#ifndef bit_PDPE1GB
#   define bit_PDPE1GB (1 << 26)
#endif


#include "backend/cpu/platform/BasicCpuInfo.h"
#include "crypto/common/Assembly.h"


#define VENDOR_ID                  (0)
#define PROCESSOR_INFO             (1)
#define EXTENDED_FEATURES          (7)
#define PROCESSOR_EXT_INFO         (0x80000001)
#define PROCESSOR_BRAND_STRING_1   (0x80000002)
#define PROCESSOR_BRAND_STRING_2   (0x80000003)
#define PROCESSOR_BRAND_STRING_3   (0x80000004)

#define EAX_Reg  (0)
#define EBX_Reg  (1)
#define ECX_Reg  (2)
#define EDX_Reg  (3)


namespace xmrig {


static inline void cpuid(uint32_t level, int32_t output[4])
{
    memset(output, 0, sizeof(int32_t) * 4);

#   ifdef _MSC_VER
    __cpuid(output, static_cast<int>(level));
#   else
    __cpuid_count(level, 0, output[0], output[1], output[2], output[3]);
#   endif
}


static void cpu_brand_string(char out[64 + 6]) {
    int32_t cpu_info[4] = { 0 };
    char buf[64]        = { 0 };

    cpuid(VENDOR_ID, cpu_info);

    if (cpu_info[EAX_Reg] >= 4) {
        for (uint32_t i = 0; i < 4; i++) {
            cpuid(0x80000002 + i, cpu_info);
            memcpy(buf + (i * 16), cpu_info, sizeof(cpu_info));
        }
    }

    size_t pos        = 0;
    const size_t size = strlen(buf);

    for (size_t i = 0; i < size; ++i) {
        if (buf[i] == ' ' && ((pos > 0 && out[pos - 1] == ' ') || pos == 0)) {
            continue;
        }

        out[pos++] = buf[i];
    }

    if (pos > 0 && out[pos - 1] == ' ') {
        out[pos - 1] = '\0';
    }
}


static inline bool has_feature(uint32_t level, uint32_t reg, int32_t bit)
{
    int32_t cpu_info[4] = { 0 };
    cpuid(level, cpu_info);

    return (cpu_info[reg] & bit) != 0;
}


static inline int32_t get_masked(int32_t val, int32_t h, int32_t l)
{
    val &= (0x7FFFFFFF >> (31 - (h - l))) << l;
    return val >> l;
}


static inline bool has_aes_ni()
{
    return has_feature(PROCESSOR_INFO, ECX_Reg, bit_AES);
}


static inline bool has_avx2()
{
    return has_feature(EXTENDED_FEATURES, EBX_Reg, bit_AVX2) && has_feature(PROCESSOR_INFO, ECX_Reg, bit_OSXSAVE);
}


static inline bool has_pdpe1gb()
{
    return has_feature(PROCESSOR_EXT_INFO, EDX_Reg, bit_PDPE1GB);
}


} // namespace xmrig


xmrig::BasicCpuInfo::BasicCpuInfo() :
    m_threads(std::thread::hardware_concurrency()),
    m_aes(has_aes_ni()),
    m_avx2(has_avx2()),
    m_pdpe1gb(has_pdpe1gb())
{
    cpu_brand_string(m_brand);

#   ifdef XMRIG_FEATURE_ASM
    if (hasAES()) {
        char vendor[13] = { 0 };
        int32_t data[4] = { 0 };

        cpuid(VENDOR_ID, data);

        memcpy(vendor + 0, &data[1], 4);
        memcpy(vendor + 4, &data[3], 4);
        memcpy(vendor + 8, &data[2], 4);

        if (memcmp(vendor, "AuthenticAMD", 12) == 0) {
            m_vendor = VENDOR_AMD;

            cpuid(PROCESSOR_INFO, data);
            const int32_t family = get_masked(data[EAX_Reg], 12, 8) + get_masked(data[EAX_Reg], 28, 20);

            m_assembly = family >= 23 ? Assembly::RYZEN : Assembly::BULLDOZER;
        }
        else if (memcmp(vendor, "GenuineIntel", 12) == 0) {
            m_vendor   = VENDOR_INTEL;
            m_assembly = Assembly::INTEL;
        }
    }
#   endif
}


const char *xmrig::BasicCpuInfo::backend() const
{
    return "basic";
}


xmrig::CpuThreads xmrig::BasicCpuInfo::threads(const Algorithm &algorithm, uint32_t) const
{
    const size_t count = std::thread::hardware_concurrency();

    if (count == 1) {
        return 1;
    }

#   ifdef XMRIG_ALGO_CN_GPU
    if (algorithm == Algorithm::CN_GPU) {
        return count;
    }
#   endif

#   ifdef XMRIG_ALGO_CN_LITE
    if (algorithm.family() == Algorithm::CN_LITE) {
        return CpuThreads(count, 1);
    }
#   endif

#   ifdef XMRIG_ALGO_CN_PICO
    if (algorithm.family() == Algorithm::CN_PICO) {
        return CpuThreads(count, 2);
    }
#   endif

#   ifdef XMRIG_ALGO_CN_HEAVY
    if (algorithm.family() == Algorithm::CN_HEAVY) {
        return CpuThreads(std::max<size_t>(count / 4, 1), 1);
    }
#   endif

#   ifdef XMRIG_ALGO_RANDOMX
    if (algorithm.family() == Algorithm::RANDOM_X) {
        if (algorithm == Algorithm::RX_WOW) {
            return count;
        }

        return std::max<size_t>(count / 2, 1);
    }
#   endif

#   ifdef XMRIG_ALGO_ARGON2
    if (algorithm.family() == Algorithm::ARGON2) {
        return count;
    }
#   endif

    return CpuThreads(std::max<size_t>(count / 2, 1), 1);
}
