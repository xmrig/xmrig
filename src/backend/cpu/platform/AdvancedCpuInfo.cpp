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

#include "backend/cpu/platform/AdvancedCpuInfo.h"
#include "3rdparty/libcpuid/libcpuid.h"


#ifdef _MSC_VER
#   include <intrin.h>
#else
#   include <cpuid.h>
#endif


#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>


namespace xmrig {


static inline void cpu_brand_string(char out[64], const char *in) {
    size_t pos        = 0;
    const size_t size = strlen(in);

    for (size_t i = 0; i < size; ++i) {
        if (in[i] == ' ' && ((pos > 0 && out[pos - 1] == ' ') || pos == 0)) {
            continue;
        }

        out[pos++] = in[i];
    }

    if (pos > 0 && out[pos - 1] == ' ') {
        out[pos - 1] = '\0';
    }
}


static inline void cpuid(uint32_t level, int32_t output[4])
{
    memset(output, 0, sizeof(int32_t) * 4);

#   ifdef _MSC_VER
    __cpuid(output, static_cast<int>(level));
#   else
    __cpuid_count(level, 0, output[0], output[1], output[2], output[3]);
#   endif
}


static inline bool has_feature(uint32_t level, uint32_t reg, int32_t bit)
{
    int32_t cpu_info[4] = { 0 };
    cpuid(level, cpu_info);

    return (cpu_info[reg] & bit) != 0;
}


static inline bool has_pdpe1gb()
{
    return has_feature(0x80000001, 3, 1 << 26);
}


} // namespace xmrig


xmrig::AdvancedCpuInfo::AdvancedCpuInfo() :
    m_pdpe1gb(has_pdpe1gb())
{
    struct cpu_raw_data_t raw = {};
    struct cpu_id_t data      = {};

    cpuid_get_raw_data(&raw);
    cpu_identify(&raw, &data);

    cpu_brand_string(m_brand, data.brand_str);
    snprintf(m_backend, sizeof m_backend, "libcpuid/%s", cpuid_lib_version());

    if (data.vendor == ::VENDOR_INTEL) {
        m_vendor = VENDOR_INTEL;
    }
    else if (data.vendor == ::VENDOR_AMD) {
        m_vendor = VENDOR_AMD;
    }

    m_threads  = static_cast<size_t>(data.total_logical_cpus);
    m_packages = std::max<size_t>(threads() / static_cast<size_t>(data.num_logical_cpus), 1);
    m_cores    = static_cast<size_t>(data.num_cores) * m_packages;
    m_L3       = data.l3_cache > 0 ? static_cast<size_t>(data.l3_cache) * m_packages : 0;

    const auto l2 = static_cast<size_t>(data.l2_cache);

    // Workaround for AMD CPUs https://github.com/anrieff/libcpuid/issues/97
    if (m_vendor == VENDOR_AMD && data.ext_family >= 0x15 && data.ext_family < 0x17) {
        m_L2 = l2 * (cores() / 2) * m_packages;
        m_L2_exclusive = true;
    }
    // Workaround for Intel Pentium Dual-Core, Core Duo, Core 2 Duo, Core 2 Quad and their Xeon homologue
    // These processors have L2 cache shared by 2 cores.
    else if (m_vendor == VENDOR_INTEL && data.ext_family == 0x06 && (data.ext_model == 0x0E || data.ext_model == 0x0F || data.ext_model == 0x17)) {
        size_t l2_count_per_socket = cores() > 1 ? cores() / 2 : 1;
        m_L2 = data.l2_cache > 0 ? l2 * l2_count_per_socket * m_packages : 0;
    }
    else{
        m_L2 = data.l2_cache > 0 ? l2 * cores() * m_packages : 0;
    }

    m_L2 *= 1024;
    m_L3 *= 1024;

    if (data.flags[CPU_FEATURE_AES]) {
        m_aes = true;

        if (m_vendor == VENDOR_AMD) {
            m_assembly = (data.ext_family >= 23) ? Assembly::RYZEN : Assembly::BULLDOZER;
        }
        else if (m_vendor == VENDOR_INTEL) {
            m_assembly = Assembly::INTEL;
        }
    }

    m_avx2 = data.flags[CPU_FEATURE_AVX2] && data.flags[CPU_FEATURE_OSXSAVE];
}


xmrig::CpuThreads xmrig::AdvancedCpuInfo::threads(const Algorithm &algorithm, uint32_t limit) const
{
    if (threads() == 1) {
        return 1;
    }

#   ifdef XMRIG_ALGO_CN_GPU
    if (algorithm == Algorithm::CN_GPU) {
        return CpuThreads(threads());
    }
#   endif

    size_t cache = 0;
    size_t count = 0;

    if (m_L3) {
        cache = m_L2_exclusive ? (m_L2 + m_L3) : m_L3;
    }
    else {
        cache = m_L2;
    }

    if (cache) {
        const size_t memory = algorithm.l3();
        assert(memory > 0);

        count = cache / memory;

        if (cache % memory >= memory / 2) {
            count++;
        }
    }
    else {
        count = threads() / 2;
    }

    uint32_t intensity = algorithm.maxIntensity() == 1 ? 0 : 1;

#   ifdef XMRIG_ALGO_CN_PICO
    if (algorithm == Algorithm::CN_PICO_0 && (count / cores()) >= 2) {
        intensity = 2;
    }
#   endif

    if (limit > 0 && limit < 100) {
        count = std::min(count, static_cast<size_t>(round(threads() * (limit / 100.0))));
    }
    else {
        count = std::min(count, threads());
    }

    return CpuThreads(std::max<size_t>(count, 1), intensity);
}
