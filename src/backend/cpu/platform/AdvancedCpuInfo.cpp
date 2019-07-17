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
#include <assert.h>
#include <libcpuid.h>
#include <math.h>
#include <string.h>


#include "backend/cpu/platform/AdvancedCpuInfo.h"


xmrig::AdvancedCpuInfo::AdvancedCpuInfo() :
    m_brand()
{
    struct cpu_raw_data_t raw = {};
    struct cpu_id_t data = {};

    cpuid_get_raw_data(&raw);
    cpu_identify(&raw, &data);

    strncpy(m_brand, data.brand_str, sizeof(m_brand));

    m_threads = static_cast<size_t>(data.total_logical_cpus);
    m_sockets = std::max<size_t>(threads() / static_cast<size_t>(data.num_logical_cpus), 1);
    m_cores   = static_cast<size_t>(data.num_cores) * m_sockets;
    m_L3      = data.l3_cache > 0 ? static_cast<size_t>(data.l3_cache) * m_sockets : 0;

    const size_t l2 = static_cast<size_t>(data.l2_cache);

    // Workaround for AMD CPUs https://github.com/anrieff/libcpuid/issues/97
    if (data.vendor == VENDOR_AMD && data.ext_family >= 0x15 && data.ext_family < 0x17) {
        m_L2 = l2 * (cores() / 2) * m_sockets;
        m_L2_exclusive = true;
    }
    // Workaround for Intel Pentium Dual-Core, Core Duo, Core 2 Duo, Core 2 Quad and their Xeon homologue
    // These processors have L2 cache shared by 2 cores.
    else if (data.vendor == VENDOR_INTEL && data.ext_family == 0x06 && (data.ext_model == 0x0E || data.ext_model == 0x0F || data.ext_model == 0x17)) {
        size_t l2_count_per_socket = cores() > 1 ? cores() / 2 : 1;
        m_L2 = data.l2_cache > 0 ? l2 * l2_count_per_socket * m_sockets : 0;
    }
    else{
        m_L2 = data.l2_cache > 0 ? l2 * cores() * m_sockets : 0;
    }

    if (data.flags[CPU_FEATURE_AES]) {
        m_aes = true;

        if (data.vendor == VENDOR_AMD) {
            m_assembly = (data.ext_family >= 23) ? Assembly::RYZEN : Assembly::BULLDOZER;
        }
        else if (data.vendor == VENDOR_INTEL) {
            m_assembly = Assembly::INTEL;
        }
    }

    m_avx2 = data.flags[CPU_FEATURE_AVX2] && data.flags[CPU_FEATURE_OSXSAVE];
}


xmrig::CpuThreads xmrig::AdvancedCpuInfo::threads(const Algorithm &algorithm) const
{
    if (threads() == 1) {
        return CpuThreads(1);
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
        cache *= 1024;
        const size_t memory = algorithm.memory();
        assert(memory > 0);

        count = cache / memory;

        if (cache % memory >= memory / 2) {
            count++;
        }
    }
    else {
        count = threads() / 2;
    }

    return CpuThreads(std::max<size_t>(std::min<size_t>(count, threads()), 1));
}
