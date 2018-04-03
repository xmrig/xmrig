/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2016-2018 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include <libcpuid.h>
#include <math.h>
#include <string.h>

#include "Cpu.h"


bool Cpu::m_l2_exclusive = false;
char Cpu::m_brand[64]    = { 0 };
int Cpu::m_flags         = 0;
int Cpu::m_l2_cache      = 0;
int Cpu::m_l3_cache      = 0;
int Cpu::m_sockets       = 1;
int Cpu::m_totalCores    = 0;
int Cpu::m_totalThreads  = 0;


int Cpu::optimalThreadsCount(xmrig::Algo algo, bool doubleHash, int maxCpuUsage)
{
    if (m_totalThreads == 1) {
        return 1;
    }

    int cache = 0;
    if (m_l3_cache) {
        cache = m_l2_exclusive ? (m_l2_cache + m_l3_cache) : m_l3_cache;
    }
    else {
        cache = m_l2_cache;
    }

    int count = 0;
    int size  = 2048;

    if (algo == xmrig::CRYPTONIGHT_LITE) {
        size = 1024;
    }
    else if (algo == xmrig::CRYPTONIGHT_HEAVY) {
        size = 4096;
    }

    if (doubleHash) {
        size *= 2;
    }

    if (cache) {
        count = cache / size;
    }
    else {
        count = m_totalThreads / 2;
    }

    if (count > m_totalThreads) {
        count = m_totalThreads;
    }

    if (((float) count / m_totalThreads * 100) > maxCpuUsage) {
        count = (int) ceil((float) m_totalThreads * (maxCpuUsage / 100.0));
    }

    return count < 1 ? 1 : count;
}


void Cpu::initCommon()
{
    struct cpu_raw_data_t raw = { 0 };
    struct cpu_id_t data = { 0 };

    cpuid_get_raw_data(&raw);
    cpu_identify(&raw, &data);

    strncpy(m_brand, data.brand_str, sizeof(m_brand) - 1);

    m_totalThreads = data.total_logical_cpus;
    m_sockets      = m_totalThreads / data.num_logical_cpus;

    if (m_sockets == 0) {
        m_sockets = 1;
    }

    m_totalCores = data.num_cores * m_sockets;
    m_l3_cache = data.l3_cache > 0 ? data.l3_cache * m_sockets : 0;

    // Workaround for AMD CPUs https://github.com/anrieff/libcpuid/issues/97
    if (data.vendor == VENDOR_AMD && data.ext_family >= 0x15 && data.ext_family < 0x17) {
        m_l2_cache = data.l2_cache * (m_totalCores / 2) * m_sockets;
        m_l2_exclusive = true;
    }
    // Workaround for Intel Pentium Dual-Core, Core Duo, Core 2 Duo, Core 2 Quad and their Xeon homologue
    // These processors have L2 cache shared by 2 cores.
    else if (data.vendor == VENDOR_INTEL && data.ext_family == 0x06 && (data.ext_model == 0x0E || data.ext_model == 0x0F || data.ext_model == 0x17)) {
        int l2_count_per_socket = m_totalCores > 1 ? m_totalCores / 2 : 1;
        m_l2_cache = data.l2_cache > 0 ? data.l2_cache * l2_count_per_socket * m_sockets : 0;
    }
    else{
        m_l2_cache = data.l2_cache > 0 ? data.l2_cache * m_totalCores * m_sockets : 0;
    }

#   if defined(__x86_64__) || defined(_M_AMD64)
    m_flags |= X86_64;
#   endif

    if (data.flags[CPU_FEATURE_AES]) {
        m_flags |= AES;
    }

    if (data.flags[CPU_FEATURE_BMI2]) {
        m_flags |= BMI2;
    }
}
