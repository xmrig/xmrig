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


#include <string.h>


#include "Cpu.h"


char Cpu::m_brand[64]      = { 0 };
int Cpu::m_flags           = 0;
int Cpu::m_l2_cache        = 0;
int Cpu::m_l3_cache        = 0;
int Cpu::m_sockets         = 1;
int Cpu::m_totalCores      = 0;
size_t Cpu::m_totalThreads = 0;


size_t Cpu::optimalThreadsCount(size_t size, int maxCpuUsage)
{
    return m_totalThreads;
}


void Cpu::initCommon()
{
    memcpy(m_brand, "Unknown", 7);

#   if defined (__arm64__) || defined (__aarch64__)
    m_flags |= X86_64;
#   endif

#   if __ARM_FEATURE_CRYPTO
    m_flags |= AES;
#   endif
}
