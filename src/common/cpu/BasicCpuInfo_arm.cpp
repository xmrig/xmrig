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

#if __ARM_FEATURE_CRYPTO
#   include <sys/auxv.h>
#   include <asm/hwcap.h>
#endif


#include "common/cpu/BasicCpuInfo.h"


xmrig::BasicCpuInfo::BasicCpuInfo() :
    m_aes(false),
    m_avx2(false),
    m_brand(),
    m_threads(std::thread::hardware_concurrency())
{
#   ifdef XMRIG_ARMv8
    memcpy(m_brand, "ARMv8", 5);
#   else
    memcpy(m_brand, "ARMv7", 5);
#   endif

#   if __ARM_FEATURE_CRYPTO
    m_aes = getauxval(AT_HWCAP) & HWCAP_AES;
#   endif
}


size_t xmrig::BasicCpuInfo::optimalThreadsCount(size_t memSize) const
{
    return threads();
}
