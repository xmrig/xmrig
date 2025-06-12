/* XMRig
 * Copyright (c) 2018-2025 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2025 XMRig       <support@xmrig.com>
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

#include "backend/cpu/platform/BasicCpuInfo.h"

#include <sys/sysctl.h>


void xmrig::BasicCpuInfo::init_arm()
{
#   if __ARM_FEATURE_CRYPTO
    m_flags.set(FLAG_AES, true); // FIXME
#   endif

    size_t buflen = sizeof(m_brand);
    sysctlbyname("machdep.cpu.brand_string", &m_brand, &buflen, nullptr, 0);
}
