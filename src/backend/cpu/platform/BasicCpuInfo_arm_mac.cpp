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

#include <cstdio>
#include <cstring>
#include <sys/sysctl.h>


void xmrig::BasicCpuInfo::init_arm()
{
    int aes = 0;
    size_t aesSize = sizeof(aes);

    if (sysctlbyname("hw.optional.arm.FEAT_AES",
                     &aes, &aesSize, nullptr, 0) == 0 && aes == 1) {
        m_flags.set(FLAG_AES, true);
    }

    size_t buflen = sizeof(m_brand);

    if (sysctlbyname("machdep.cpu.brand_string",
                     m_brand, &buflen, nullptr, 0) != 0) {
        std::snprintf(m_brand, sizeof(m_brand), "Apple ARM");
    }

    m_brand[sizeof(m_brand) - 1] = '\0';

    if (std::strncmp(m_brand, "Apple M4", 8) == 0 ||
        std::strncmp(m_brand, "Apple M5", 8) == 0) {
        constexpr const char suffix[] = " (ARMv9)";
        const size_t used = std::strlen(m_brand);

        if (used + sizeof(suffix) <= sizeof(m_brand)) {
            std::memcpy(m_brand + used, suffix, sizeof(suffix));
        }
    }
}
