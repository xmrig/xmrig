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
#include "base/tools/String.h"


#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>


#if __ARM_FEATURE_CRYPTO
#   include <sys/auxv.h>
#   if !defined(XMRIG_OS_FREEBSD)
#       include <asm/hwcap.h>
#   else
#       include <stdint.h>
#       include <machine/armreg.h>
#       ifndef ID_AA64ISAR0_AES_VAL
#           define ID_AA64ISAR0_AES_VAL ID_AA64ISAR0_AES
#       endif
#   endif
#endif


namespace xmrig {


extern String cpu_name_arm();


static void appendArmVersion(char *brand, size_t size)
{
#   if defined(XMRIG_OS_LINUX)
    bool hasA520 = false;
    bool hasA720 = false;

    for (unsigned int cpu = 0; cpu < 1024; ++cpu) {
        char path[160];

        std::snprintf(
            path,
            sizeof(path),
            "/sys/devices/system/cpu/cpu%u/regs/identification/midr_el1",
            cpu
        );

        std::ifstream file(path);

        if (!file.good()) {
            if (cpu >= std::thread::hardware_concurrency()) {
                break;
            }

            continue;
        }

        std::string value;
        file >> value;

        if (value.empty()) {
            continue;
        }

        unsigned long long midr = 0;

        try {
            midr = std::stoull(value, nullptr, 0);
        }
        catch (...) {
            continue;
        }

        const unsigned int implementer =
            static_cast<unsigned int>((midr >> 24) & 0xff);

        const unsigned int part =
            static_cast<unsigned int>((midr >> 4) & 0xfff);

        if (implementer != 0x41) {
            continue;
        }

        if (part == 0xd80) {
            hasA520 = true;
        }
        else if (part == 0xd81) {
            hasA720 = true;
        }
    }

    const char *suffix = nullptr;

    if (hasA520 && hasA720) {
        suffix = " Cortex-A520/A720 (ARMv9.2-A)";
    }
    else if (hasA520) {
        suffix = " Cortex-A520 (ARMv9.2-A)";
    }
    else if (hasA720) {
        suffix = " Cortex-A720 (ARMv9.2-A)";
    }

    if (suffix != nullptr) {
        std::snprintf(brand, size, "ARM%s", suffix);
    }
#   else
    (void) brand;
    (void) size;
#   endif
}


} // namespace xmrig


void xmrig::BasicCpuInfo::init_arm()
{
#   if __ARM_FEATURE_CRYPTO
#   if defined(XMRIG_OS_FREEBSD)
    uint64_t isar0 = READ_SPECIALREG(id_aa64isar0_el1);
    m_flags.set(
        FLAG_AES,
        ID_AA64ISAR0_AES_VAL(isar0) >= ID_AA64ISAR0_AES_BASE
    );
#   else
    m_flags.set(FLAG_AES, getauxval(AT_HWCAP) & HWCAP_AES);
#   endif
#   endif

#   if defined(XMRIG_OS_UNIX)
    auto name = cpu_name_arm();

    if (!name.isNull()) {
        std::strncpy(m_brand, name, sizeof(m_brand) - 1);
        m_brand[sizeof(m_brand) - 1] = '\0';
    }

    appendArmVersion(m_brand, sizeof(m_brand));

    m_flags.set(
        FLAG_PDPE1GB,
        std::ifstream(
            "/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages"
        ).good()
    );
#   endif
}
