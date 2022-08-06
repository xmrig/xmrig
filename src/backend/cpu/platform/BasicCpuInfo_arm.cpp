/* XMRig
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig       <support@xmrig.com>
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

#include "base/tools/String.h"


#include <array>
#include <cstring>
#include <fstream>
#include <thread>


#if __ARM_FEATURE_CRYPTO && !defined(__APPLE__)
#   include <sys/auxv.h>
#   ifndef __FreeBSD__
#       include <asm/hwcap.h>
#   else
#       include <stdint.h>
#       include <machine/armreg.h>
#       ifndef ID_AA64ISAR0_AES_VAL
#           define ID_AA64ISAR0_AES_VAL ID_AA64ISAR0_AES
#       endif
#   endif
#endif


#include "backend/cpu/platform/BasicCpuInfo.h"
#include "3rdparty/rapidjson/document.h"


#if defined(XMRIG_OS_UNIX)
namespace xmrig {

extern String cpu_name_arm();

} // namespace xmrig
#elif defined(XMRIG_OS_MACOS)
#   include <sys/sysctl.h>
#endif


xmrig::BasicCpuInfo::BasicCpuInfo() :
    m_threads(std::thread::hardware_concurrency())
{
    m_units.resize(m_threads);
    for (int32_t i = 0; i < static_cast<int32_t>(m_threads); ++i) {
        m_units[i] = i;
    }

#   if (XMRIG_ARM == 8)
    memcpy(m_brand, "ARMv8", 5);
#   else
    memcpy(m_brand, "ARMv7", 5);
#   endif

#   if __ARM_FEATURE_CRYPTO
#   if defined(__APPLE__)
    m_flags.set(FLAG_AES, true);
#   elif defined(__FreeBSD__)
    uint64_t isar0 = READ_SPECIALREG(id_aa64isar0_el1);
    m_flags.set(FLAG_AES, ID_AA64ISAR0_AES_VAL(isar0) >= ID_AA64ISAR0_AES_BASE);
#   else
    m_flags.set(FLAG_AES, getauxval(AT_HWCAP) & HWCAP_AES);
#   endif
#   endif

#   if defined(XMRIG_OS_UNIX)
    auto name = cpu_name_arm();
    if (!name.isNull()) {
        strncpy(m_brand, name, sizeof(m_brand) - 1);
    }

    m_flags.set(FLAG_PDPE1GB, std::ifstream("/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages").good());
#   elif defined(XMRIG_OS_MACOS)
    size_t buflen = sizeof(m_brand);
    sysctlbyname("machdep.cpu.brand_string", &m_brand, &buflen, nullptr, 0);
#   endif
}


const char *xmrig::BasicCpuInfo::backend() const
{
    return "basic/1";
}


xmrig::CpuThreads xmrig::BasicCpuInfo::threads(const Algorithm &algorithm, uint32_t) const
{
#   ifdef XMRIG_ALGO_GHOSTRIDER
    if (algorithm.family() == Algorithm::GHOSTRIDER) {
        return CpuThreads(threads(), 8);
    }
#   endif

    return CpuThreads(threads());
}


rapidjson::Value xmrig::BasicCpuInfo::toJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    Value out(kObjectType);

    out.AddMember("brand",      StringRef(brand()), allocator);
    out.AddMember("aes",        hasAES(), allocator);
    out.AddMember("avx2",       false, allocator);
    out.AddMember("x64",        is64bit(), allocator); // DEPRECATED will be removed in the next major release.
    out.AddMember("64_bit",     is64bit(), allocator);
    out.AddMember("l2",         static_cast<uint64_t>(L2()), allocator);
    out.AddMember("l3",         static_cast<uint64_t>(L3()), allocator);
    out.AddMember("cores",      static_cast<uint64_t>(cores()), allocator);
    out.AddMember("threads",    static_cast<uint64_t>(threads()), allocator);
    out.AddMember("packages",   static_cast<uint64_t>(packages()), allocator);
    out.AddMember("nodes",      static_cast<uint64_t>(nodes()), allocator);
    out.AddMember("backend",    StringRef(backend()), allocator);
    out.AddMember("msr",        "none", allocator);
    out.AddMember("assembly",   "none", allocator);

#   if (XMRIG_ARM == 8)
    out.AddMember("arch", "aarch64", allocator);
#   else
    out.AddMember("arch", "aarch32", allocator);
#   endif

    Value flags(kArrayType);

    if (hasAES()) {
        flags.PushBack("aes", allocator);
    }

    out.AddMember("flags", flags, allocator);

    return out;
}
