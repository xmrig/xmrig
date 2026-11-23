/* XMRig
 * Copyright (c) 2018-2025 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2025 XMRig       <support@xmrig.com>
 * Copyright (c) 2026 PalindromicBreadLoaf  <palindromicbreadloaf@tuta.com>
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

#include <array>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>


#include "backend/cpu/platform/BasicCpuInfo.h"
#include "3rdparty/rapidjson/document.h"


#if defined(XMRIG_OS_LINUX)
#   include <sys/auxv.h>
#   if defined(__has_include)
#       if __has_include(<asm/cputable.h>)
#           include <asm/cputable.h>
#       endif
#   endif
#endif


namespace xmrig {


static void read_cpu_brand(char *out, size_t out_len)
{
    std::ifstream f("/proc/cpuinfo");
    if (!f.is_open()) {
        return;
    }

    std::string line;
    while (std::getline(f, line)) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }

        std::string key = line.substr(0, colon);
        while (!key.empty() && (key.back() == ' ' || key.back() == '\t')) {
            key.pop_back();
        }

        if (key == "cpu" || key == "model name" || key == "Processor") {
            std::string value = line.substr(colon + 1);
            size_t start = value.find_first_not_of(" \t");
            if (start != std::string::npos) {
                value.erase(0, start);
                strncpy(out, value.c_str(), out_len - 1);
                out[out_len - 1] = '\0';
            }
            return;
        }
    }
}


} // namespace xmrig


xmrig::BasicCpuInfo::BasicCpuInfo() :
    m_threads(std::thread::hardware_concurrency())
{
    m_units.resize(m_threads);
    for (int32_t i = 0; i < static_cast<int32_t>(m_threads); ++i) {
        m_units[i] = i;
    }

#   if defined(XMRIG_PPC_BITS) && (XMRIG_PPC_BITS == 64)
#       if defined(__LITTLE_ENDIAN__) || (defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__))
    memcpy(m_brand, "PowerPC64LE", 11);
#       else
    memcpy(m_brand, "PowerPC64", 9);
#       endif
#   else
    memcpy(m_brand, "PowerPC", 7);
#   endif

    read_cpu_brand(m_brand, sizeof(m_brand));

#   if defined(XMRIG_OS_LINUX) && defined(AT_HWCAP2) && defined(PPC_FEATURE2_VEC_CRYPTO)
    const unsigned long hwcap2 = getauxval(AT_HWCAP2);
    if (hwcap2 & PPC_FEATURE2_VEC_CRYPTO) {
        m_flags.set(FLAG_AES, true);
    }
#   endif

#   if defined(XMRIG_OS_LINUX)
    m_flags.set(FLAG_PDPE1GB, std::ifstream("/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages").good());
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

#   if defined(XMRIG_PPC_BITS) && (XMRIG_PPC_BITS == 64)
#       if defined(__LITTLE_ENDIAN__) || (defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__))
    out.AddMember("arch", "ppc64le", allocator);
#       else
    out.AddMember("arch", "ppc64", allocator);
#       endif
#   else
    out.AddMember("arch", "ppc", allocator);
#   endif

    Value flags(kArrayType);

    if (hasAES()) {
        flags.PushBack("aes", allocator);
    }

    out.AddMember("flags", flags, allocator);

    return out;
}
