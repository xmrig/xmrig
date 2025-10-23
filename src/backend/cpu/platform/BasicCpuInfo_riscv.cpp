/* XMRig
 * Copyright (c) 2025      Slayingripper <https://github.com/Slayingripper>
 * Copyright (c) 2018-2025 SChernykh     <https://github.com/SChernykh>
 * Copyright (c) 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright (c) 2016-2025 XMRig         <support@xmrig.com>
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
#include <thread>


#include "backend/cpu/platform/BasicCpuInfo.h"
#include "base/tools/String.h"
#include "3rdparty/rapidjson/document.h"


namespace xmrig {


extern String cpu_name_riscv();
extern bool has_riscv_vector();
extern bool has_riscv_crypto();


} // namespace xmrig


xmrig::BasicCpuInfo::BasicCpuInfo() :
    m_threads(std::thread::hardware_concurrency())
{
    m_units.resize(m_threads);
    for (int32_t i = 0; i < static_cast<int32_t>(m_threads); ++i) {
        m_units[i] = i;
    }

    memcpy(m_brand, "RISC-V", 6);

    auto name = cpu_name_riscv();
    if (!name.isNull()) {
        strncpy(m_brand, name.data(), sizeof(m_brand) - 1);
    }
    
    // Check for crypto extensions (Zknd/Zkne/Zknh - AES and SHA)
    m_flags.set(FLAG_AES, has_riscv_crypto());
    
    // RISC-V typically supports 1GB huge pages
    m_flags.set(FLAG_PDPE1GB, std::ifstream("/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages").good());
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
    out.AddMember("arch",       "riscv64", allocator);

    Value flags(kArrayType);

    if (hasAES()) {
        flags.PushBack("aes", allocator);
    }

    out.AddMember("flags", flags, allocator);

    return out;
}
