/* XMRig
 * Copyright (c) 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright (c) 2018-2026 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2026 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#include <cstring>
#include <fstream>
#include <thread>


#include "backend/cpu/platform/BasicCpuInfo.h"
#include "base/tools/String.h"
#include "3rdparty/rapidjson/document.h"


namespace xmrig {


extern String cpu_name_loongarch();
extern bool has_loongarch_lsx();
extern bool has_loongarch_lasx();


} // namespace xmrig


xmrig::BasicCpuInfo::BasicCpuInfo() :
    m_threads(std::thread::hardware_concurrency())
{
    m_units.resize(m_threads);
    for (int32_t i = 0; i < static_cast<int32_t>(m_threads); ++i) {
        m_units[i] = i;
    }

    memcpy(m_brand, "LoongArch", 10);

    auto name = cpu_name_loongarch();
    if (!name.isNull()) {
        strncpy(m_brand, name.data(), sizeof(m_brand) - 1);
    }

    m_flags.set(FLAG_AES, false);

#   ifdef XMRIG_FEATURE_LSX
    m_flags.set(FLAG_LSX, has_loongarch_lsx());
#   endif

#   ifdef XMRIG_FEATURE_LASX
    m_flags.set(FLAG_LASX, has_loongarch_lasx());
#   endif

    m_flags.set(FLAG_PDPE1GB, std::ifstream("/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages").good());
}


const char *xmrig::BasicCpuInfo::backend() const
{
    return "basic/1";
}


xmrig::CpuThreads xmrig::BasicCpuInfo::threads(const Algorithm &algorithm, uint32_t) const
{
    (void) algorithm;

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
    out.AddMember("arch",       "loongarch64", allocator);

    Value flags(kArrayType);

#   ifdef XMRIG_FEATURE_LSX
    if (has(FLAG_LSX)) {
        flags.PushBack("lsx", allocator);
    }
#   endif

#   ifdef XMRIG_FEATURE_LASX
    if (has(FLAG_LASX)) {
        flags.PushBack("lasx", allocator);
    }
#   endif

    out.AddMember("flags", flags, allocator);

    return out;
}
