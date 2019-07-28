/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2019 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include <assert.h>


#include "backend/cpu/Cpu.h"
#include "rapidjson/document.h"


#if defined(XMRIG_FEATURE_HWLOC)
#   include "backend/cpu/platform/HwlocCpuInfo.h"
#elif defined(XMRIG_FEATURE_LIBCPUID)
#   include "backend/cpu/platform/AdvancedCpuInfo.h"
#else
#   include "backend/cpu/platform/BasicCpuInfo.h"
#endif


static xmrig::ICpuInfo *cpuInfo = nullptr;


xmrig::ICpuInfo *xmrig::Cpu::info()
{
    assert(cpuInfo != nullptr);

    return cpuInfo;
}


rapidjson::Value xmrig::Cpu::toJSON(rapidjson::Document &doc)
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    ICpuInfo *i = info();
    Value cpu(kObjectType);
    Assembly assembly(i->assembly());

    cpu.AddMember("brand",      StringRef(i->brand()), allocator);
    cpu.AddMember("aes",        i->hasAES(), allocator);
    cpu.AddMember("avx2",       i->hasAVX2(), allocator);
    cpu.AddMember("x64",        i->isX64(), allocator);
    cpu.AddMember("assembly",   StringRef(assembly.toString()), allocator);
    cpu.AddMember("l2",         static_cast<uint64_t>(i->L2()), allocator);
    cpu.AddMember("l3",         static_cast<uint64_t>(i->L3()), allocator);
    cpu.AddMember("cores",      static_cast<uint64_t>(i->cores()), allocator);
    cpu.AddMember("threads",    static_cast<uint64_t>(i->threads()), allocator);
    cpu.AddMember("packages",   static_cast<uint64_t>(i->packages()), allocator);
    cpu.AddMember("nodes",      static_cast<uint64_t>(i->nodes()), allocator);
    cpu.AddMember("backend",    StringRef(i->backend()), allocator);

    return cpu;
}


void xmrig::Cpu::init()
{
    assert(cpuInfo == nullptr);

#   if defined(XMRIG_FEATURE_HWLOC)
    cpuInfo = new HwlocCpuInfo();
#   elif defined(XMRIG_FEATURE_LIBCPUID)
    cpuInfo = new AdvancedCpuInfo();
#   else
    cpuInfo = new BasicCpuInfo();
#   endif
}


void xmrig::Cpu::release()
{
    assert(cpuInfo != nullptr);

    delete cpuInfo;
    cpuInfo = nullptr;
}
