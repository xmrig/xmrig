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


#include "base/io/log/Log.h"
#include "common/cpu/Cpu.h"
#include "crypto/cn/CnHash.h"
#include "crypto/common/Assembly.h"
#include "crypto/common/VirtualMemory.h"
#include "Mem.h"
#include "rapidjson/document.h"
#include "workers/CpuThread.h"



static const xmrig::CnHash cnHash;


xmrig::CpuThread::CpuThread(size_t index, Algorithm algorithm, AlgoVariant av, Multiway multiway, int64_t affinity, int priority, bool softAES, bool prefetch, Assembly assembly) :
    m_algorithm(algorithm),
    m_av(av),
    m_assembly(assembly),
    m_prefetch(prefetch),
    m_softAES(softAES),
    m_priority(priority),
    m_affinity(affinity),
    m_multiway(multiway),
    m_index(index)
{
}


xmrig::cn_hash_fun xmrig::CpuThread::fn(const Algorithm &algorithm) const
{
    return cnHash.fn(algorithm, m_av, m_assembly);
}



bool xmrig::CpuThread::isSoftAES(AlgoVariant av)
{
    return av == AV_SINGLE_SOFT || av == AV_DOUBLE_SOFT || av > AV_PENTA;
}


xmrig::CpuThread *xmrig::CpuThread::createFromAV(size_t index, const Algorithm &algorithm, AlgoVariant av, int64_t affinity, int priority, Assembly assembly)
{
    assert(av > AV_AUTO && av < AV_MAX);

    int64_t cpuId = -1L;

    if (affinity != -1L) {
        size_t idx = 0;

        for (size_t i = 0; i < 64; i++) {
            if (!(affinity & (1ULL << i))) {
                continue;
            }

            if (idx == index) {
                cpuId = i;
                break;
            }

            idx++;
        }
    }

    return new CpuThread(index, algorithm, av, multiway(av), cpuId, priority, isSoftAES(av), false, assembly);
}


xmrig::CpuThread *xmrig::CpuThread::createFromData(size_t index, const Algorithm &algorithm, const CpuThread::Data &data, int priority, bool softAES)
{
    int av                  = AV_AUTO;
    const Multiway multiway = data.multiway;

    if (multiway <= DoubleWay) {
        av = softAES ? (multiway + 2) : multiway;
    }
    else {
        av = softAES ? (multiway + 5) : (multiway + 2);
    }

    assert(av > AV_AUTO && av < AV_MAX);

    return new CpuThread(index, algorithm, static_cast<AlgoVariant>(av), multiway, data.affinity, priority, softAES, false, data.assembly);
}


xmrig::CpuThread::Data xmrig::CpuThread::parse(const rapidjson::Value &object)
{
    Data data;

    const auto &multiway = object["low_power_mode"];
    if (multiway.IsBool()) {
        data.multiway = multiway.IsTrue() ? DoubleWay : SingleWay;
        data.valid    = true;
    }
    else if (multiway.IsUint()) {
        data.setMultiway(multiway.GetInt());
    }

    if (!data.valid) {
        return data;
    }

    const auto &affinity = object["affine_to_cpu"];
    if (affinity.IsUint64()) {
        data.affinity = affinity.GetInt64();
    }

#   ifdef XMRIG_FEATURE_ASM
    data.assembly = object["asm"];
#   endif

    return data;
}


xmrig::IThread::Multiway xmrig::CpuThread::multiway(AlgoVariant av)
{
    switch (av) {
    case AV_SINGLE:
    case AV_SINGLE_SOFT:
        return SingleWay;

    case AV_DOUBLE_SOFT:
    case AV_DOUBLE:
        return DoubleWay;

    case AV_TRIPLE_SOFT:
    case AV_TRIPLE:
        return TripleWay;

    case AV_QUAD_SOFT:
    case AV_QUAD:
        return QuadWay;

    case AV_PENTA_SOFT:
    case AV_PENTA:
        return PentaWay;

    default:
        break;
    }

    return SingleWay;
}


#ifdef APP_DEBUG
void xmrig::CpuThread::print() const
{
    LOG_DEBUG(GREEN_BOLD("CPU thread:   ") " index " WHITE_BOLD("%zu") ", multiway " WHITE_BOLD("%d") ", av " WHITE_BOLD("%d") ",",
              index(), static_cast<int>(multiway()), static_cast<int>(m_av));

#   ifdef XMRIG_FEATURE_ASM
    LOG_DEBUG("               assembly: %s, affine_to_cpu: %" PRId64, m_assembly.toString(), affinity());
#   else
    LOG_DEBUG("               affine_to_cpu: %" PRId64, affinity());
#   endif
}
#endif


#ifdef XMRIG_FEATURE_API
rapidjson::Value xmrig::CpuThread::toAPI(rapidjson::Document &doc) const
{
    using namespace rapidjson;

    Value obj(kObjectType);
    auto &allocator = doc.GetAllocator();

    obj.AddMember("type",          "cpu", allocator);
    obj.AddMember("av",             m_av, allocator);
    obj.AddMember("low_power_mode", multiway(), allocator);
    obj.AddMember("affine_to_cpu",  affinity(), allocator);
    obj.AddMember("priority",       priority(), allocator);
    obj.AddMember("soft_aes",       isSoftAES(), allocator);

    return obj;
}
#endif


rapidjson::Value xmrig::CpuThread::toConfig(rapidjson::Document &doc) const
{
    using namespace rapidjson;

    Value obj(kObjectType);
    auto &allocator = doc.GetAllocator();

    obj.AddMember("low_power_mode", multiway(), allocator);
    obj.AddMember("affine_to_cpu",  affinity() == -1L ? Value(kFalseType) : Value(affinity()), allocator);

#   ifdef XMRIG_FEATURE_ASM
    obj.AddMember("asm", m_assembly.toJSON(), allocator);
#   endif

    return obj;
}
