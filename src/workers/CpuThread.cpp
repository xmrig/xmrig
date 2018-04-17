/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2016-2018 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include "net/Pool.h"
#include "rapidjson/document.h"
#include "workers/CpuThread.h"


#if defined(XMRIG_ARM)
#   include "crypto/CryptoNight_arm.h"
#else
#   include "crypto/CryptoNight_x86.h"
#endif


xmrig::CpuThread::CpuThread(size_t index, Algo algorithm, AlgoVariant av, Multiway multiway, int64_t affinity, int priority, bool softAES, bool prefetch) :
    m_algorithm(algorithm),
    m_av(av),
    m_prefetch(prefetch),
    m_softAES(softAES),
    m_priority(priority),
    m_affinity(affinity),
    m_multiway(multiway),
    m_index(index)
{
}


xmrig::CpuThread::~CpuThread()
{
}


bool xmrig::CpuThread::isSoftAES(AlgoVariant av)
{
    return av == AV_SINGLE_SOFT || av == AV_DOUBLE_SOFT || av > AV_PENTA;
}


xmrig::CpuThread::cn_hash_fun xmrig::CpuThread::fn(Algo algorithm, AlgoVariant av, Variant variant)
{
    assert(variant == VARIANT_NONE || variant == VARIANT_V1);

    static const cn_hash_fun func_table[50] = {
        cryptonight_single_hash<CRYPTONIGHT, false, VARIANT_NONE>,
        cryptonight_double_hash<CRYPTONIGHT, false, VARIANT_NONE>,
        cryptonight_single_hash<CRYPTONIGHT, true,  VARIANT_NONE>,
        cryptonight_double_hash<CRYPTONIGHT, true,  VARIANT_NONE>,
        cryptonight_triple_hash<CRYPTONIGHT, false, VARIANT_NONE>,
        cryptonight_quad_hash<CRYPTONIGHT,   false, VARIANT_NONE>,
        cryptonight_penta_hash<CRYPTONIGHT,  false, VARIANT_NONE>,
        cryptonight_triple_hash<CRYPTONIGHT, true,  VARIANT_NONE>,
        cryptonight_quad_hash<CRYPTONIGHT,   true,  VARIANT_NONE>,
        cryptonight_penta_hash<CRYPTONIGHT,  true,  VARIANT_NONE>,

        cryptonight_single_hash<CRYPTONIGHT, false, VARIANT_V1>,
        cryptonight_double_hash<CRYPTONIGHT, false, VARIANT_V1>,
        cryptonight_single_hash<CRYPTONIGHT, true,  VARIANT_V1>,
        cryptonight_double_hash<CRYPTONIGHT, true,  VARIANT_V1>,
        cryptonight_triple_hash<CRYPTONIGHT, false, VARIANT_V1>,
        cryptonight_quad_hash<CRYPTONIGHT,   false, VARIANT_V1>,
        cryptonight_penta_hash<CRYPTONIGHT,  false, VARIANT_V1>,
        cryptonight_triple_hash<CRYPTONIGHT, true,  VARIANT_V1>,
        cryptonight_quad_hash<CRYPTONIGHT,   true,  VARIANT_V1>,
        cryptonight_penta_hash<CRYPTONIGHT,  true,  VARIANT_V1>,

#       ifndef XMRIG_NO_AEON
        cryptonight_single_hash<CRYPTONIGHT_LITE, false, VARIANT_NONE>,
        cryptonight_double_hash<CRYPTONIGHT_LITE, false, VARIANT_NONE>,
        cryptonight_single_hash<CRYPTONIGHT_LITE, true,  VARIANT_NONE>,
        cryptonight_double_hash<CRYPTONIGHT_LITE, true,  VARIANT_NONE>,
        cryptonight_triple_hash<CRYPTONIGHT_LITE, false, VARIANT_NONE>,
        cryptonight_quad_hash<CRYPTONIGHT_LITE,   false, VARIANT_NONE>,
        cryptonight_penta_hash<CRYPTONIGHT_LITE,  false, VARIANT_NONE>,
        cryptonight_triple_hash<CRYPTONIGHT_LITE, true,  VARIANT_NONE>,
        cryptonight_quad_hash<CRYPTONIGHT_LITE,   true,  VARIANT_NONE>,
        cryptonight_penta_hash<CRYPTONIGHT_LITE,  true,  VARIANT_NONE>,

        cryptonight_single_hash<CRYPTONIGHT_LITE, false, VARIANT_V1>,
        cryptonight_double_hash<CRYPTONIGHT_LITE, false, VARIANT_V1>,
        cryptonight_single_hash<CRYPTONIGHT_LITE, true,  VARIANT_V1>,
        cryptonight_double_hash<CRYPTONIGHT_LITE, true,  VARIANT_V1>,
        cryptonight_triple_hash<CRYPTONIGHT_LITE, false, VARIANT_V1>,
        cryptonight_quad_hash<CRYPTONIGHT_LITE,   false, VARIANT_V1>,
        cryptonight_penta_hash<CRYPTONIGHT_LITE,  false, VARIANT_V1>,
        cryptonight_triple_hash<CRYPTONIGHT_LITE, true,  VARIANT_V1>,
        cryptonight_quad_hash<CRYPTONIGHT_LITE,   true,  VARIANT_V1>,
        cryptonight_penta_hash<CRYPTONIGHT_LITE,  true,  VARIANT_V1>,
#       else
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
#       endif

#       ifndef XMRIG_NO_SUMO
        cryptonight_single_hash<CRYPTONIGHT_HEAVY, false, VARIANT_NONE>,
        cryptonight_double_hash<CRYPTONIGHT_HEAVY, false, VARIANT_NONE>,
        cryptonight_single_hash<CRYPTONIGHT_HEAVY, true,  VARIANT_NONE>,
        cryptonight_double_hash<CRYPTONIGHT_HEAVY, true,  VARIANT_NONE>,
        cryptonight_triple_hash<CRYPTONIGHT_HEAVY, false, VARIANT_NONE>,
        cryptonight_quad_hash<CRYPTONIGHT_HEAVY,   false, VARIANT_NONE>,
        cryptonight_penta_hash<CRYPTONIGHT_HEAVY,  false, VARIANT_NONE>,
        cryptonight_triple_hash<CRYPTONIGHT_HEAVY, true,  VARIANT_NONE>,
        cryptonight_quad_hash<CRYPTONIGHT_HEAVY,   true,  VARIANT_NONE>,
        cryptonight_penta_hash<CRYPTONIGHT_HEAVY,  true,  VARIANT_NONE>,
#       else
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
#       endif
    };

#   ifndef XMRIG_NO_SUMO
    if (algorithm == CRYPTONIGHT_HEAVY) {
        variant = VARIANT_NONE;
    }
#   endif

    return func_table[20 * algorithm + 10 * variant + av - 1];
}


xmrig::CpuThread *xmrig::CpuThread::createFromAV(size_t index, Algo algorithm, AlgoVariant av, int64_t affinity, int priority)
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

    return new CpuThread(index, algorithm, av, multiway(av), cpuId, priority, isSoftAES(av), false);
}


xmrig::CpuThread *xmrig::CpuThread::createFromData(size_t index, Algo algorithm, const CpuThread::Data &data, int priority, bool softAES)
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

    return new CpuThread(index, algorithm, static_cast<AlgoVariant>(av), multiway, data.affinity, priority, softAES, false);
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


#ifndef XMRIG_NO_API
rapidjson::Value xmrig::CpuThread::toAPI(rapidjson::Document &doc) const
{
    using namespace rapidjson;

    Value obj(kObjectType);
    auto &allocator = doc.GetAllocator();

    obj.AddMember("type",          "cpu", allocator);
    obj.AddMember("algo",           rapidjson::StringRef(Pool::algoName(algorithm())), allocator);
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

    return obj;
}
