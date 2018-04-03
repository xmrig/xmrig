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


#include "core/CommonConfig.h"
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

    Multiway multiway = SingleWay;
    bool softAES = false;

    switch (av) {
    case AV_SINGLE_SOFT:
        softAES  = true;
        break;

    case AV_DOUBLE:
        multiway = DoubleWay;
    case AV_DOUBLE_SOFT:
        softAES  = true;
        break;

    case AV_TRIPLE:
        multiway = TripleWay;
    case AV_TRIPLE_SOFT:
        softAES  = true;
        break;

    case AV_QUAD:
        multiway = QuadWay;
    case AV_QUAD_SOFT:
        softAES  = true;
        break;

    case AV_PENTA:
        multiway = PentaWay;
    case AV_PENTA_SOFT:
        softAES  = true;
        break;

    default:
        break;
    }

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

    return new CpuThread(index, algorithm, av, multiway, cpuId, priority, softAES, false);
}


#ifndef XMRIG_NO_API
rapidjson::Value xmrig::CpuThread::toAPI(rapidjson::Document &doc) const
{
    rapidjson::Value obj(rapidjson::kObjectType);
    auto &allocator = doc.GetAllocator();

    obj.AddMember("type",          "cpu", allocator);
    obj.AddMember("algo",           rapidjson::StringRef(CommonConfig::algoName(algorithm())), allocator);
    obj.AddMember("av",             m_av, allocator);
    obj.AddMember("low_power_mode", multiway(), allocator);
    obj.AddMember("affine_to_cpu",  affinity(), allocator);
    obj.AddMember("priority",       priority(), allocator);
    obj.AddMember("soft_aes",       isSoftAES(), allocator);

    return obj;
}
#endif
