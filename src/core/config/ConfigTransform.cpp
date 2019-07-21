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


#include "base/kernel/interfaces/IConfig.h"
#include "core/config/ConfigTransform.h"
#include "crypto/cn/CnHash.h"


namespace xmrig
{


static const char *kAffinity    = "affinity";
static const char *kAsterisk    = "*";
static const char *kCpu         = "cpu";
static const char *kIntensity   = "intensity";


static inline uint64_t intensity(uint64_t av)
{
    switch (av) {
    case CnHash::AV_SINGLE:
    case CnHash::AV_SINGLE_SOFT:
        return 1;

    case CnHash::AV_DOUBLE_SOFT:
    case CnHash::AV_DOUBLE:
        return 2;

    case CnHash::AV_TRIPLE_SOFT:
    case CnHash::AV_TRIPLE:
        return 3;

    case CnHash::AV_QUAD_SOFT:
    case CnHash::AV_QUAD:
        return 4;

    case CnHash::AV_PENTA_SOFT:
    case CnHash::AV_PENTA:
        return 5;

    default:
        break;
    }

    return 1;
}


static inline bool isHwAes(uint64_t av)
{
    return av == CnHash::AV_SINGLE || av == CnHash::AV_DOUBLE || (av > CnHash::AV_DOUBLE_SOFT && av < CnHash::AV_TRIPLE_SOFT);
}


static inline int64_t affinity(uint64_t index, int64_t affinity)
{
    if (affinity == -1L) {
        return -1L;
    }

    size_t idx = 0;

    for (size_t i = 0; i < 64; i++) {
        if (!(static_cast<uint64_t>(affinity) & (1ULL << i))) {
            continue;
        }

        if (idx == index) {
            return static_cast<int64_t>(i);
        }

        idx++;
    }

    return -1L;
}


}


xmrig::ConfigTransform::ConfigTransform() : BaseTransform()
{
}


void xmrig::ConfigTransform::finalize(rapidjson::Document &doc)
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    BaseTransform::finalize(doc);

    if (m_threads) {
        if (!doc.HasMember(kCpu)) {
            doc.AddMember(StringRef(kCpu), Value(kObjectType), allocator);
        }

        Value threads(kArrayType);

        if (m_intensity > 1) {
            for (uint64_t i = 0; i < m_threads; ++i) {
                Value thread(kObjectType);
                thread.AddMember(StringRef(kIntensity), m_intensity, allocator);
                thread.AddMember(StringRef(kAffinity), affinity(i, m_affinity), allocator);

                threads.PushBack(thread, doc.GetAllocator());
            }
        }
        else {
            for (uint64_t i = 0; i < m_threads; ++i) {
                threads.PushBack(affinity(i, m_affinity), doc.GetAllocator());
            }
        }

        doc[kCpu].AddMember(StringRef(kAsterisk), threads, doc.GetAllocator());
    }
}


void xmrig::ConfigTransform::transform(rapidjson::Document &doc, int key, const char *arg)
{
    BaseTransform::transform(doc, key, arg);

    switch (key) {
    case IConfig::AVKey:          /* --av */
    case IConfig::CPUPriorityKey: /* --cpu-priority */
    case IConfig::ThreadsKey:     /* --threads */
        return transformUint64(doc, key, static_cast<uint64_t>(strtol(arg, nullptr, 10)));

    case IConfig::HugePagesKey: /* --no-huge-pages */
        return transformBoolean(doc, key, false);

    case IConfig::CPUAffinityKey: /* --cpu-affinity */
        {
            const char *p  = strstr(arg, "0x");
            return transformUint64(doc, key, p ? strtoull(p, nullptr, 16) : strtoull(arg, nullptr, 10));
        }

#   ifndef XMRIG_NO_ASM
    case IConfig::AssemblyKey: /* --asm */
        return set(doc, kCpu, "asm", arg);
#   endif

    default:
        break;
    }
}


void xmrig::ConfigTransform::transformBoolean(rapidjson::Document &doc, int key, bool enable)
{
    switch (key) {
    case IConfig::HugePagesKey: /* --no-huge-pages */
        return set(doc, kCpu, "huge-pages", enable);

    default:
        break;
    }
}


void xmrig::ConfigTransform::transformUint64(rapidjson::Document &doc, int key, uint64_t arg)
{
    using namespace rapidjson;

    switch (key) {
    case IConfig::CPUAffinityKey: /* --cpu-affinity */
        m_affinity = static_cast<int64_t>(arg);
        break;

    case IConfig::ThreadsKey: /* --threads */
        m_threads = arg;
        break;

    case IConfig::AVKey: /* --av */
        m_intensity = intensity(arg);
        set(doc, kCpu, "hw-aes", isHwAes(arg));
        break;

    case IConfig::CPUPriorityKey: /* --cpu-priority */
        return set(doc, kCpu, "priority", arg);

    default:
        break;
    }
}

