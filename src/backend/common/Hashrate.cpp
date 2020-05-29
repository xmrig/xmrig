/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include <cassert>
#include <cmath>
#include <memory.h>
#include <cstdio>


#include "backend/common/Hashrate.h"
#include "3rdparty/rapidjson/document.h"
#include "base/io/json/Json.h"
#include "base/tools/Chrono.h"
#include "base/tools/Handle.h"


inline static const char *format(double h, char *buf, size_t size)
{
    if (std::isnormal(h)) {
        snprintf(buf, size, (h < 100.0) ? "%04.2f" : "%03.1f", h);
        return buf;
    }

    return "n/a";
}


xmrig::Hashrate::Hashrate(size_t threads) :
    m_threads(threads)
{
    m_counts     = new uint64_t*[threads];
    m_timestamps = new uint64_t*[threads];
    m_top        = new uint32_t[threads];

    for (size_t i = 0; i < threads; i++) {
        m_counts[i]     = new uint64_t[kBucketSize]();
        m_timestamps[i] = new uint64_t[kBucketSize]();
        m_top[i]        = 0;
    }
}


xmrig::Hashrate::~Hashrate()
{
    for (size_t i = 0; i < m_threads; i++) {
        delete [] m_counts[i];
        delete [] m_timestamps[i];
    }

    delete [] m_counts;
    delete [] m_timestamps;
    delete [] m_top;
}


double xmrig::Hashrate::calc(size_t ms) const
{
    double result = 0.0;
    double data;

    for (size_t i = 0; i < m_threads; ++i) {
        data = calc(i, ms);
        if (std::isnormal(data)) {
            result += data;
        }
    }

    return result;
}


double xmrig::Hashrate::calc(size_t threadId, size_t ms) const
{
    assert(threadId < m_threads);
    if (threadId >= m_threads) {
        return nan("");
    }

    uint64_t earliestHashCount = 0;
    uint64_t earliestStamp     = 0;
    bool haveFullSet           = false;

    const uint64_t timeStampLimit = xmrig::Chrono::highResolutionMSecs() - ms;
    uint64_t* timestamps = m_timestamps[threadId];
    uint64_t* counts = m_counts[threadId];

    const size_t idx_start = (m_top[threadId] - 1) & kBucketMask;
    size_t idx = idx_start;

    uint64_t lastestStamp = timestamps[idx];
    uint64_t lastestHashCnt = counts[idx];

    do {
        if (timestamps[idx] < timeStampLimit) {
            haveFullSet = (timestamps[idx] != 0);
            if (idx != idx_start) {
                idx = (idx + 1) & kBucketMask;
                earliestStamp = timestamps[idx];
                earliestHashCount = counts[idx];
            }
            break;
        }
        idx = (idx - 1) & kBucketMask;
    } while (idx != idx_start);

    if (!haveFullSet || earliestStamp == 0 || lastestStamp == 0) {
        return nan("");
    }

    if (lastestStamp - earliestStamp == 0) {
        return nan("");
    }

    const auto hashes = static_cast<double>(lastestHashCnt - earliestHashCount);
    const auto time   = static_cast<double>(lastestStamp - earliestStamp) / 1000.0;

    return hashes / time;
}


void xmrig::Hashrate::add(size_t threadId, uint64_t count, uint64_t timestamp)
{
    const size_t top = m_top[threadId];
    m_counts[threadId][top]     = count;
    m_timestamps[threadId][top] = timestamp;

    m_top[threadId] = (top + 1) & kBucketMask;
}


const char *xmrig::Hashrate::format(double h, char *buf, size_t size)
{
    return ::format(h, buf, size);
}


rapidjson::Value xmrig::Hashrate::normalize(double d)
{
    return Json::normalize(d, false);
}


#ifdef XMRIG_FEATURE_API
rapidjson::Value xmrig::Hashrate::toJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    Value out(kArrayType);
    out.PushBack(normalize(calc(ShortInterval)),  allocator);
    out.PushBack(normalize(calc(MediumInterval)), allocator);
    out.PushBack(normalize(calc(LargeInterval)),  allocator);

    return out;
}


rapidjson::Value xmrig::Hashrate::toJSON(size_t threadId, rapidjson::Document &doc) const
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    Value out(kArrayType);
    out.PushBack(normalize(calc(threadId, ShortInterval)),  allocator);
    out.PushBack(normalize(calc(threadId, MediumInterval)), allocator);
    out.PushBack(normalize(calc(threadId, LargeInterval)),  allocator);

    return out;
}
#endif
