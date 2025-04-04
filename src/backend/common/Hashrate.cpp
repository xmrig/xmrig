/* XMRig
 * Copyright (c) 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright (c) 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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
#include <memory.h>
#include <cstdio>


#include "backend/common/Hashrate.h"
#include "3rdparty/rapidjson/document.h"
#include "base/io/json/Json.h"
#include "base/tools/Chrono.h"
#include "base/tools/Handle.h"


inline static const char *format(std::pair<bool, double> h, char *buf, size_t size)
{
    if (h.first) {
        snprintf(buf, size, (h.second < 100.0) ? "%04.2f" : "%03.1f", h.second);
        return buf;
    }

    return "n/a";
}


xmrig::Hashrate::Hashrate(size_t threads) :
    m_threads(threads + 1)
{
    m_counts     = new uint64_t*[m_threads];
    m_timestamps = new uint64_t*[m_threads];
    m_top        = new uint32_t[m_threads];

    for (size_t i = 0; i < m_threads; i++) {
        m_counts[i]     = new uint64_t[kBucketSize]();
        m_timestamps[i] = new uint64_t[kBucketSize]();
        m_top[i]        = 0;
    }

    m_earliestTimestamp = std::numeric_limits<uint64_t>::max();
    m_totalCount = 0;
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


double xmrig::Hashrate::average() const
{
    const uint64_t ts = Chrono::steadyMSecs();
    return (ts > m_earliestTimestamp) ? (m_totalCount * 1e3 / (ts - m_earliestTimestamp)) : 0.0;
}


const char *xmrig::Hashrate::format(std::pair<bool, double> h, char *buf, size_t size)
{
    return ::format(h, buf, size);
}


rapidjson::Value xmrig::Hashrate::normalize(std::pair<bool, double> d)
{
    using namespace rapidjson;
    return d.first ? Value(floor(d.second * 100.0) / 100.0) : Value(kNullType);
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


std::pair<bool, double> xmrig::Hashrate::hashrate(size_t index, size_t ms) const
{
    assert(index < m_threads);
    if (index >= m_threads) {
        return { false, 0.0 };
    }

    uint64_t earliestHashCount = 0;
    uint64_t earliestStamp     = 0;
    bool haveFullSet           = false;

    const uint64_t timeStampLimit = xmrig::Chrono::steadyMSecs() - ms;
    uint64_t* timestamps          = m_timestamps[index];
    uint64_t* counts              = m_counts[index];

    const size_t idx_start  = (m_top[index] - 1) & kBucketMask;
    size_t idx              = idx_start;

    uint64_t lastestStamp   = timestamps[idx];
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
        return { false, 0.0 };
    }

    if (lastestHashCnt == earliestHashCount) {
        return { true, 0.0 };
    }

    if (lastestStamp == earliestStamp) {
        return { false, 0.0 };
    }

    const auto hashes = static_cast<double>(lastestHashCnt - earliestHashCount);
    const auto time   = static_cast<double>(lastestStamp - earliestStamp);

    const auto hr = hashes * 1000.0 / time;

    if (!std::isnormal(hr)) {
        return { false, 0.0 };
    }

    return { true, hr };
}


void xmrig::Hashrate::addData(size_t index, uint64_t count, uint64_t timestamp)
{
    const size_t top         = m_top[index];
    m_counts[index][top]     = count;
    m_timestamps[index][top] = timestamp;

    m_top[index] = (top + 1) & kBucketMask;

    if (index == 0) {
        if (m_earliestTimestamp == std::numeric_limits<uint64_t>::max()) {
            m_earliestTimestamp = timestamp;
        }
        m_totalCount = count;
    }
}
