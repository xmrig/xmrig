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
#include "base/io/log/Log.h"


inline static const char *format(double h, char *buf, size_t size, double errorDown, double errorUp)
{
    if (std::isnormal(h)) {
        if (std::max(errorDown, errorUp) >= (h < 100.0 ? 0.01 : 0.1)) {
            snprintf(buf, size, (h < 100.0) ? "%5.2f" BLACK_BOLD("(-%.2f/+%.2f}") : "%5.1f" BLACK_BOLD("(-%.1f/+%.1f)"), h, errorDown, errorUp);
        }
        else {
            snprintf(buf, size, (h < 100.0) ? "%5.2f" : "%5.1f", h);
        }
        return buf;
    }

    return "n/a";
}


xmrig::Hashrate::Hashrate(size_t threads) :
    m_threads(threads)
{
    m_counts     = new uint64_t*[threads];
    m_timestamps = new uint64_t*[threads];
    m_head       = new uint32_t[threads];
    m_tail       = new uint32_t*[threads];

    const uint64_t now = xmrig::Chrono::highResolutionMSecs();
    for (size_t i = 0; i < threads; i++) {
        m_counts[i]     = new uint64_t[kBucketSize]();
        m_timestamps[i] = new uint64_t[kBucketSize]();
        m_head[i]       = 0;
        m_tail[i]       = new uint32_t[3]();
        m_timestamps[i][0] = now;
    }
}


xmrig::Hashrate::~Hashrate()
{
    for (size_t i = 0; i < m_threads; i++) {
        delete [] m_counts[i];
        delete [] m_timestamps[i];
        delete [] m_tail[i];
    }

    delete [] m_counts;
    delete [] m_timestamps;
    delete [] m_head;
    delete [] m_tail;
}



xmrig::Hashrate::Value xmrig::Hashrate::calc(size_t threadId, Intervals ms) const
{
    TimeRange time;
    uint64_t count;
    if (!findRecords(threadId, ms, time, count)) {
        return { nan(""), nan(""), nan("") };
    }
    return { count * 1000.0 / (time.second - time.first), 0.0, 0.0 };
}

xmrig::Hashrate::Value xmrig::Hashrate::calc(Intervals ms) const
{
    const uint64_t now = xmrig::Chrono::highResolutionMSecs();
    uint64_t time_earliest[2] = { now, 0 };
    uint64_t time_latest[2]   = { 0, now };
    uint64_t total_count      = 0;
    double result             = 0.0;
    for (size_t threadId = 0; threadId < m_threads; ++threadId) {
        TimeRange time;
        uint64_t count;
        if (!findRecords(threadId, ms, time, count)) {
            continue;
        }
        for (size_t j = 0; j < 2; ++j) {
            if ((time.first < time_earliest[j]) ^ j) {
                time_earliest[j] = time.first;
            }
            if ((time.second > time_latest[j]) ^ j) {
                time_latest[j] = time.second;
            }
        }
        total_count += count;
        result += count * 1000.0 / (time.second - time.first);
    }
    const double lower = total_count  * 1000.0 / (time_latest[0] - time_earliest[0]);
    const double upper = total_count  * 1000.0 / (time_latest[1] - time_earliest[1]);
    return { result, result - lower, upper - result };
}


bool xmrig::Hashrate::findRecords(size_t threadId, Intervals ms, TimeRange &time, uint64_t &count) const
{
    assert(threadId < m_threads);
    if (threadId >= m_threads) {
        return false;
    }

    const uint64_t timeStampLimit = xmrig::Chrono::highResolutionMSecs() - ms;
    const uint32_t head = m_head[threadId];
    // time[tale] < timeStampLimit <= time[later_tail] <= time[head]
    uint32_t &tail = m_tail[threadId][ms == ShortInterval ? 0 : (ms == MediumInterval ? 1 : 2)];
    if (m_timestamps[threadId][tail] >= timeStampLimit) {
        return false;
    }
    while (tail != head) {
        const uint32_t later_tail = (tail + 1) & kBucketMask;
        if (m_timestamps[threadId][later_tail] >= timeStampLimit) {
            time = { m_timestamps[threadId][later_tail], m_timestamps[threadId][head] };
            count = m_counts[threadId][head] - m_counts[threadId][later_tail];
            return true;
        }
        tail = later_tail;
    }
    return false;
}


void xmrig::Hashrate::add(size_t threadId, uint64_t count, uint64_t timestamp)
{
    const uint32_t head = (m_head[threadId] + 1) & kBucketMask;
    m_counts[threadId][head]     = count;
    m_timestamps[threadId][head] = timestamp;

    m_head[threadId] = head;
}


const char *xmrig::Hashrate::format(Value h, char *buf, size_t size)
{
    return ::format(h.estimate, buf, size, h.errorDown, h.errorUp);
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
