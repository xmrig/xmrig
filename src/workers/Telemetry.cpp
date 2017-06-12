/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
 *
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


#include <memory.h>
#include <math.h>
#include <chrono>

#include "Console.h"
#include "workers/Telemetry.h"


Telemetry::Telemetry(int threads) :
    m_threads(threads)
{
    m_counts     = new uint64_t*[threads];
    m_timestamps = new uint64_t*[threads];
    m_top        = new uint32_t[threads];

    for (int i = 0; i < threads; i++) {
        m_counts[i] = new uint64_t[kBucketSize];
        m_timestamps[i] = new uint64_t[kBucketSize];
        m_top[i] = 0;

        memset(m_counts[0], 0, sizeof(uint64_t) * kBucketSize);
        memset(m_timestamps[0], 0, sizeof(uint64_t) * kBucketSize);
    }
}


double Telemetry::calc(size_t threadId, size_t ms) const
{
    using namespace std::chrono;
    const uint64_t now = time_point_cast<milliseconds>(high_resolution_clock::now()).time_since_epoch().count();

    uint64_t earliestHashCount = 0;
    uint64_t earliestStamp     = 0;
    uint64_t lastestStamp      = 0;
    uint64_t lastestHashCnt    = 0;
    bool haveFullSet           = false;

    for (size_t i = 1; i < kBucketSize; i++) {
        const size_t idx = (m_top[threadId] - i) & kBucketMask;

        if (m_timestamps[threadId][idx] == 0) {
            break;
        }

        if (lastestStamp == 0) {
            lastestStamp = m_timestamps[threadId][idx];
            lastestHashCnt = m_counts[threadId][idx];
        }

        if (now - m_timestamps[threadId][idx] > ms) {
            haveFullSet = true;
            break;
        }

        earliestStamp = m_timestamps[threadId][idx];
        earliestHashCount = m_counts[threadId][idx];
    }

    if (!haveFullSet || earliestStamp == 0 || lastestStamp == 0) {
        return nan("");
    }

    if (lastestStamp - earliestStamp == 0) {
        return nan("");
    }

    double hashes, time;
    hashes = lastestHashCnt - earliestHashCount;
    time   = lastestStamp - earliestStamp;
    time  /= 1000.0;

    return hashes / time;
}


void Telemetry::add(size_t threadId, uint64_t count, uint64_t timestamp)
{
    const size_t top = m_top[threadId];
    m_counts[threadId][top]     = count;
    m_timestamps[threadId][top] = timestamp;

    m_top[threadId] = (top + 1) & kBucketMask;
}
