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
#include <chrono>
#include <math.h>
#include <memory.h>
#include <stdio.h>


#include "base/io/log/Log.h"
#include "base/tools/Handle.h"
#include "core/config/Config.h"
#include "core/Controller.h"
#include "workers/Hashrate.h"


inline static const char *format(double h, char *buf, size_t size)
{
    if (isnormal(h)) {
        snprintf(buf, size, "%03.1f", h);
        return buf;
    }

    return "n/a";
}


Hashrate::Hashrate(size_t threads, xmrig::Controller *controller) :
    m_highest(0.0),
    m_threads(threads),
    m_timer(nullptr)
{
    m_counts     = new uint64_t*[threads];
    m_timestamps = new uint64_t*[threads];
    m_top        = new uint32_t[threads];

    for (size_t i = 0; i < threads; i++) {
        m_counts[i]     = new uint64_t[kBucketSize]();
        m_timestamps[i] = new uint64_t[kBucketSize]();
        m_top[i]        = 0;
    }

    const int printTime = controller->config()->printTime();

    if (printTime > 0) {
        m_timer = new uv_timer_t;
        uv_timer_init(uv_default_loop(), m_timer);
        m_timer->data = this;

        uv_timer_start(m_timer, Hashrate::onReport, (printTime + 4) * 1000, printTime * 1000);
    }
}


double Hashrate::calc(size_t ms) const
{
    double result = 0.0;
    double data;

    for (size_t i = 0; i < m_threads; ++i) {
        data = calc(i, ms);
        if (isnormal(data)) {
            result += data;
        }
    }

    return result;
}


double Hashrate::calc(size_t threadId, size_t ms) const
{
    assert(threadId < m_threads);
    if (threadId >= m_threads) {
        return nan("");
    }

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
    hashes = (double) lastestHashCnt - earliestHashCount;
    time   = (double) lastestStamp - earliestStamp;
    time  /= 1000.0;

    return hashes / time;
}


void Hashrate::add(size_t threadId, uint64_t count, uint64_t timestamp)
{
    const size_t top = m_top[threadId];
    m_counts[threadId][top]     = count;
    m_timestamps[threadId][top] = timestamp;

    m_top[threadId] = (top + 1) & kBucketMask;
}


void Hashrate::print() const
{
    char num1[8] = { 0 };
    char num2[8] = { 0 };
    char num3[8] = { 0 };
    char num4[8] = { 0 };

    LOG_INFO(WHITE_BOLD("speed") " 10s/60s/15m " CYAN_BOLD("%s") CYAN(" %s %s ") CYAN_BOLD("H/s") " max " CYAN_BOLD("%s H/s"),
             format(calc(ShortInterval),  num1, sizeof(num1)),
             format(calc(MediumInterval), num2, sizeof(num2)),
             format(calc(LargeInterval),  num3, sizeof(num3)),
             format(m_highest,            num4, sizeof(num4))
             );
}


void Hashrate::stop()
{
    xmrig::Handle::close(m_timer);
    m_timer = nullptr;
}


void Hashrate::updateHighest()
{
   double highest = calc(ShortInterval);
   if (isnormal(highest) && highest > m_highest) {
       m_highest = highest;
   }
}


const char *Hashrate::format(double h, char *buf, size_t size)
{
    return ::format(h, buf, size);
}


void Hashrate::onReport(uv_timer_t *handle)
{
    static_cast<Hashrate*>(handle->data)->print();
}
