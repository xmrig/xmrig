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
#include <chrono>
#include <math.h>
#include <memory.h>
#include <stdio.h>


#include "common/log/Log.h"
#include "core/Config.h"
#include "core/Controller.h"
#include "workers/Hashrate.h"

namespace {

uint64_t pow10(unsigned n)
{
    if (n == 0) return 1;

    uint64_t pow = 10;
    for (unsigned i = 0; i < n - 1; i++) pow *= 10;

    return pow;
}

} // namespace

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
    m_controller(controller),
    m_start_time(time(NULL))
{
    m_counts     = new uint64_t*[threads];
    m_timestamps = new uint64_t*[threads];
    m_top        = new uint32_t[threads];
    m_totals     = new uint64_t[threads];

    for (size_t i = 0; i < threads; i++) {
        m_counts[i]     = new uint64_t[kBucketSize]();
        m_timestamps[i] = new uint64_t[kBucketSize]();
        m_top[i]        = 0;
        m_totals[i]     = 0;
    }

    const int printTime = controller->config()->printTime();

    if (printTime > 0) {
        uv_timer_init(uv_default_loop(), &m_timer);
        m_timer.data = this;

       uv_timer_start(&m_timer, Hashrate::onReport, (printTime + 4) * 1000, printTime * 1000);
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

uint64_t Hashrate::count() const
{
    uint64_t total = 0;
    for (unsigned i = 0; i < m_threads; i++) {
        total += count(i);
    }

    return total;
}

uint64_t Hashrate::count(size_t threadId) const
{
    assert(threadId < m_threads);
    if (threadId >= m_threads) return 0;

    return m_totals[threadId];
}

uint64_t Hashrate::elapsed() const
{
    return time(NULL) - m_start_time;
}


void Hashrate::add(size_t threadId, uint64_t count, uint64_t timestamp)
{
    m_totals[threadId] = count;

    const size_t top = m_top[threadId];
    m_counts[threadId][top]     = count;
    m_timestamps[threadId][top] = timestamp;

    m_top[threadId] = (top + 1) & kBucketMask;
}


void Hashrate::print() const
{
    char num1[8]  = { 0 };
    char num2[8]  = { 0 };
    char num3[8]  = { 0 };
    char num4[8]  = { 0 };
    char num5[128] = { 0 };
    char num6[128] = { 0 };
    char num7[128] = { 0 };

    bool color = m_controller->config()->isColors();
    uint64_t e = elapsed();
    uint64_t c = count();

    LOG_INFO(color ? WHITE_BOLD("speed") " 10s/60s/15m " CYAN_BOLD("%s") CYAN(" %s %s ") CYAN_BOLD("H/s") " max " CYAN_BOLD("%s H/s") : "speed 10s/60s/15m %s %s %s H/s max %s H/s",
             format(calc(ShortInterval),  num1, sizeof(num1)),
             format(calc(MediumInterval), num2, sizeof(num2)),
             format(calc(LargeInterval),  num3, sizeof(num3)),
             format(m_highest,            num4, sizeof(num4))
     );
    LOG_INFO(color ? WHITE_BOLD("totals") " time %s hashes %s rate " CYAN("%s ") CYAN_BOLD("H/s") : "totals: time %s hashes %s rate %s H/s",
         timeFormat(e,       num5, sizeof(num5), color),
         siFormat(c,         num6, sizeof(num6), color),
         format(c * 1.0 / e, num7, sizeof(num7))
    );
}


void Hashrate::stop()
{
    uv_timer_stop(&m_timer);
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

const char *Hashrate::siFormat(uint64_t h, char *buf, size_t size, bool color)
{

    char c[2];
    c[1] = '\0';
    uint64_t d = 0;

    if (h > pow10(9)) {
        c[0] = 'G';
        d = (h % pow10(9)) / pow10(6);
        h /= pow10(9);
    }
    else if (h > pow10(6)) {
        c[0] = 'M';
        d = (h % pow10(6)) / pow10(3);
        h /= pow10(6);
    }
    else if (h > pow10(3)) {
        c[0] = 'k';
        d = h % pow10(3);
        h /= pow10(3);
    }
    else {
        snprintf(buf, size, (color) ? CYAN("%lu") : "%lu", h);
        return buf;
    }

    snprintf(buf, size, (color) ? CYAN("%lu.%03lu") CYAN_BOLD("%s") : "%lu.%03lu%s", h, d, c);
    return buf;
}

const char *Hashrate::timeFormat(uint64_t s, char *buf, size_t size, bool color)
{
    uint64_t d = 0;
    uint64_t h = 0;
    uint64_t m = 0;

    if (s > 3600 * 24) {
        d = s / (3600*24);
        s = s % (3600*24);
    }
    if (s > 3600) {
        h = s / (3600);
        s = s % (3600);
    }
    if (s > 60) {
        m = s / (60);
        s = s % (60);
    }

    if (!color) {
        snprintf(buf, size, "%02lud::%02luh::%02lum::%02lus", d, h, m, s);
    }
    else {
        snprintf(buf, size, CYAN("%02lu") CYAN_BOLD("d") CYAN("%02lu") CYAN_BOLD("h") CYAN("%02lu") CYAN_BOLD("m") CYAN("%02lu") CYAN_BOLD("s"), d, h, m, s);
    }
    return buf;
}


void Hashrate::onReport(uv_timer_t *handle)
{
    static_cast<Hashrate*>(handle->data)->print();
}
