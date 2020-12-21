/* XMRig
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


#include "crypto/rx/Profiler.h"
#include "base/io/log/Log.h"
#include "base/io/log/Tags.h"


#include <cstring>
#include <sstream>
#include <thread>
#include <chrono>
#include <algorithm>


#ifdef XMRIG_FEATURE_PROFILING


ProfileScopeData* ProfileScopeData::s_data[MAX_DATA_COUNT] = {};
volatile long ProfileScopeData::s_dataCount = 0;
double ProfileScopeData::s_tscSpeed = 0.0;


#ifndef NOINLINE
#ifdef __GNUC__
#define NOINLINE __attribute__ ((noinline))
#elif _MSC_VER
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE
#endif
#endif


static std::string get_thread_id()
{
    std::stringstream ss;
    ss << std::this_thread::get_id();

    std::string s = ss.str();
    if (s.length() > ProfileScopeData::MAX_THREAD_ID_LENGTH) {
        s.resize(ProfileScopeData::MAX_THREAD_ID_LENGTH);
    }

    return s;
}


NOINLINE void ProfileScopeData::Register(ProfileScopeData* data)
{
#ifdef _MSC_VER
    const long id = _InterlockedIncrement(&s_dataCount) - 1;
#else
    const long id = __sync_fetch_and_add(&s_dataCount, 1);
#endif

    if (static_cast<unsigned long>(id) < MAX_DATA_COUNT) {
        s_data[id] = data;

        const std::string s = get_thread_id();
        memcpy(data->m_threadId, s.c_str(), s.length() + 1);
    }
}


NOINLINE void ProfileScopeData::Init()
{
    using namespace std::chrono;

    const uint64_t t1 = static_cast<uint64_t>(time_point_cast<nanoseconds>(high_resolution_clock::now()).time_since_epoch().count());
    const uint64_t count1 = ReadTSC();

    for (;;)
    {
        const uint64_t t2 = static_cast<uint64_t>(time_point_cast<nanoseconds>(high_resolution_clock::now()).time_since_epoch().count());
        const uint64_t count2 = ReadTSC();

        if (t2 - t1 > 1000000000) {
            s_tscSpeed = (count2 - count1) * 1e9 / (t2 - t1);
            LOG_INFO("%s TSC speed = %.3f GHz", xmrig::Tags::profiler(), s_tscSpeed / 1e9);
            return;
        }
    }
}


#endif /* XMRIG_FEATURE_PROFILING */
