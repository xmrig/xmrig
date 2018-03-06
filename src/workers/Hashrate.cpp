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

#ifndef _WIN32
#if __cplusplus <= 199711L
#include <sys/time.h>
#else
#include <chrono>
#define USE_CHRONO
#endif
#else
#define isnormal(x) (_fpclass(x) == _FPCLASS_NN || _fpclass(x) == _FPCLASS_PN)
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <stdint.h> // portable: uint64_t   MSVC: __int64 
#include <WinSock2.h>

static int gettimeofday(struct timeval* tp, struct timezone* tzp)
{
	// Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
	// This magic number is the number of 100 nanosecond intervals since January 1, 1601 (UTC)
	// until 00:00:00 January 1, 1970
	static const uint64_t EPOCH = ((uint64_t) 116444736000000000ULL);

	SYSTEMTIME  system_time;
	FILETIME    file_time;
	uint64_t    time;

	GetSystemTime(&system_time);
	SystemTimeToFileTime(&system_time, &file_time);
	time = ((uint64_t)file_time.dwLowDateTime)      ;
	time += ((uint64_t)file_time.dwHighDateTime) << 32;

	tp->tv_sec  = (long)((time - EPOCH) / 10000000L);
	tp->tv_usec = (long)(system_time.wMilliseconds * 1000);
	return 0;
}
#endif

#include <math.h>
#include <limits.h>
#include <memory.h>
#include <stdio.h>

#include "log/Log.h"
#include "Options.h"
#include "workers/Hashrate.h"


inline std::string format(double value)
{
	char buff[8];
	if(isnormal(value))
	{
		snprintf(buff, sizeof(buff), "%03.1f", value);
		return buff;
	}

	return "n/a";
}


Hashrate::Hashrate(int threads) :
	m_highest(0.0),
	m_threads(threads)
{
	m_counts     = new uint64_t* [threads];
	m_timestamps = new uint64_t* [threads];
	m_top        = new uint32_t[threads];

	for(int i = 0; i < threads; i++)
	{
		m_counts[i] = new uint64_t[kBucketSize];
		m_timestamps[i] = new uint64_t[kBucketSize];
		m_top[i] = 0;

		memset(m_counts[0], 0, sizeof(uint64_t) * kBucketSize);
		memset(m_timestamps[0], 0, sizeof(uint64_t) * kBucketSize);
	}

	const int printTime = Options::i()->printTime();

	if(printTime > 0)
	{
		uv_timer_init(uv_default_loop(), &m_timer);
		m_timer.data = this;

		uv_timer_start(&m_timer, Hashrate::onReport, (printTime + 4) * 1000, printTime * 1000);
	}
}


double Hashrate::calc(size_t ms) const
{
	double result = 0.0;
	double data;

	for(int i = 0; i < m_threads; ++i)
	{
		data = calc(i, ms);
		if(isnormal(data))
		{
			result += data;
		}
	}

	return result;
}


double Hashrate::calc(size_t threadId, size_t ms) const
{
#ifdef USE_CHRONO
	using namespace std::chrono;
	const uint64_t now = time_point_cast<milliseconds>(high_resolution_clock::now()).time_since_epoch().count();
#else
	struct timeval tp;
	gettimeofday(&tp, NULL);
	const uint64_t now = tp.tv_sec * 1000 + tp.tv_usec / 1000;
#endif

	uint64_t earliestHashCount = 0;
	uint64_t earliestStamp     = 0;
	uint64_t lastestStamp      = 0;
	uint64_t lastestHashCnt    = 0;
	bool haveFullSet           = false;

	for(size_t i = 1; i < kBucketSize; i++)
	{
		const size_t idx = (m_top[threadId] - i) & kBucketMask;

		if(m_timestamps[threadId][idx] == 0)
		{
			break;
		}

		if(lastestStamp == 0)
		{
			lastestStamp = m_timestamps[threadId][idx];
			lastestHashCnt = m_counts[threadId][idx];
		}

		if(now - m_timestamps[threadId][idx] > ms)
		{
			haveFullSet = true;
			break;
		}

		earliestStamp = m_timestamps[threadId][idx];
		earliestHashCount = m_counts[threadId][idx];
	}

	if(!haveFullSet || earliestStamp == 0 || lastestStamp == 0)
	{
		return std::numeric_limits<double>::quiet_NaN() ;
	}

	if(lastestStamp - earliestStamp == 0)
	{
		return std::numeric_limits<double>::quiet_NaN() ;
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


void Hashrate::print()
{
	LOG_INFO("speed 2.5s/60s/15m " <<
	         format(calc(ShortInterval)) << " " <<
	         format(calc(MediumInterval)) << " " <<
	         format(calc(LargeInterval)) << " H/s max: " <<
	         format(m_highest) << " H/s");
}


void Hashrate::stop()
{
	uv_timer_stop(&m_timer);
}


void Hashrate::updateHighest()
{
	double highest = calc(ShortInterval);
	if(isnormal(highest) && highest > m_highest)
	{
		m_highest = highest;
	}
}


void Hashrate::onReport(uv_timer_t* handle)
{
	static_cast<Hashrate*>(handle->data)->print();
}
