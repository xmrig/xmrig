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

#include "Cpu.h"
#include "Mem.h"
#include "Platform.h"
#include "workers/Handle.h"
#include "workers/Worker.h"

#ifndef _WIN32
#if __cplusplus <= 199711L
#include <sys/time.h>
#else
#include <chrono>
#define USE_CHRONO
#endif
#else
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

Worker::Worker(Handle* handle) :
	m_id(handle->threadId()),
	m_threads(handle->threads()),
	m_hashCount(0),
	m_timestamp(0),
	m_count(0),
	m_sequence(0)
{
	if(Cpu::threads() > 1 && handle->affinity() != -1L)
	{
		Cpu::setAffinity(m_id, handle->affinity());
	}

	Platform::setThreadPriority(handle->priority());
	m_ctx = Mem::create(m_id);
}


Worker::~Worker()
{
}


void Worker::storeStats()
{
#ifdef USE_CHRONO
	using namespace std::chrono;
	const uint64_t now = time_point_cast<milliseconds>(high_resolution_clock::now()).time_since_epoch().count();
#else
	struct timeval tp;
	gettimeofday(&tp, NULL);
	const uint64_t now = tp.tv_sec * 1000 + tp.tv_usec / 1000;
#endif

	m_hashCount = m_count;
	m_timestamp = now;
}
