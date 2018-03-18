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

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <iostream>

#include "log/ConsoleLog.h"
#include "log/Log.h"

ConsoleLog::ConsoleLog(bool colors) :
	m_colors(colors)
{
}

void ConsoleLog::message(Level level, const std::string & text)
{
	if(!isWritable())
	{
		return;
	}

	//
	//
	time_t now = time(nullptr);
	tm stime;

#ifdef _WIN32
	localtime_s(&stime, &now);
#else
	localtime_r(&now, &stime);
#endif

	char buf[25];
	int size = snprintf(buf, sizeof(buf), "[%d-%02d-%02d %02d:%02d:%02d] ",
	                    stime.tm_year + 1900,
	                    stime.tm_mon + 1,
	                    stime.tm_mday,
	                    stime.tm_hour,
	                    stime.tm_min,
	                    stime.tm_sec);

	//
	//
	std::string colorIni, colorEnd;
	if(m_colors)
	{
		colorEnd = Log::CL_N();
		switch(level)
		{
		case ILogBackend::ERR:
			colorIni = Log::CL_RED();
			break;

		case ILogBackend::WARNING:
			colorIni = Log::CL_YELLOW();
			break;

		case ILogBackend::NOTICE:
			colorIni = Log::CL_WHITE();
			break;

		case ILogBackend::DEBUG:
			colorIni = Log::CL_GRAY();
			break;

		default:
			break;
		}
	}

	print(std::string(buf, size) + colorIni + text + colorEnd);
}

void ConsoleLog::text(const std::string & txt)
{
	if(!isWritable())
	{
		return;
	}

	print(txt);
}

bool ConsoleLog::isWritable() const
{
	return std::cout.good();
}

void ConsoleLog::print(const std::string & txt)
{
	std::cout << txt << std::endl;
	std::cout.flush();
}
