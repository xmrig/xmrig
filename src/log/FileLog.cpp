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

#include <fstream>
#include <iostream>

#include "log/Log.h"
#include "log/FileLog.h"

FileLog::FileLog(const std::string & fileName)
	: m_file_name(fileName)
{
}

void FileLog::message(Level level, const std::string & txt)
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
	write(std::string(buf, size) + txt);
}

void FileLog::text(const std::string & txt)
{
	if(!isWritable())
	{
		return;
	}

	write(txt);
}

bool FileLog::isWritable() const
{
	return (m_file_name != "") && std::ofstream(m_file_name, std::ios_base::app).good();
}

void FileLog::write(const std::string & txt)
{
	std::ofstream outfile;

	outfile.open(m_file_name, std::ios_base::app);
	outfile << txt << std::endl;
}
