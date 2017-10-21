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


#include "interfaces/ILogBackend.h"
#include "log/Log.h"


Log *Log::m_self = nullptr;


void Log::message(Log::Level level, const char* fmt, ...)
{
    va_list args;
    va_list copy;
    va_start(args, fmt);

    for (ILogBackend *backend : m_backends) {
        va_copy(copy, args);
        backend->message(level, fmt, copy);
        va_end(copy);
    }
}


void Log::text(const char* fmt, ...)
{
    va_list args;
    va_list copy;
    va_start(args, fmt);

    for (ILogBackend *backend : m_backends) {
        va_copy(copy, args);
        backend->text(fmt, copy);
        va_end(copy);
    }

    va_end(args);
}


Log::~Log()
{
    for (auto backend : m_backends) {
        delete backend;
    }
}
