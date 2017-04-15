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

#ifndef __COMPAT_H__
#define __COMPAT_H__

#define unlikely(expr) (__builtin_expect(!!(expr), 0))
#define likely(expr)   (__builtin_expect(!!(expr), 1))

#ifdef WIN32

#include <windows.h>

#define sleep(secs) Sleep((secs) * 1000)

enum {
	PRIO_PROCESS		= 0,
};

static inline int setpriority(int which, int who, int prio)
{
	return -!SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_IDLE);
}

#endif /* WIN32 */

#endif /* __COMPAT_H__ */
