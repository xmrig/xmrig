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

#ifndef __XMRIG_H__
#define __XMRIG_H__

#include <stdbool.h>
#include <inttypes.h>
#include <jansson.h>
#include <curl/curl.h>
#include <pthread.h>

#define unlikely(expr) (__builtin_expect(!!(expr), 0))
#define likely(expr)   (__builtin_expect(!!(expr), 1))


struct thr_info {
    int id;
    pthread_t pth;
    struct thread_q *q;
};


struct work_restart {
    volatile unsigned long restart;
    char padding[128 - sizeof(unsigned long)];
};


struct work;


extern struct thr_info *thr_info;
extern struct work_restart *work_restart;
extern void os_specific_init();

#endif /* __XMRIG_H__ */
