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

#include <unistd.h>
#include <sched.h>
#include <pthread.h>

#include "cpu.h"


struct cpu_info cpu_info = { 0 };
void cpu_init_common();


void cpu_init() {
    cpu_info.count = sysconf(_SC_NPROCESSORS_CONF);

    cpu_init_common();
}


int get_optimal_threads_count() {
    int count = cpu_info.count / 2;
    return count < 1 ? 1 : count;
}


int affine_to_cpu_mask(int id, unsigned long mask)
{
    cpu_set_t set;
    CPU_ZERO(&set);

    for (unsigned i = 0; i < cpu_info.count; i++) {
        if (mask & (1UL << i)) {
            CPU_SET(i, &set);
        }
    }

    if (id == -1) {
        sched_setaffinity(0, sizeof(&set), &set);
    } else {
        pthread_setaffinity_np(pthread_self(), sizeof(&set), &set);
    }
}
