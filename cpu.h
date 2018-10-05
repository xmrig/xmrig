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

#ifndef XMRIG_CPU_H
#define XMRIG_CPU_H

#include <stdbool.h>

struct cpu_info {
    int total_cores;
    int total_logical_cpus;
    int flags;
    int sockets;
    int l2_cache;
    int l3_cache;
    char brand[64];
    int assembly;
};

extern struct cpu_info cpu_info;


enum cpu_flags {
    CPU_FLAG_X86_64 = 1,
    CPU_FLAG_AES    = 2,
    CPU_FLAG_BMI2   = 4
};


void cpu_init();
int get_optimal_threads_count(int algo, bool double_hash, int max_cpu_usage);
int affine_to_cpu_mask(int id, unsigned long mask);

#endif /* XMRIG_CPU_H */
