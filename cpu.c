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

#include <cpuid.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#ifndef BUILD_TEST
#   include <libcpuid.h>
#endif

#include "cpu.h"
#include "utils/applog.h"


#ifndef BUILD_TEST
void cpu_init_common() {
    struct cpu_raw_data_t raw = { 0 };
    struct cpu_id_t data = { 0 };

    cpuid_get_raw_data(&raw);
    cpu_identify(&raw, &data);

    strncpy(cpu_info.brand, data.brand_str, sizeof(cpu_info.brand) - 1);

    cpu_info.total_logical_cpus = data.total_logical_cpus;
    cpu_info.sockets            = data.total_logical_cpus / data.num_logical_cpus;
    cpu_info.total_cores        = data.num_cores * cpu_info.sockets;
    cpu_info.l2_cache           = data.l2_cache > 0 ? data.l2_cache * cpu_info.total_cores * cpu_info.sockets : 0;
    cpu_info.l3_cache           = data.l3_cache > 0 ? data.l3_cache * cpu_info.sockets : 0;

#   ifdef __x86_64__
    cpu_info.flags |= CPU_FLAG_X86_64;
#   endif

    if (data.flags[CPU_FEATURE_AES]) {
        cpu_info.flags |= CPU_FLAG_AES;
    }

    if (data.flags[CPU_FEATURE_BMI2]) {
        cpu_info.flags |= CPU_FLAG_BMI2;
    }
}
#endif


int get_optimal_threads_count(int algo, bool double_hash, int max_cpu_usage) {
    if (cpu_info.total_logical_cpus == 1) {
        return 1;
    }

    int cache = cpu_info.l3_cache ? cpu_info.l3_cache : cpu_info.l2_cache;
    int count = 0;
    const int size = (algo ? 1024 : 2048) * (double_hash ? 2 : 1);

    if (cache) {
        count = cache / size;
    }
    else {
        count = cpu_info.total_logical_cpus / 2;
    }

    if (count > cpu_info.total_logical_cpus) {
        count = cpu_info.total_logical_cpus;
    }

    if (((float) count / cpu_info.total_logical_cpus * 100) > max_cpu_usage) {
        count = ceil((float) cpu_info.total_logical_cpus * (max_cpu_usage / 100.0));
    }

    return count < 1 ? 1 : count;
}
