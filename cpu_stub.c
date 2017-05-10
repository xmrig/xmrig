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
#include "cpu.h"


#define VENDOR_ID                  (0)
#define PROCESSOR_INFO             (1)
#define CACHE_TLB_DESCRIPTOR       (2)
#define EXTENDED_FEATURES          (7)
#define PROCESSOR_BRAND_STRING_1   (0x80000002)
#define PROCESSOR_BRAND_STRING_2   (0x80000003)
#define PROCESSOR_BRAND_STRING_3   (0x80000004)

#define EAX_Reg  (0)
#define EBX_Reg  (1)
#define ECX_Reg  (2)
#define EDX_Reg  (3)


static inline void cpuid(int level, int output[4]) {
    int a, b, c, d;
    __cpuid_count(level, 0, a, b, c, d);

    output[0] = a;
    output[1] = b;
    output[2] = c;
    output[3] = d;
}


static void cpu_brand_string(char* s) {
    int cpu_info[4] = { 0 };
    cpuid(VENDOR_ID, cpu_info);

    if (cpu_info[EAX_Reg] >= 4) {
        for (int i = 0; i < 4; i++) {
            cpuid(0x80000002 + i, cpu_info);
            memcpy(s, cpu_info, sizeof(cpu_info));
            s += 16;
        }
    }
}


static bool has_aes_ni()
{
    int cpu_info[4] = { 0 };
    cpuid(PROCESSOR_INFO, cpu_info);

    return cpu_info[ECX_Reg] & bit_AES;
}


static bool has_bmi2() {
    int cpu_info[4] = { 0 };
    cpuid(EXTENDED_FEATURES, cpu_info);

    return cpu_info[EBX_Reg] & bit_BMI2;
}


void cpu_init_common() {
    cpu_info.sockets = 1;
    cpu_brand_string(cpu_info.brand);

#   ifdef __x86_64__
    cpu_info.flags |= CPU_FLAG_X86_64;
#   endif

    if (has_aes_ni()) {
        cpu_info.flags |= CPU_FLAG_AES;
    }

    if (has_bmi2()) {
        cpu_info.flags |= CPU_FLAG_BMI2;
    }
}


int get_optimal_threads_count(int algo, bool double_hash, int max_cpu_usage) {
    int count = cpu_info.total_logical_cpus / 2;
    return count < 1 ? 1 : count;
}
