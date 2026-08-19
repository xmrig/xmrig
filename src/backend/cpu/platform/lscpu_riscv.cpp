/* XMRig
 * Copyright (c) 2025      Slayingripper <https://github.com/Slayingripper>
 * Copyright (c) 2018-2025 SChernykh     <https://github.com/SChernykh>
 * Copyright (c) 2016-2025 XMRig         <support@xmrig.com>
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

#include "backend/cpu/platform/riscv_vlen.h"
#include "base/tools/String.h"
#include "3rdparty/fmt/core.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace xmrig {

struct riscv_cpu_desc
{
    String model;
    String isa;
    String uarch;
    bool has_vector = false;
    bool has_aes = false;
    
    inline bool isReady() const { return !isa.isNull(); }
};

static bool lookup_riscv(char *line, const char *pattern, String &value)
{
    char *p = strstr(line, pattern);
    if (!p) {
        return false;
    }

    p += strlen(pattern);
    while (isspace(*p)) {
        ++p;
    }

    if (*p == ':') {
        ++p;
    }

    while (isspace(*p)) {
        ++p;
    }

    // Remove trailing newline
    size_t len = strlen(p);
    if (len > 0 && p[len - 1] == '\n') {
        p[len - 1] = '\0';
    }

    // Ensure we call the const char* assignment (which performs a copy)
    // instead of the char* overload (which would take ownership of the pointer)
    value = (const char*)p;
    return true;
}

static bool read_riscv_cpuinfo(riscv_cpu_desc *desc)
{
    auto fp = fopen("/proc/cpuinfo", "r");
    if (!fp) {
        return false;
    }

    char buf[2048]; // Larger buffer for long ISA strings
    while (fgets(buf, sizeof(buf), fp) != nullptr) {
        lookup_riscv(buf, "model name", desc->model);
        
        if (lookup_riscv(buf, "isa", desc->isa)) {
            desc->isa.toLower();

            for (const String& s : desc->isa.split('_')) {
                const char* p = s.data();
                const size_t n = s.size();

                if ((s.size() > 4) && (memcmp(p, "rv64", 4) == 0)) {
                    for (size_t i = 4; i < n; ++i) {
                        if (p[i] == 'v') {
                            desc->has_vector = true;
                            break;
                        }
                    }
                }
                else if (s == "zve64d") {
                    desc->has_vector = true;
                }
                else if ((s == "zvkn") || (s == "zvknc") || (s == "zvkned") || (s == "zvkng")){
                    desc->has_aes = true;
                }
            }
        }
        
        lookup_riscv(buf, "uarch", desc->uarch);

        if (desc->isReady()) {
            break;
        }
    }

    fclose(fp);

    return desc->isReady();
}

String cpu_name_riscv()
{
    riscv_cpu_desc desc;
    if (read_riscv_cpuinfo(&desc)) {
        if (!desc.uarch.isNull()) {
            return fmt::format("{} ({})", desc.model, desc.uarch).c_str();
        }
        return desc.model;
    }

    return "RISC-V";
}

bool has_riscv_vector()
{
    riscv_cpu_desc desc;
    if (read_riscv_cpuinfo(&desc)) {
        return desc.has_vector;
    }
    return false;
}

bool has_riscv_aes()
{
    riscv_cpu_desc desc;
    if (read_riscv_cpuinfo(&desc)) {
        return desc.has_aes;
    }
    return false;
}


// Reads the vlenb CSR. Only valid once the V extension is known to be present,
// otherwise it traps. The .option block keeps this assembling even when the
// translation unit is built without "v" in -march.
static inline uint32_t read_vlenb()
{
    unsigned long vlenb;

    __asm__ volatile(
        ".option push\n\t"
        ".option arch, +v\n\t"
        "vsetvli t0, x0, e8, m1, ta, ma\n\t"
        "csrr %0, vlenb\n\t"
        ".option pop"
        : "=r"(vlenb)
        :
        : "t0");

    return static_cast<uint32_t>(vlenb);
}


uint32_t riscv_vlen()
{
    static thread_local uint32_t vlen = UINT32_MAX;

    if (vlen == UINT32_MAX) {
        vlen = has_riscv_vector() ? read_vlenb() * 8 : 0;
    }

    return vlen;
}

} // namespace xmrig
