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

#include "base/tools/String.h"
#include "3rdparty/fmt/core.h"

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
    bool has_crypto = false;
    
    inline bool isReady() const { return !model.isNull(); }
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
            // Check for vector extensions
            if (strstr(buf, "zve") || strstr(buf, "v_")) {
                desc->has_vector = true;
            }
            // Check for crypto extensions (AES, SHA, etc.)
            // zkn* = NIST crypto suite, zks* = SM crypto suite
            // Note: zba/zbb/zbc/zbs are bit-manipulation, NOT crypto
            if (strstr(buf, "zknd") || strstr(buf, "zkne") || strstr(buf, "zknh") ||
                strstr(buf, "zksed") || strstr(buf, "zksh")) {
                desc->has_crypto = true;
            }
        }
        
        lookup_riscv(buf, "uarch", desc->uarch);

        if (desc->isReady() && !desc->isa.isNull()) {
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

bool has_riscv_crypto()
{
    riscv_cpu_desc desc;
    if (read_riscv_cpuinfo(&desc)) {
        return desc.has_crypto;
    }
    return false;
}

} // namespace xmrig
