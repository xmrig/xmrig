/* XMRig
 * Copyright (c) 2026       XMRig LoongArch port <https://github.com/xmrig>
 * Copyright (c) 2025      Slayingripper <https://github.com/Slayingripper>
 * Copyright (c) 2018-2025 SChernykh     <https://github.com/SChernykh>
 * Copyright (c) 2016-2026 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#include <cctype>
#include <cstdio>
#include <cstring>
#include <string>

namespace xmrig {

struct loongarch_cpu_desc
{
    String model;
    String family;
    String prid;
    bool has_lsx = false;
    bool has_lasx = false;

    inline bool isReady() const { return !model.isNull(); }
};

static bool lookup_loongarch(char *line, const char *pattern, String &value)
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

    size_t len = strlen(p);
    while (len > 0 && (p[len - 1] == '\n' || p[len - 1] == '\r')) {
        p[--len] = '\0';
    }

    if (len == 0) {
        return false;
    }

    value = (const char*) p;
    return true;
}

static bool has_word(const String &line, const char *word)
{
    return line.contains(word);
}

static bool read_loongarch_cpuinfo(loongarch_cpu_desc *desc)
{
    auto fp = fopen("/proc/cpuinfo", "r");
    if (!fp) {
        return false;
    }

    char buf[2048];
    while (fgets(buf, sizeof(buf), fp) != nullptr) {
        lookup_loongarch(buf, "Model Name", desc->model);
        lookup_loongarch(buf, "CPU Family", desc->family);

        if (lookup_loongarch(buf, "PRID", desc->prid)) {
            continue;
        }

        static const char *kFeatures = "Features";
        char *p = strstr(buf, kFeatures);

        if (p) {
            String features;
            if (lookup_loongarch(buf, kFeatures, features)) {
                desc->has_lsx = has_word(features, "lsx");
                desc->has_lasx = has_word(features, "lasx");
            }
        }

        if (desc->isReady()) {
            break;
        }
    }

    fclose(fp);

    return desc->isReady();
}

String cpu_name_loongarch()
{
    loongarch_cpu_desc desc;
    if (read_loongarch_cpuinfo(&desc) && !desc.model.isNull()) {
        if (!desc.prid.isNull()) {
            return fmt::format("{} ({})", desc.model, desc.prid).c_str();
        }

        return desc.model;
    }

    return "LoongArch";
}

bool has_loongarch_lsx()
{
    loongarch_cpu_desc desc;
    return read_loongarch_cpuinfo(&desc) && desc.has_lsx;
}

bool has_loongarch_lasx()
{
    loongarch_cpu_desc desc;
    return read_loongarch_cpuinfo(&desc) && desc.has_lasx;
}

} // namespace xmrig
