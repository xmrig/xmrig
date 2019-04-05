/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2019 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include "core/config/ConfigTransform.h"
#include "base/kernel/interfaces/IConfig.h"


xmrig::ConfigTransform::ConfigTransform()
{

}


void xmrig::ConfigTransform::transform(rapidjson::Document &doc, int key, const char *arg)
{
    BaseTransform::transform(doc, key, arg);

    switch (key) {
    case IConfig::AVKey:          /* --av */
    case IConfig::MaxCPUUsageKey: /* --max-cpu-usage */
    case IConfig::CPUPriorityKey: /* --cpu-priority */
    case IConfig::ThreadsKey:     /* --threads */
        return transformUint64(doc, key, static_cast<uint64_t>(strtol(arg, nullptr, 10)));

    case IConfig::SafeKey: /* --safe */
        return transformBoolean(doc, key, true);

    case IConfig::HugePagesKey: /* --no-huge-pages */
        return transformBoolean(doc, key, false);

    case IConfig::CPUAffinityKey: /* --cpu-affinity */
        {
            const char *p  = strstr(arg, "0x");
            return transformUint64(doc, key, p ? strtoull(p, nullptr, 16) : strtoull(arg, nullptr, 10));
        }

#   ifndef XMRIG_NO_ASM
    case IConfig::AssemblyKey: /* --asm */
        return set<const char *>(doc, "asm", arg);
#   endif

    default:
        break;
    }
}


void xmrig::ConfigTransform::transformBoolean(rapidjson::Document &doc, int key, bool enable)
{
    switch (key) {
    case IConfig::SafeKey: /* --safe */
        return set<bool>(doc, "safe", enable);

    case IConfig::HugePagesKey: /* --no-huge-pages */
        return set<bool>(doc, "huge-pages", enable);

    default:
        break;
    }
}


void xmrig::ConfigTransform::transformUint64(rapidjson::Document &doc, int key, uint64_t arg)
{
    switch (key) {
    case IConfig::CPUAffinityKey: /* --cpu-affinity */
        return set<int64_t>(doc, "cpu-affinity", static_cast<int64_t>(arg));

    case IConfig::ThreadsKey: /* --threads */
        return set<uint64_t>(doc, "threads", arg);

    case IConfig::AVKey: /* --av */
        return set<uint64_t>(doc, "av", arg);

    case IConfig::MaxCPUUsageKey: /* --max-cpu-usage */
        return set<uint64_t>(doc, "max-cpu-usage", arg);

    case IConfig::CPUPriorityKey: /* --cpu-priority */
        return set<uint64_t>(doc, "cpu-priority", arg);

    default:
        break;
    }
}
