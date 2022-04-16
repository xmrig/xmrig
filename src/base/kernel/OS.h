/* XMRig
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_OS_H
#define XMRIG_OS_H


#include "base/tools/String.h"


#include <string>


namespace xmrig {


class OS
{
public:
    static const char *arch;

    static inline bool isUserActive(uint64_t ms)            { return idleTime() < ms; }
    static inline bool trySetThreadAffinity(int64_t cpu_id) { return cpu_id >= 0 && setThreadAffinity(static_cast<uint64_t>(cpu_id)); }

    static bool isOnBatteryPower();
    static bool setThreadAffinity(uint64_t cpu_id);
    static std::string name();
    static std::string userAgent();
    static String hostname();
    static uint64_t freemem();
    static uint64_t idleTime();
    static uint64_t totalmem();
    static void destroy();
    static void init();
    static void setProcessPriority(int priority);
    static void setThreadPriority(int priority);
};


} // namespace xmrig


#endif // XMRIG_OS_H
