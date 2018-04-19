/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2018 XMRig       <support@xmrig.com>
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

#ifndef __ICONFIG_H__
#define __ICONFIG_H__


#include "rapidjson/fwd.h"


namespace xmrig {


class IConfig
{
public:
    enum Keys {
        // common
        AlgorithmKey      = 'a',
        ApiPort           = 4000,
        ApiAccessTokenKey = 4001,
        ApiWorkerIdKey    = 4002,
        ApiIPv6Key        = 4003,
        ApiRestrictedKey  = 4004,
        BackgroundKey     = 'B',
        ConfigKey         = 'c',
        DonateLevelKey    = 1003,
        HelpKey           = 'h',
        KeepAliveKey      = 'k',
        LogFileKey        = 'l',
        ColorKey          = 1002,
        WatchKey          = 1105,
        PasswordKey       = 'p',
        RetriesKey        = 'r',
        RetryPauseKey     = 'R',
        SyslogKey         = 'S',
        UrlKey            = 'o',
        UserKey           = 'u',
        UserAgentKey      = 1008,
        UserpassKey       = 'O',
        VerboseKey        = 1100,
        VersionKey        = 'V',
        VariantKey        = 1010,

        // xmrig common
        CPUPriorityKey    = 1021,
        NicehashKey       = 1006,
        PrintTimeKey      = 1007,

        // xmrig cpu
        AVKey             = 'v',
        CPUAffinityKey    = 1020,
        DryRunKey         = 5000,
        HugePagesKey      = 1009,
        MaxCPUUsageKey    = 1004,
        SafeKey           = 1005,
        ThreadsKey        = 't',
        HardwareAESKey    = 1011,

        // xmrig-proxy
        AccessLogFileKey  = 'A',
        BindKey           = 'b',
        CoinKey           = 1104,
        CustomDiffKey     = 1102,
        DebugKey          = 1101,
        ModeKey           = 'm',
        PoolCoinKey       = 'C',
        ReuseTimeoutKey   = 1106,
        WorkersKey        = 1103,
    };

    virtual ~IConfig() {}

    virtual bool adjust()                                  = 0;
    virtual bool isValid() const                           = 0;
    virtual bool isWatch() const                           = 0;
    virtual bool parseBoolean(int key, bool enable)        = 0;
    virtual bool parseString(int key, const char *arg)     = 0;
    virtual bool parseUint64(int key, uint64_t arg)        = 0;
    virtual bool save()                                    = 0;
    virtual const char *fileName() const                   = 0;
    virtual void getJSON(rapidjson::Document &doc) const   = 0;
    virtual void parseJSON(const rapidjson::Document &doc) = 0;
    virtual void setFileName(const char *fileName)         = 0;
};


} /* namespace xmrig */


#endif // __ICONFIG_H__
