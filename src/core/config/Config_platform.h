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

#ifndef XMRIG_CONFIG_PLATFORM_H
#define XMRIG_CONFIG_PLATFORM_H


#ifdef _MSC_VER
#   include "getopt/getopt.h"
#else
#   include <getopt.h>
#endif


#include "base/kernel/interfaces/IConfig.h"
#include "version.h"


namespace xmrig {


static const char short_options[] = "a:c:kBp:Px:r:R:s:t:T:o:u:O:v:l:S";


static const option options[] = {
    { "algo",                  1, nullptr, IConfig::AlgorithmKey          },
    { "api-worker-id",         1, nullptr, IConfig::ApiWorkerIdKey        },
    { "api-id",                1, nullptr, IConfig::ApiIdKey              },
    { "http-enabled",          0, nullptr, IConfig::HttpEnabledKey        },
    { "http-host",             1, nullptr, IConfig::HttpHostKey           },
    { "http-access-token",     1, nullptr, IConfig::HttpAccessTokenKey    },
    { "http-port",             1, nullptr, IConfig::HttpPort              },
    { "http-no-restricted",    0, nullptr, IConfig::HttpRestrictedKey     },
    { "av",                    1, nullptr, IConfig::AVKey                 },
    { "background",            0, nullptr, IConfig::BackgroundKey         },
    { "config",                1, nullptr, IConfig::ConfigKey             },
    { "cpu-affinity",          1, nullptr, IConfig::CPUAffinityKey        },
    { "cpu-priority",          1, nullptr, IConfig::CPUPriorityKey        },
    { "donate-level",          1, nullptr, IConfig::DonateLevelKey        },
    { "donate-over-proxy",     1, nullptr, IConfig::ProxyDonateKey        },
    { "dry-run",               0, nullptr, IConfig::DryRunKey             },
    { "keepalive",             0, nullptr, IConfig::KeepAliveKey          },
    { "log-file",              1, nullptr, IConfig::LogFileKey            },
    { "nicehash",              0, nullptr, IConfig::NicehashKey           },
    { "no-color",              0, nullptr, IConfig::ColorKey              },
    { "no-huge-pages",         0, nullptr, IConfig::HugePagesKey          },
    { "pass",                  1, nullptr, IConfig::PasswordKey           },
    { "print-time",            1, nullptr, IConfig::PrintTimeKey          },
    { "retries",               1, nullptr, IConfig::RetriesKey            },
    { "retry-pause",           1, nullptr, IConfig::RetryPauseKey         },
    { "syslog",                0, nullptr, IConfig::SyslogKey             },
    { "threads",               1, nullptr, IConfig::ThreadsKey            },
    { "url",                   1, nullptr, IConfig::UrlKey                },
    { "user",                  1, nullptr, IConfig::UserKey               },
    { "user-agent",            1, nullptr, IConfig::UserAgentKey          },
    { "userpass",              1, nullptr, IConfig::UserpassKey           },
    { "rig-id",                1, nullptr, IConfig::RigIdKey              },
    { "tls",                   0, nullptr, IConfig::TlsKey                },
    { "tls-fingerprint",       1, nullptr, IConfig::FingerprintKey        },
    { "asm",                   1, nullptr, IConfig::AssemblyKey           },
    { "daemon",                0, nullptr, IConfig::DaemonKey             },
    { "daemon-poll-interval",  1, nullptr, IConfig::DaemonPollKey         },
    { nullptr,                 0, nullptr, 0 }
};


} // namespace xmrig


#endif /* XMRIG_CONFIG_PLATFORM_H */
