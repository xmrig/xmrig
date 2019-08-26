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

#ifndef XMRIG_CONFIGLOADER_PLATFORM_H
#define XMRIG_CONFIGLOADER_PLATFORM_H


#ifdef _MSC_VER
#   include "getopt/getopt.h"
#else
#   include <getopt.h>
#endif


#include "common/interfaces/IConfig.h"
#include "version.h"


namespace xmrig {


static char const short_options[] = "a:c:Bp:Px:r:R:s:t:T:o:u:O:v:l:S";


static struct option const options[] = {
    { "algo",              1, nullptr, xmrig::IConfig::AlgorithmKey      },
    { "api-access-token",  1, nullptr, xmrig::IConfig::ApiAccessTokenKey },
    { "api-port",          1, nullptr, xmrig::IConfig::ApiPort           },
    { "api-worker-id",     1, nullptr, xmrig::IConfig::ApiWorkerIdKey    },
    { "api-id",            1, nullptr, xmrig::IConfig::ApiIdKey          },
    { "api-ipv6",          0, nullptr, xmrig::IConfig::ApiIPv6Key        },
    { "api-no-restricted", 0, nullptr, xmrig::IConfig::ApiRestrictedKey  },
    { "background",        0, nullptr, xmrig::IConfig::BackgroundKey     },
    { "config",            1, nullptr, xmrig::IConfig::ConfigKey         },
    { "cpu-threads",       1, nullptr, xmrig::IConfig::CPUThreadsKey     },
    { "cpu-optimization",  1, nullptr, xmrig::IConfig::CPUOptimizationKey},
    { "cpu-affinity",      1, nullptr, xmrig::IConfig::CPUAffinityKey    },
    { "use-gpu",           1, nullptr, xmrig::IConfig::UseGPUKey         },
    { "gpu-intensity",     1, nullptr, xmrig::IConfig::GPUIntensityKey   },
    { "gpu-filter",        1, nullptr, xmrig::IConfig::GPUFilterKey      },
    { "priority",          1, nullptr, xmrig::IConfig::PriorityKey       },
    { "donate-level",      1, nullptr, xmrig::IConfig::DonateLevelKey    },
    { "dry-run",           0, nullptr, xmrig::IConfig::DryRunKey         },
    { "keepalive",         0, nullptr, xmrig::IConfig::KeepAliveKey      },
    { "log-file",          1, nullptr, xmrig::IConfig::LogFileKey        },
    { "nicehash",          0, nullptr, xmrig::IConfig::NicehashKey       },
    { "no-color",          0, nullptr, xmrig::IConfig::ColorKey          },
    { "no-watch",          0, nullptr, xmrig::IConfig::WatchKey          },
    { "variant",           1, nullptr, xmrig::IConfig::VariantKey        },
    { "pass",              1, nullptr, xmrig::IConfig::PasswordKey       },
    { "print-time",        1, nullptr, xmrig::IConfig::PrintTimeKey      },
    { "retries",           1, nullptr, xmrig::IConfig::RetriesKey        },
    { "retry-pause",       1, nullptr, xmrig::IConfig::RetryPauseKey     },
    { "syslog",            0, nullptr, xmrig::IConfig::SyslogKey         },
    { "url",               1, nullptr, xmrig::IConfig::UrlKey            },
    { "user",              1, nullptr, xmrig::IConfig::UserKey           },
    { "user-agent",        1, nullptr, xmrig::IConfig::UserAgentKey      },
    { "userpass",          1, nullptr, xmrig::IConfig::UserpassKey       },
    { "rig-id",            1, nullptr, xmrig::IConfig::RigIdKey          },
    { "tls",               0, nullptr, xmrig::IConfig::TlsKey            },
    { "tls-fingerprint",   1, nullptr, xmrig::IConfig::FingerprintKey    },
    { nullptr,             0, nullptr, 0 }
};


static struct option const config_options[] = {
    { "algo",          1, nullptr, xmrig::IConfig::AlgorithmKey   },
    { "background",    0, nullptr, xmrig::IConfig::BackgroundKey  },
    { "colors",        0, nullptr, xmrig::IConfig::ColorKey       },
    { "cpu-threads",   1, nullptr, xmrig::IConfig::CPUThreadsKey  },
    { "cpu-optimization",1, nullptr, xmrig::IConfig::CPUOptimizationKey },
    { "cpu-affinity",  1, nullptr, xmrig::IConfig::CPUAffinityKey },
    { "use-gpu",       1, nullptr, xmrig::IConfig::UseGPUKey      },
    { "gpu-intensity", 1, nullptr, xmrig::IConfig::GPUIntensityKey},
    { "gpu-filter",    1, nullptr, xmrig::IConfig::GPUFilterKey   },
    { "priority",      1, nullptr, xmrig::IConfig::PriorityKey    },
    { "donate-level",  1, nullptr, xmrig::IConfig::DonateLevelKey },
    { "dry-run",       0, nullptr, xmrig::IConfig::DryRunKey      },
    { "log-file",      1, nullptr, xmrig::IConfig::LogFileKey     },
    { "print-time",    1, nullptr, xmrig::IConfig::PrintTimeKey   },
    { "retries",       1, nullptr, xmrig::IConfig::RetriesKey     },
    { "retry-pause",   1, nullptr, xmrig::IConfig::RetryPauseKey  },
    { "syslog",        0, nullptr, xmrig::IConfig::SyslogKey      },
    { "user-agent",    1, nullptr, xmrig::IConfig::UserAgentKey   },
    { "watch",         0, nullptr, xmrig::IConfig::WatchKey       },
    { "autosave",      0, nullptr, xmrig::IConfig::AutoSaveKey    },
    { nullptr,         0, nullptr, 0 }
};


static struct option const api_options[] = {
    { "port",          1, nullptr, xmrig::IConfig::ApiPort           },
    { "access-token",  1, nullptr, xmrig::IConfig::ApiAccessTokenKey },
    { "worker-id",     1, nullptr, xmrig::IConfig::ApiWorkerIdKey    },
    { "ipv6",          0, nullptr, xmrig::IConfig::ApiIPv6Key        },
    { "restricted",    0, nullptr, xmrig::IConfig::ApiRestrictedKey  },
    { "id",            1, nullptr, xmrig::IConfig::ApiIdKey          },
    { nullptr,         0, nullptr, 0 }
};


} /* namespace xmrig */

#endif /* XMRIG_CONFIGLOADER_PLATFORM_H */
