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


static const char short_options[] = "a:c:kBp:Px:r:R:s:t:T:o:u:O:v:l:Sx:";


static const option options[] = {
    { "algo",                  1, nullptr, IConfig::AlgorithmKey          },
    { "coin",                  1, nullptr, IConfig::CoinKey               },
#   ifdef XMRIG_FEATURE_HTTP
    { "api-worker-id",         1, nullptr, IConfig::ApiWorkerIdKey        },
    { "api-id",                1, nullptr, IConfig::ApiIdKey              },
    { "http-enabled",          0, nullptr, IConfig::HttpEnabledKey        },
    { "http-host",             1, nullptr, IConfig::HttpHostKey           },
    { "http-access-token",     1, nullptr, IConfig::HttpAccessTokenKey    },
    { "http-port",             1, nullptr, IConfig::HttpPort              },
    { "http-no-restricted",    0, nullptr, IConfig::HttpRestrictedKey     },
    { "daemon",                0, nullptr, IConfig::DaemonKey             },
    { "daemon-poll-interval",  1, nullptr, IConfig::DaemonPollKey         },
    { "self-select",           1, nullptr, IConfig::SelfSelectKey         },
    { "submit-to-origin",      0, nullptr, IConfig::SubmitToOriginKey     },
#   endif
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
#   ifdef XMRIG_FEATURE_MO_BENCHMARK
    { "rebench-algo",          0, nullptr, IConfig::RebenchAlgoKey        },
    { "bench-algo-time",       1, nullptr, IConfig::BenchAlgoTimeKey      },
#   endif
    { "no-color",              0, nullptr, IConfig::ColorKey              },
    { "no-huge-pages",         0, nullptr, IConfig::HugePagesKey          },
    { "no-hugepages",          0, nullptr, IConfig::HugePagesKey          },
    { "hugepage-size",         1, nullptr, IConfig::HugePageSizeKey       },
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
    { "no-cpu",                0, nullptr, IConfig::CPUKey                },
    { "max-cpu-usage",         1, nullptr, IConfig::CPUMaxThreadsKey      },
    { "cpu-max-threads-hint",  1, nullptr, IConfig::CPUMaxThreadsKey      },
    { "cpu-memory-pool",       1, nullptr, IConfig::MemoryPoolKey         },
    { "cpu-no-yield",          0, nullptr, IConfig::YieldKey              },
    { "no-yield",              0, nullptr, IConfig::YieldKey              },
    { "cpu-argon2-impl",       1, nullptr, IConfig::Argon2ImplKey         },
    { "argon2-impl",           1, nullptr, IConfig::Argon2ImplKey         },
    { "verbose",               0, nullptr, IConfig::VerboseKey            },
    { "proxy",                 1, nullptr, IConfig::ProxyKey              },
    { "data-dir",              1, nullptr, IConfig::DataDirKey            },
    { "title",                 1, nullptr, IConfig::TitleKey              },
    { "no-title",              0, nullptr, IConfig::NoTitleKey            },
    { "pause-on-battery",      0, nullptr, IConfig::PauseOnBatteryKey     },
    { "pause-on-active",       1, nullptr, IConfig::PauseOnActiveKey     },
#   ifdef XMRIG_FEATURE_BENCHMARK
    { "stress",                0, nullptr, IConfig::StressKey             },
    { "bench",                 1, nullptr, IConfig::BenchKey              },
    { "benchmark",             1, nullptr, IConfig::BenchKey              },
#   ifdef XMRIG_FEATURE_HTTP
    { "submit",                0, nullptr, IConfig::BenchSubmitKey        },
    { "verify",                1, nullptr, IConfig::BenchVerifyKey        },
    { "token",                 1, nullptr, IConfig::BenchTokenKey         },
#   endif
    { "seed",                  1, nullptr, IConfig::BenchSeedKey          },
    { "hash",                  1, nullptr, IConfig::BenchHashKey          },
#   endif
#   ifdef XMRIG_FEATURE_TLS
    { "tls",                   0, nullptr, IConfig::TlsKey                },
    { "tls-fingerprint",       1, nullptr, IConfig::FingerprintKey        },
    { "tls-cert",              1, nullptr, IConfig::TlsCertKey            },
    { "tls-cert-key",          1, nullptr, IConfig::TlsCertKeyKey         },
    { "tls-dhparam",           1, nullptr, IConfig::TlsDHparamKey         },
    { "tls-protocols",         1, nullptr, IConfig::TlsProtocolsKey       },
    { "tls-ciphers",           1, nullptr, IConfig::TlsCiphersKey         },
    { "tls-ciphersuites",      1, nullptr, IConfig::TlsCipherSuitesKey    },
    { "tls-gen",               1, nullptr, IConfig::TlsGenKey             },
#   endif
#   ifdef XMRIG_FEATURE_ASM
    { "asm",                   1, nullptr, IConfig::AssemblyKey           },
#   endif
#   ifdef XMRIG_ALGO_RANDOMX
    { "randomx-init",          1, nullptr, IConfig::RandomXInitKey        },
    { "randomx-no-numa",       0, nullptr, IConfig::RandomXNumaKey        },
    { "randomx-mode",          1, nullptr, IConfig::RandomXModeKey        },
    { "randomx-1gb-pages",     0, nullptr, IConfig::RandomX1GbPagesKey    },
    { "1gb-pages",             0, nullptr, IConfig::RandomX1GbPagesKey    },
    { "randomx-wrmsr",         2, nullptr, IConfig::RandomXWrmsrKey       },
    { "wrmsr",                 2, nullptr, IConfig::RandomXWrmsrKey       },
    { "randomx-no-rdmsr",      0, nullptr, IConfig::RandomXRdmsrKey       },
    { "no-rdmsr",              0, nullptr, IConfig::RandomXRdmsrKey       },
    { "randomx-cache-qos",     0, nullptr, IConfig::RandomXCacheQoSKey    },
    { "cache-qos",             0, nullptr, IConfig::RandomXCacheQoSKey    },
#   endif
    #ifdef XMRIG_ALGO_ASTROBWT
    { "astrobwt-max-size",     1, nullptr, IConfig::AstroBWTMaxSizeKey    },
    { "astrobwt-avx2",         0, nullptr, IConfig::AstroBWTAVX2Key       },
    #endif
#   ifdef XMRIG_FEATURE_OPENCL
    { "opencl",                0, nullptr, IConfig::OclKey                },
    { "opencl-devices",        1, nullptr, IConfig::OclDevicesKey         },
    { "opencl-platform",       1, nullptr, IConfig::OclPlatformKey        },
    { "opencl-loader",         1, nullptr, IConfig::OclLoaderKey          },
    { "opencl-no-cache",       0, nullptr, IConfig::OclCacheKey           },
#   endif
#   ifdef XMRIG_FEATURE_CUDA
    { "cuda",                  0, nullptr, IConfig::CudaKey               },
    { "cuda-loader",           1, nullptr, IConfig::CudaLoaderKey         },
    { "cuda-devices",          1, nullptr, IConfig::CudaDevicesKey        },
    { "cuda-bfactor-hint",     1, nullptr, IConfig::CudaBFactorKey        },
    { "cuda-bsleep-hint",      1, nullptr, IConfig::CudaBSleepKey         },
#   endif
#   ifdef XMRIG_FEATURE_NVML
    { "no-nvml",               0, nullptr, IConfig::NvmlKey               },
#   endif
#   if defined(XMRIG_FEATURE_NVML) || defined (XMRIG_FEATURE_ADL)
    { "health-print-time",     1, nullptr, IConfig::HealthPrintTimeKey    },
#   endif
#   ifdef XMRIG_FEATURE_DMI
    { "no-dmi",                0, nullptr, IConfig::DmiKey                },
#   endif
    { nullptr,                 0, nullptr, 0 }
};


} // namespace xmrig


#endif /* XMRIG_CONFIG_PLATFORM_H */
