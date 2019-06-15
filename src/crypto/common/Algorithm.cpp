/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
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


#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>


#include "crypto/common/Algorithm.h"
#include "rapidjson/document.h"


#ifdef _MSC_VER
#   define strncasecmp _strnicmp
#   define strcasecmp  _stricmp
#endif


#ifndef ARRAY_SIZE
#   define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#endif


namespace xmrig {


struct AlgoName
{
    const char *name;
    const char *shortName;
    const Algorithm::Id id;
};


static AlgoName const algorithm_names[] = {
    { "cryptonight/0",             "cn/0",             Algorithm::CN_0            },
    { "cryptonight",               "cn",               Algorithm::CN_0            },
    { "cryptonight/1",             "cn/1",             Algorithm::CN_1            },
    { "cryptonight-monerov7",      nullptr,            Algorithm::CN_1            },
    { "cryptonight_v7",            nullptr,            Algorithm::CN_1            },
    { "cryptonight/2",             "cn/2",             Algorithm::CN_2            },
    { "cryptonight-monerov8",      nullptr,            Algorithm::CN_2            },
    { "cryptonight_v8",            nullptr,            Algorithm::CN_2            },
    { "cryptonight/r",             "cn/r",             Algorithm::CN_R            },
    { "cryptonight_r",             nullptr,            Algorithm::CN_R            },
    { "cryptonight/wow",           "cn/wow",           Algorithm::CN_WOW          },
    { "cryptonight/fast",          "cn/fast",          Algorithm::CN_FAST         },
    { "cryptonight/msr",           "cn/msr",           Algorithm::CN_FAST         },
    { "cryptonight/half",          "cn/half",          Algorithm::CN_HALF         },
    { "cryptonight/xao",           "cn/xao",           Algorithm::CN_XAO          },
    { "cryptonight_alloy",         nullptr,            Algorithm::CN_XAO          },
    { "cryptonight/rto",           "cn/rto",           Algorithm::CN_RTO          },
    { "cryptonight/rwz",           "cn/rwz",           Algorithm::CN_RWZ          },
    { "cryptonight/zls",           "cn/zls",           Algorithm::CN_ZLS          },
    { "cryptonight/double",        "cn/double",        Algorithm::CN_DOUBLE       },
#   ifdef XMRIG_ALGO_CN_GPU
    { "cryptonight/gpu",           "cn/gpu",           Algorithm::CN_GPU          },
    { "cryptonight_gpu",           nullptr,            Algorithm::CN_GPU          },
#   endif
#   ifdef XMRIG_ALGO_CN_LITE
    { "cryptonight-lite/0",        "cn-lite/0",        Algorithm::CN_LITE_0       },
    { "cryptonight-lite/1",        "cn-lite/1",        Algorithm::CN_LITE_1       },
    { "cryptonight-lite",          "cn-lite",          Algorithm::CN_LITE_1       },
    { "cryptonight-light",         "cn-light",         Algorithm::CN_LITE_1       },
    { "cryptonight_lite",          nullptr,            Algorithm::CN_LITE_1       },
    { "cryptonight-aeonv7",        nullptr,            Algorithm::CN_LITE_1       },
    { "cryptonight_lite_v7",       nullptr,            Algorithm::CN_LITE_1       },
#   endif
#   ifdef XMRIG_ALGO_CN_HEAVY
    { "cryptonight-heavy/0",       "cn-heavy/0",       Algorithm::CN_HEAVY_0      },
    { "cryptonight-heavy",         "cn-heavy",         Algorithm::CN_HEAVY_0      },
    { "cryptonight_heavy",         nullptr,            Algorithm::CN_HEAVY_0      },
    { "cryptonight-heavy/xhv",     "cn-heavy/xhv",     Algorithm::CN_HEAVY_XHV    },
    { "cryptonight_haven",         nullptr,            Algorithm::CN_HEAVY_XHV    },
    { "cryptonight-heavy/tube",    "cn-heavy/tube",    Algorithm::CN_HEAVY_TUBE   },
    { "cryptonight-bittube2",      nullptr,            Algorithm::CN_HEAVY_TUBE   },
#   endif
#   ifdef XMRIG_ALGO_CN_PICO
    { "cryptonight-pico",          "cn-pico",          Algorithm::CN_PICO_0       },
    { "cryptonight-pico/trtl",     "cn-pico/trtl",     Algorithm::CN_PICO_0       },
    { "cryptonight-turtle",        "cn-trtl",          Algorithm::CN_PICO_0       },
    { "cryptonight-ultralite",     "cn-ultralite",     Algorithm::CN_PICO_0       },
    { "cryptonight_turtle",        "cn_turtle",        Algorithm::CN_PICO_0       },
#   endif
};


} /* namespace xmrig */


rapidjson::Value xmrig::Algorithm::toJSON() const
{
    using namespace rapidjson;

    return isValid() ? Value(StringRef(shortName())) : Value(kNullType);
}


xmrig::Algorithm::Family xmrig::Algorithm::family(Id id)
{
    switch (id) {
    case CN_0:
    case CN_1:
    case CN_2:
    case CN_R:
    case CN_WOW:
    case CN_FAST:
    case CN_HALF:
    case CN_XAO:
    case CN_RTO:
    case CN_RWZ:
    case CN_DOUBLE:
#   ifdef XMRIG_ALGO_CN_GPU
    case CN_GPU:
#   endif
        return CN;

#   ifdef XMRIG_ALGO_CN_LITE
    case CN_LITE_0:
    case CN_LITE_1:
        return CN_LITE;
#   endif

#   ifdef XMRIG_ALGO_CN_HEAVY
    case CN_HEAVY_0:
    case CN_HEAVY_TUBE:
    case CN_HEAVY_XHV:
        return CN_HEAVY;
#   endif

#   ifdef XMRIG_ALGO_CN_PICO
    case Algorithm::CN_PICO_0:
        return CN_PICO;
#   endif

    default:
        break;
    }

    return UNKNOWN;
}


xmrig::Algorithm::Id xmrig::Algorithm::parse(const char *name)
{
    if (name == nullptr || strlen(name) < 1) {
        return INVALID;
    }

    for (size_t i = 0; i < ARRAY_SIZE(algorithm_names); i++) {
        if ((strcasecmp(name, algorithm_names[i].name) == 0) || (algorithm_names[i].shortName != nullptr && strcasecmp(name, algorithm_names[i].shortName) == 0)) {
            return algorithm_names[i].id;
        }
    }

    return INVALID;
}


const char *xmrig::Algorithm::name(bool shortName) const
{
    for (size_t i = 0; i < ARRAY_SIZE(algorithm_names); i++) {
        if (algorithm_names[i].id == m_id) {
            return shortName ? algorithm_names[i].shortName : algorithm_names[i].name;
        }
    }

    return "invalid";
}
