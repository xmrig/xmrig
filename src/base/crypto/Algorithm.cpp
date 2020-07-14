/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include "base/crypto/Algorithm.h"
#include "3rdparty/rapidjson/document.h"


#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>


#ifdef _MSC_VER
#   define strcasecmp  _stricmp
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
    { "cryptonight/fast",          "cn/fast",          Algorithm::CN_FAST         },
    { "cryptonight/msr",           "cn/msr",           Algorithm::CN_FAST         },
    { "cryptonight/half",          "cn/half",          Algorithm::CN_HALF         },
    { "cryptonight/xao",           "cn/xao",           Algorithm::CN_XAO          },
    { "cryptonight_alloy",         nullptr,            Algorithm::CN_XAO          },
    { "cryptonight/rto",           "cn/rto",           Algorithm::CN_RTO          },
    { "cryptonight/rwz",           "cn/rwz",           Algorithm::CN_RWZ          },
    { "cryptonight/zls",           "cn/zls",           Algorithm::CN_ZLS          },
    { "cryptonight/double",        "cn/double",        Algorithm::CN_DOUBLE       },
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
    { "cryptonight-pico/tlo",      "cn-pico/tlo",      Algorithm::CN_PICO_TLO     },
    { "cryptonight/ultra",         "cn/ultra",         Algorithm::CN_PICO_TLO     },
    { "cryptonight-talleo",        "cn-talleo",        Algorithm::CN_PICO_TLO     },
    { "cryptonight_talleo",        "cn_talleo",        Algorithm::CN_PICO_TLO     },
#   endif
#   ifdef XMRIG_ALGO_RANDOMX
    { "randomx/0",                 "rx/0",             Algorithm::RX_0            },
    { "randomx/test",              "rx/test",          Algorithm::RX_0            },
    { "RandomX",                   "rx",               Algorithm::RX_0            },
    { "randomx/wow",               "rx/wow",           Algorithm::RX_WOW          },
    { "RandomWOW",                 nullptr,            Algorithm::RX_WOW          },
    { "randomx/loki",              "rx/loki",          Algorithm::RX_LOKI         },
    { "RandomXL",                  nullptr,            Algorithm::RX_LOKI         },
    { "randomx/arq",               "rx/arq",           Algorithm::RX_ARQ          },
    { "RandomARQ",                 nullptr,            Algorithm::RX_ARQ          },
    { "randomx/sfx",               "rx/sfx",           Algorithm::RX_SFX          },
    { "RandomSFX",                 nullptr,            Algorithm::RX_SFX          },
    { "randomx/keva",              "rx/keva",          Algorithm::RX_KEVA         },
    { "RandomKEVA",                nullptr,            Algorithm::RX_KEVA         },
    { "defyx",                     "defyx",            Algorithm::RX_DEFYX        },
    { "DefyX",                     nullptr,            Algorithm::RX_DEFYX        },
#   endif
#   ifdef XMRIG_ALGO_ARGON2
    { "argon2/chukwa",             nullptr,            Algorithm::AR2_CHUKWA      },
    { "chukwa",                    nullptr,            Algorithm::AR2_CHUKWA      },
    { "argon2/wrkz",               nullptr,            Algorithm::AR2_WRKZ        },
#   endif
#   ifdef XMRIG_ALGO_ASTROBWT
    { "astrobwt",                  nullptr,            Algorithm::ASTROBWT_DERO   },
    { "astrobwt/dero",             nullptr,            Algorithm::ASTROBWT_DERO   },
#   endif
#   ifdef XMRIG_ALGO_KAWPOW
    { "kawpow",                    nullptr,            Algorithm::KAWPOW_RVN      },
    { "kawpow/rvn",                nullptr,            Algorithm::KAWPOW_RVN      },
#   endif
    { "cryptonight/ccx",           "cn/ccx",           Algorithm::CN_CCX          },
    { "cryptonight/conceal",       "cn/conceal",       Algorithm::CN_CCX          },
#   ifdef XMRIG_ALGO_CN_GPU
    { "cryptonight/gpu",           "cn/gpu",           Algorithm::CN_GPU          },
    { "cryptonight_gpu",           nullptr,            Algorithm::CN_GPU          },
#   endif
};


} /* namespace xmrig */


rapidjson::Value xmrig::Algorithm::toJSON() const
{
    using namespace rapidjson;

    return isValid() ? Value(StringRef(shortName())) : Value(kNullType);
}


size_t xmrig::Algorithm::l2() const
{
#   ifdef XMRIG_ALGO_RANDOMX
    switch (m_id) {
    case RX_0:
    case RX_LOKI:
    case RX_SFX:
        return 0x40000;

    case RX_WOW:
    case RX_KEVA:
    case RX_DEFYX:
        return 0x20000;

    case RX_ARQ:
        return 0x10000;

    default:
        break;
    }
#   endif

    return 0;
}


size_t xmrig::Algorithm::l3() const
{
    constexpr size_t oneMiB = 0x100000;

    const auto f = family();
    assert(f != UNKNOWN);

    switch (f) {
    case CN:
        return oneMiB * 2;

    case CN_LITE:
        return oneMiB;

    case CN_HEAVY:
        return oneMiB * 4;

    case CN_PICO:
        return oneMiB / 4;

    default:
        break;
    }

#   ifdef XMRIG_ALGO_RANDOMX
    if (f == RANDOM_X) {
        switch (m_id) {
        case RX_0:
        case RX_LOKI:
        case RX_SFX:
            return oneMiB * 2;

        case RX_WOW:
        case RX_KEVA:
            return oneMiB;

        case RX_ARQ:
        case RX_DEFYX:
            return oneMiB / 4;

        default:
            break;
        }
    }
#   endif

#   ifdef XMRIG_ALGO_ARGON2
    if (f == ARGON2) {
        switch (m_id) {
        case AR2_CHUKWA:
            return oneMiB / 2;

        case AR2_WRKZ:
            return oneMiB / 4;

        default:
            break;
        }
    }
#   endif

#   ifdef XMRIG_ALGO_ASTROBWT
    if (f == ASTROBWT) {
        switch (m_id) {
        case ASTROBWT_DERO:
            return oneMiB * 20;

        default:
            break;
        }
    }
#   endif

#   ifdef XMRIG_ALGO_KAWPOW
    if (f == KAWPOW) {
        switch (m_id) {
        case KAWPOW_RVN:
            return 32768;

        default:
            break;
        }
    }
#   endif

    return 0;
}


uint32_t xmrig::Algorithm::maxIntensity() const
{
#   ifdef XMRIG_ALGO_RANDOMX
    if (family() == RANDOM_X) {
        return 1;
    }
#   endif

#   ifdef XMRIG_ALGO_ARGON2
    if (family() == ARGON2) {
        return 1;
    }
#   endif

#   ifdef XMRIG_ALGO_ASTROBWT
    if (family() == ASTROBWT) {
        return 1;
    }
#   endif

#   ifdef XMRIG_ALGO_CN_GPU
    if (m_id == CN_GPU) {
        return 1;
    }
#   endif

    return 5;
}


xmrig::Algorithm::Family xmrig::Algorithm::family(Id id)
{
    switch (id) {
    case CN_0:
    case CN_1:
    case CN_2:
    case CN_R:
    case CN_FAST:
    case CN_HALF:
    case CN_XAO:
    case CN_RTO:
    case CN_RWZ:
    case CN_ZLS:
    case CN_DOUBLE:
    case CN_CCX:
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
    case CN_PICO_0:
    case CN_PICO_TLO:
        return CN_PICO;
#   endif

#   ifdef XMRIG_ALGO_RANDOMX
    case RX_0:
    case RX_WOW:
    case RX_LOKI:
    case RX_ARQ:
    case RX_SFX:
    case RX_KEVA:
    case RX_DEFYX:
        return RANDOM_X;
#   endif

#   ifdef XMRIG_ALGO_ARGON2
    case AR2_CHUKWA:
    case AR2_WRKZ:
        return ARGON2;
#   endif

#   ifdef XMRIG_ALGO_ASTROBWT
    case ASTROBWT_DERO:
        return ASTROBWT;
#   endif

#   ifdef XMRIG_ALGO_KAWPOW
    case KAWPOW_RVN:
        return KAWPOW;
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

    for (const AlgoName &item : algorithm_names) {
        if ((strcasecmp(name, item.name) == 0) || (item.shortName != nullptr && strcasecmp(name, item.shortName) == 0)) {
            return item.id;
        }
    }

    return INVALID;
}


const char *xmrig::Algorithm::name(bool shortName) const
{
    for (const AlgoName &item : algorithm_names) {
        if (item.id == m_id) {
            return (shortName && item.shortName) ? item.shortName : item.name;
        }
    }

    return "invalid";
}
