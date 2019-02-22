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
 * Copyright 2018-2019 MoneroOcean <https://github.com/MoneroOcean>, <support@moneroocean.stream>
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


#include "common/crypto/Algorithm.h"


#ifdef _MSC_VER
#   define strncasecmp _strnicmp
#   define strcasecmp  _stricmp
#endif


#ifndef ARRAY_SIZE
#   define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#endif


struct AlgoData
{
    const char *name;
    const char *shortName;
    xmrig::Algo algo;
    xmrig::Variant variant;
};


static AlgoData const algorithms[] = {
    { "cryptonight",           "cn",           xmrig::CRYPTONIGHT,       xmrig::VARIANT_AUTO },
    { "cryptonight/0",         "cn/0",         xmrig::CRYPTONIGHT,       xmrig::VARIANT_0    },
    { "cryptonight/1",         "cn/1",         xmrig::CRYPTONIGHT,       xmrig::VARIANT_1    },
    { "cryptonight/xtl",       "cn/xtl",       xmrig::CRYPTONIGHT,       xmrig::VARIANT_XTL  },
    { "cryptonight/msr",       "cn/msr",       xmrig::CRYPTONIGHT,       xmrig::VARIANT_MSR  },
    { "cryptonight/xao",       "cn/xao",       xmrig::CRYPTONIGHT,       xmrig::VARIANT_XAO  },
    { "cryptonight/rto",       "cn/rto",       xmrig::CRYPTONIGHT,       xmrig::VARIANT_RTO  },
    { "cryptonight/2",         "cn/2",         xmrig::CRYPTONIGHT,       xmrig::VARIANT_2    },
    { "cryptonight/half",      "cn/half",      xmrig::CRYPTONIGHT,       xmrig::VARIANT_HALF },
    { "cryptonight/xtlv9",     "cn/xtlv9",     xmrig::CRYPTONIGHT,       xmrig::VARIANT_HALF },
    { "cryptonight/wow",       "cn/wow",       xmrig::CRYPTONIGHT,       xmrig::VARIANT_WOW  },
    { "cryptonight/r",         "cn/r",         xmrig::CRYPTONIGHT,       xmrig::VARIANT_4    },

#   ifndef XMRIG_NO_AEON
    { "cryptonight-lite",      "cn-lite",      xmrig::CRYPTONIGHT_LITE,  xmrig::VARIANT_AUTO },
    { "cryptonight-light",     "cn-light",     xmrig::CRYPTONIGHT_LITE,  xmrig::VARIANT_AUTO },
    { "cryptonight-lite/0",    "cn-lite/0",    xmrig::CRYPTONIGHT_LITE,  xmrig::VARIANT_0    },
    { "cryptonight-lite/1",    "cn-lite/1",    xmrig::CRYPTONIGHT_LITE,  xmrig::VARIANT_1    },
#   endif

#   ifndef XMRIG_NO_SUMO
    { "cryptonight-heavy",      "cn-heavy",      xmrig::CRYPTONIGHT_HEAVY, xmrig::VARIANT_AUTO },
    { "cryptonight-heavy/0",    "cn-heavy/0",    xmrig::CRYPTONIGHT_HEAVY, xmrig::VARIANT_0    },
    { "cryptonight-heavy/xhv",  "cn-heavy/xhv",  xmrig::CRYPTONIGHT_HEAVY, xmrig::VARIANT_XHV  },
    { "cryptonight-heavy/tube", "cn-heavy/tube", xmrig::CRYPTONIGHT_HEAVY, xmrig::VARIANT_TUBE },
#   endif

#   ifndef XMRIG_NO_CN_PICO
    { "cryptonight-pico/trtl",  "cn-pico/trtl",  xmrig::CRYPTONIGHT_PICO, xmrig::VARIANT_TRTL },
    { "cryptonight-pico",       "cn-pico",       xmrig::CRYPTONIGHT_PICO, xmrig::VARIANT_TRTL },
    { "cryptonight-turtle",     "cn-trtl",       xmrig::CRYPTONIGHT_PICO, xmrig::VARIANT_TRTL },
    { "cryptonight-ultralite",  "cn-ultralite",  xmrig::CRYPTONIGHT_PICO, xmrig::VARIANT_TRTL },
    { "cryptonight_turtle",     "cn_turtle",     xmrig::CRYPTONIGHT_PICO, xmrig::VARIANT_TRTL },
#   endif

#   ifndef XMRIG_NO_CN_GPU
    { "cryptonight/gpu",        "cn/gpu",  xmrig::CRYPTONIGHT, xmrig::VARIANT_GPU },
#   endif
};


#ifdef XMRIG_PROXY_PROJECT
static AlgoData const xmrStakAlgorithms[] = {
    { "cryptonight-monerov7",    nullptr, xmrig::CRYPTONIGHT,       xmrig::VARIANT_1    },
    { "cryptonight_v7",          nullptr, xmrig::CRYPTONIGHT,       xmrig::VARIANT_1    },
    { "cryptonight-monerov8",    nullptr, xmrig::CRYPTONIGHT,       xmrig::VARIANT_2    },
    { "cryptonight_v8",          nullptr, xmrig::CRYPTONIGHT,       xmrig::VARIANT_2    },
    { "cryptonight_v7_stellite", nullptr, xmrig::CRYPTONIGHT,       xmrig::VARIANT_XTL  },
    { "cryptonight_lite",        nullptr, xmrig::CRYPTONIGHT_LITE,  xmrig::VARIANT_0    },
    { "cryptonight-aeonv7",      nullptr, xmrig::CRYPTONIGHT_LITE,  xmrig::VARIANT_1    },
    { "cryptonight_lite_v7",     nullptr, xmrig::CRYPTONIGHT_LITE,  xmrig::VARIANT_1    },
    { "cryptonight_heavy",       nullptr, xmrig::CRYPTONIGHT_HEAVY, xmrig::VARIANT_0    },
    { "cryptonight_haven",       nullptr, xmrig::CRYPTONIGHT_HEAVY, xmrig::VARIANT_XHV  },
    { "cryptonight_masari",      nullptr, xmrig::CRYPTONIGHT,       xmrig::VARIANT_MSR  },
    { "cryptonight_masari",      nullptr, xmrig::CRYPTONIGHT,       xmrig::VARIANT_MSR  },
    { "cryptonight-bittube2",    nullptr, xmrig::CRYPTONIGHT_HEAVY, xmrig::VARIANT_TUBE }, // bittube-miner
    { "cryptonight_alloy",       nullptr, xmrig::CRYPTONIGHT,       xmrig::VARIANT_XAO  }, // xmr-stak-alloy
    { "cryptonight_turtle",      nullptr, xmrig::CRYPTONIGHT_PICO,  xmrig::VARIANT_TRTL },
    { "cryptonight_gpu",         nullptr, xmrig::CRYPTONIGHT,       xmrig::VARIANT_GPU  },
};
#endif


static const char *variants[] = {
    "0",
    "1",
    "tube",
    "xtl",
    "msr",
    "xhv",
    "xao",
    "rto",
    "2",
    "half",
    "trtl",
    "gpu",
    "wow",
    "r",
};


static_assert(xmrig::VARIANT_MAX == ARRAY_SIZE(variants), "variants size mismatch");


bool xmrig::Algorithm::isValid() const
{
    if (m_algo == INVALID_ALGO) {
        return false;
    }

    for (size_t i = 0; i < ARRAY_SIZE(algorithms); i++) {
        if (algorithms[i].algo == m_algo && algorithms[i].variant == m_variant) {
            return true;
        }
    }

    return false;
}


const char *xmrig::Algorithm::variantName() const
{
    if (m_variant == VARIANT_AUTO) {
        return "auto";
    }

    return variants[m_variant];
}


void xmrig::Algorithm::parseAlgorithm(const char *algo)
{
    m_algo    = INVALID_ALGO;
    m_variant = VARIANT_AUTO;

    assert(algo != nullptr);
    if (algo == nullptr || strlen(algo) < 1) {
        return;
    }

    if (*algo == '!') {
        m_flags |= Forced;

        return parseAlgorithm(algo + 1);
    }

    for (size_t i = 0; i < ARRAY_SIZE(algorithms); i++) {
        if ((strcasecmp(algo, algorithms[i].name) == 0) || (strcasecmp(algo, algorithms[i].shortName) == 0)) {
            m_algo    = algorithms[i].algo;
            m_variant = algorithms[i].variant;
            break;
        }
    }

    if (m_algo == INVALID_ALGO) {
        assert(false);
    }
}


void xmrig::Algorithm::parseVariant(const char *variant)
{
    m_variant = VARIANT_AUTO;

    if (variant == nullptr || strlen(variant) < 1) {
        return;
    }

    if (*variant == '!') {
        m_flags |= Forced;

        return parseVariant(variant + 1);
    }

    for (size_t i = 0; i < ARRAY_SIZE(variants); i++) {
        if (strcasecmp(variant, variants[i]) == 0) {
            m_variant = static_cast<Variant>(i);
            return;
        }
    }

    if (strcasecmp(variant, "xtlv9") == 0) {
        m_variant = VARIANT_HALF;
    }
}


void xmrig::Algorithm::parseVariant(int variant)
{
    assert(variant >= -1 && variant <= 2);

    switch (variant) {
    case -1:
    case 0:
    case 1:
        m_variant = static_cast<Variant>(variant);
        break;

    case 2:
        m_variant = VARIANT_2;
        break;

    default:
        break;
    }
}


void xmrig::Algorithm::setAlgo(Algo algo)
{
    m_algo = algo;

    if (m_algo == CRYPTONIGHT_PICO && m_variant == VARIANT_AUTO) {
        m_variant = xmrig::VARIANT_TRTL;
    }
}


#ifdef XMRIG_PROXY_PROJECT
void xmrig::Algorithm::parseXmrStakAlgorithm(const char *algo)
{
    m_algo    = INVALID_ALGO;
    m_variant = VARIANT_AUTO;

    assert(algo != nullptr);
    if (algo == nullptr) {
        return;
    }

    for (size_t i = 0; i < ARRAY_SIZE(xmrStakAlgorithms); i++) {
        if (strcasecmp(algo, xmrStakAlgorithms[i].name) == 0) {
            m_algo    = xmrStakAlgorithms[i].algo;
            m_variant = xmrStakAlgorithms[i].variant;
            break;
        }
    }

    if (m_algo == INVALID_ALGO) {
        assert(false);
    }
}
#endif


const char *xmrig::Algorithm::name(bool shortName) const
{
    for (size_t i = 0; i < ARRAY_SIZE(algorithms); i++) {
        if (algorithms[i].algo == m_algo && algorithms[i].variant == m_variant) {
            return shortName ? algorithms[i].shortName : algorithms[i].name;
        }
    }

    return "invalid";
}


// returns string name of the PerfAlgo
const char *xmrig::Algorithm::perfAlgoName(const xmrig::PerfAlgo pa) {
    static const char* perf_algo_names[xmrig::PerfAlgo::PA_MAX] = {
        "cn",
        "cn/2",
        "cn/half",
        "cn/gpu",
        "cn/r",
        "cn-lite",
        "cn-heavy",
        "cn-pico",
    };
    return perf_algo_names[pa];
}

// constructs Algorithm from PerfAlgo
xmrig::Algorithm::Algorithm(const xmrig::PerfAlgo pa) {
    switch (pa) {
       case PA_CN:
           m_algo    = xmrig::CRYPTONIGHT;
           m_variant = xmrig::VARIANT_1;
           break;
       case PA_CN2:
           m_algo    = xmrig::CRYPTONIGHT;
           m_variant = xmrig::VARIANT_2;
           break;
       case PA_CN_HALF:
           m_algo    = xmrig::CRYPTONIGHT;
           m_variant = xmrig::VARIANT_HALF;
           break;
       case PA_CN_GPU:
           m_algo    = xmrig::CRYPTONIGHT;
           m_variant = xmrig::VARIANT_GPU;
           break;
       case PA_CN_R:
           m_algo    = xmrig::CRYPTONIGHT;
           m_variant = xmrig::VARIANT_4;
           break;
       case PA_CN_LITE:
           m_algo    = xmrig::CRYPTONIGHT_LITE;
           m_variant = xmrig::VARIANT_1;
           break;
       case PA_CN_HEAVY:
           m_algo    = xmrig::CRYPTONIGHT_HEAVY;
           m_variant = xmrig::VARIANT_0;
           break;
       case PA_CN_PICO:
           m_algo    = xmrig::CRYPTONIGHT_PICO;
           m_variant = xmrig::VARIANT_TRTL;
           break;
       default:
           m_algo    = xmrig::INVALID_ALGO;
           m_variant = xmrig::VARIANT_AUTO;
    }
}

// returns PerfAlgo that corresponds to current Algorithm
xmrig::PerfAlgo xmrig::Algorithm::perf_algo() const {
    switch (m_algo) {
       case CRYPTONIGHT:
           switch (m_variant) {
               case VARIANT_2:    return PA_CN2;
               case VARIANT_HALF: return PA_CN_HALF;
               case VARIANT_GPU:  return PA_CN_GPU;
               case VARIANT_WOW:  return PA_CN_R;
               case VARIANT_4:    return PA_CN_R;
               default:           return PA_CN;
           }
       case CRYPTONIGHT_LITE:  return PA_CN_LITE;
       case CRYPTONIGHT_HEAVY: return PA_CN_HEAVY;
       case CRYPTONIGHT_PICO:  return PA_CN_PICO;
       default: return PA_INVALID;
    }
}
