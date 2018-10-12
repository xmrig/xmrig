/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018      SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2018 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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
};


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
    if (algo == nullptr) {
        return;
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

    for (size_t i = 0; i < ARRAY_SIZE(variants); i++) {
        if (strcasecmp(variant, variants[i]) == 0) {
            m_variant = static_cast<Variant>(i);
            break;
        }
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
