/* XMRig
 * Copyright (c) 2018      Lee Clagett <https://github.com/vtnerd>
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

#include "base/crypto/Algorithm.h"
#include "3rdparty/rapidjson/document.h"
#include "base/tools/String.h"


#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <map>


#ifdef _MSC_VER
#   define strcasecmp  _stricmp
#endif


namespace xmrig {


const char *Algorithm::kINVALID         = "invalid";
const char *Algorithm::kCN              = "cn";
const char *Algorithm::kCN_0            = "cn/0";
const char *Algorithm::kCN_1            = "cn/1";
const char *Algorithm::kCN_2            = "cn/2";
const char *Algorithm::kCN_R            = "cn/r";
const char *Algorithm::kCN_FAST         = "cn/fast";
const char *Algorithm::kCN_HALF         = "cn/half";
const char *Algorithm::kCN_XAO          = "cn/xao";
const char *Algorithm::kCN_RTO          = "cn/rto";
const char *Algorithm::kCN_RWZ          = "cn/rwz";
const char *Algorithm::kCN_ZLS          = "cn/zls";
const char *Algorithm::kCN_DOUBLE       = "cn/double";
const char *Algorithm::kCN_CCX          = "cn/ccx";

#ifdef XMRIG_ALGO_CN_LITE
const char *Algorithm::kCN_LITE         = "cn-lite";
const char *Algorithm::kCN_LITE_0       = "cn-lite/0";
const char *Algorithm::kCN_LITE_1       = "cn-lite/1";
#endif

#ifdef XMRIG_ALGO_CN_HEAVY
const char *Algorithm::kCN_HEAVY        = "cn-heavy";
const char *Algorithm::kCN_HEAVY_0      = "cn-heavy/0";
const char *Algorithm::kCN_HEAVY_TUBE   = "cn-heavy/tube";
const char *Algorithm::kCN_HEAVY_XHV    = "cn-heavy/xhv";
#endif

#ifdef XMRIG_ALGO_CN_PICO
const char *Algorithm::kCN_PICO         = "cn-pico";
const char *Algorithm::kCN_PICO_0       = "cn-pico";
const char *Algorithm::kCN_PICO_TLO     = "cn-pico/tlo";
#endif

#ifdef XMRIG_ALGO_CN_FEMTO
const char *Algorithm::kCN_UPX2         = "cn/upx2";
#endif

#ifdef XMRIG_ALGO_CN_GPU
const char *Algorithm::kCN_GPU          = "cn/gpu";
#endif

#ifdef XMRIG_ALGO_RANDOMX
const char *Algorithm::kRX              = "rx";
const char *Algorithm::kRX_0            = "rx/0";
const char *Algorithm::kRX_WOW          = "rx/wow";
const char *Algorithm::kRX_ARQ          = "rx/arq";
const char *Algorithm::kRX_XEQ          = "rx/xeq";
const char *Algorithm::kRX_GRAFT        = "rx/graft";
const char *Algorithm::kRX_SFX          = "rx/sfx";
const char *Algorithm::kRX_KEVA         = "rx/keva";
#endif

#ifdef XMRIG_ALGO_ARGON2
const char *Algorithm::kAR2             = "argon2";
const char *Algorithm::kAR2_CHUKWA      = "argon2/chukwa";
const char *Algorithm::kAR2_CHUKWA_V2   = "argon2/chukwav2";
const char *Algorithm::kAR2_WRKZ        = "argon2/ninja";
#endif

#ifdef XMRIG_ALGO_KAWPOW
const char *Algorithm::kKAWPOW          = "kawpow";
const char *Algorithm::kKAWPOW_RVN      = "kawpow";
#endif

#ifdef XMRIG_ALGO_GHOSTRIDER
const char* Algorithm::kGHOSTRIDER      = "ghostrider";
const char* Algorithm::kGHOSTRIDER_RTM  = "ghostrider";
const char* Algorithm::kFLEX            = "flex";
const char* Algorithm::kFLEX_KCN        = "flex";
#endif

#ifdef XMRIG_ALGO_RANDOMX
const char *Algorithm::kRX_XLA          = "panthera";
#endif


#define ALGO_NAME(ALGO)         { Algorithm::ALGO, Algorithm::k##ALGO }
#define ALGO_ALIAS(ALGO, NAME)  { NAME, Algorithm::ALGO }
#define ALGO_ALIAS_AUTO(ALGO)   { Algorithm::k##ALGO, Algorithm::ALGO }


static const std::map<uint32_t, const char *> kAlgorithmNames = {
    ALGO_NAME(CN_0),
    ALGO_NAME(CN_1),
    ALGO_NAME(CN_2),
    ALGO_NAME(CN_R),
    ALGO_NAME(CN_FAST),
    ALGO_NAME(CN_HALF),
    ALGO_NAME(CN_XAO),
    ALGO_NAME(CN_RTO),
    ALGO_NAME(CN_RWZ),
    ALGO_NAME(CN_ZLS),
    ALGO_NAME(CN_DOUBLE),
    ALGO_NAME(CN_CCX),

#   ifdef XMRIG_ALGO_CN_LITE
    ALGO_NAME(CN_LITE_0),
    ALGO_NAME(CN_LITE_1),
#   endif

#   ifdef XMRIG_ALGO_CN_HEAVY
    ALGO_NAME(CN_HEAVY_0),
    ALGO_NAME(CN_HEAVY_TUBE),
    ALGO_NAME(CN_HEAVY_XHV),
#   endif

#   ifdef XMRIG_ALGO_CN_PICO
    ALGO_NAME(CN_PICO_0),
    ALGO_NAME(CN_PICO_TLO),
#   endif

#   ifdef XMRIG_ALGO_CN_FEMTO
    ALGO_NAME(CN_UPX2),
#   endif

#   ifdef XMRIG_ALGO_CN_GPU
    ALGO_NAME(CN_GPU),
#   endif

#   ifdef XMRIG_ALGO_RANDOMX
    ALGO_NAME(RX_0),
    ALGO_NAME(RX_WOW),
    ALGO_NAME(RX_ARQ),
    ALGO_NAME(RX_XEQ),
    ALGO_NAME(RX_GRAFT),
    ALGO_NAME(RX_SFX),
    ALGO_NAME(RX_KEVA),
#   endif

#   ifdef XMRIG_ALGO_ARGON2
    ALGO_NAME(AR2_CHUKWA),
    ALGO_NAME(AR2_CHUKWA_V2),
    ALGO_NAME(AR2_WRKZ),
#   endif

#   ifdef XMRIG_ALGO_KAWPOW
    ALGO_NAME(KAWPOW_RVN),
#   endif

#   ifdef XMRIG_ALGO_RANDOMX
    ALGO_NAME(RX_XLA),
#   endif

#   ifdef XMRIG_ALGO_GHOSTRIDER
    ALGO_NAME(GHOSTRIDER_RTM),
    ALGO_NAME(FLEX_KCN),
#   endif
};


struct aliasCompare
{
   inline bool operator()(const char *a, const char *b) const   { return strcasecmp(a, b) < 0; }
};


static const std::map<const char *, Algorithm::Id, aliasCompare> kAlgorithmAliases = {
    ALGO_ALIAS_AUTO(CN_0),          ALGO_ALIAS(CN_0,            "cryptonight/0"),
                                    ALGO_ALIAS(CN_0,            "cryptonight"),
                                    ALGO_ALIAS(CN_0,            "cn"),
    ALGO_ALIAS_AUTO(CN_1),          ALGO_ALIAS(CN_1,            "cryptonight/1"),
                                    ALGO_ALIAS(CN_1,            "cryptonight-monerov7"),
                                    ALGO_ALIAS(CN_1,            "cryptonight_v7"),
    ALGO_ALIAS_AUTO(CN_2),          ALGO_ALIAS(CN_2,            "cryptonight/2"),
                                    ALGO_ALIAS(CN_2,            "cryptonight-monerov8"),
                                    ALGO_ALIAS(CN_2,            "cryptonight_v8"),
    ALGO_ALIAS_AUTO(CN_FAST),       ALGO_ALIAS(CN_FAST,         "cryptonight/fast"),
                                    ALGO_ALIAS(CN_FAST,         "cryptonight/msr"),
                                    ALGO_ALIAS(CN_FAST,         "cn/msr"),
    ALGO_ALIAS_AUTO(CN_R),          ALGO_ALIAS(CN_R,            "cryptonight/r"),
                                    ALGO_ALIAS(CN_R,            "cryptonight_r"),
    ALGO_ALIAS_AUTO(CN_XAO),        ALGO_ALIAS(CN_XAO,          "cryptonight/xao"),
                                    ALGO_ALIAS(CN_XAO,          "cryptonight_alloy"),
    ALGO_ALIAS_AUTO(CN_HALF),       ALGO_ALIAS(CN_HALF,         "cryptonight/half"),
    ALGO_ALIAS_AUTO(CN_RTO),        ALGO_ALIAS(CN_RTO,          "cryptonight/rto"),
    ALGO_ALIAS_AUTO(CN_RWZ),        ALGO_ALIAS(CN_RWZ,          "cryptonight/rwz"),
    ALGO_ALIAS_AUTO(CN_ZLS),        ALGO_ALIAS(CN_ZLS,          "cryptonight/zls"),
    ALGO_ALIAS_AUTO(CN_DOUBLE),     ALGO_ALIAS(CN_DOUBLE,       "cryptonight/double"),
    ALGO_ALIAS_AUTO(CN_CCX),        ALGO_ALIAS(CN_CCX,          "cryptonight/ccx"),
                                    ALGO_ALIAS(CN_CCX,          "cryptonight/conceal"),
                                    ALGO_ALIAS(CN_CCX,          "cn/conceal"),

#   ifdef XMRIG_ALGO_CN_LITE
    ALGO_ALIAS_AUTO(CN_LITE_0),     ALGO_ALIAS(CN_LITE_0,       "cryptonight-lite/0"),
                                    ALGO_ALIAS(CN_LITE_0,       "cryptonight-lite"),
                                    ALGO_ALIAS(CN_LITE_0,       "cryptonight-light"),
                                    ALGO_ALIAS(CN_LITE_0,       "cn-lite"),
                                    ALGO_ALIAS(CN_LITE_0,       "cn-light"),
                                    ALGO_ALIAS(CN_LITE_0,       "cryptonight_lite"),
    ALGO_ALIAS_AUTO(CN_LITE_1),     ALGO_ALIAS(CN_LITE_1,       "cryptonight-lite/1"),
                                    ALGO_ALIAS(CN_LITE_1,       "cryptonight-aeonv7"),
                                    ALGO_ALIAS(CN_LITE_1,       "cryptonight_lite_v7"),
#   endif

#   ifdef XMRIG_ALGO_CN_HEAVY
    ALGO_ALIAS_AUTO(CN_HEAVY_0),    ALGO_ALIAS(CN_HEAVY_0,      "cryptonight-heavy/0"),
                                    ALGO_ALIAS(CN_HEAVY_0,      "cryptonight-heavy"),
                                    ALGO_ALIAS(CN_HEAVY_0,      "cn-heavy"),
                                    ALGO_ALIAS(CN_HEAVY_0,      "cryptonight_heavy"),
    ALGO_ALIAS_AUTO(CN_HEAVY_XHV),  ALGO_ALIAS(CN_HEAVY_XHV,    "cryptonight-heavy/xhv"),
                                    ALGO_ALIAS(CN_HEAVY_XHV,    "cryptonight_haven"),
    ALGO_ALIAS_AUTO(CN_HEAVY_TUBE), ALGO_ALIAS(CN_HEAVY_TUBE,   "cryptonight-heavy/tube"),
                                    ALGO_ALIAS(CN_HEAVY_TUBE,   "cryptonight-bittube2"),
#   endif

#   ifdef XMRIG_ALGO_CN_PICO
    ALGO_ALIAS_AUTO(CN_PICO_0),     ALGO_ALIAS(CN_PICO_0,       "cryptonight-pico"),
                                    ALGO_ALIAS(CN_PICO_0,       "cn-pico/0"),
                                    ALGO_ALIAS(CN_PICO_0,       "cryptonight-pico/trtl"),
                                    ALGO_ALIAS(CN_PICO_0,       "cn-pico/trtl"),
                                    ALGO_ALIAS(CN_PICO_0,       "cryptonight-turtle"),
                                    ALGO_ALIAS(CN_PICO_0,       "cn-trtl"),
                                    ALGO_ALIAS(CN_PICO_0,       "cryptonight-ultralite"),
                                    ALGO_ALIAS(CN_PICO_0,       "cn-ultralite"),
                                    ALGO_ALIAS(CN_PICO_0,       "cryptonight_turtle"),
                                    ALGO_ALIAS(CN_PICO_0,       "cn_turtle"),
    ALGO_ALIAS_AUTO(CN_PICO_TLO),   ALGO_ALIAS(CN_PICO_TLO,     "cryptonight-pico/tlo"),
                                    ALGO_ALIAS(CN_PICO_TLO,     "cryptonight/ultra"),
                                    ALGO_ALIAS(CN_PICO_TLO,     "cn/ultra"),
                                    ALGO_ALIAS(CN_PICO_TLO,     "cryptonight-talleo"),
                                    ALGO_ALIAS(CN_PICO_TLO,     "cn-talleo"),
                                    ALGO_ALIAS(CN_PICO_TLO,     "cryptonight_talleo"),
                                    ALGO_ALIAS(CN_PICO_TLO,     "cn_talleo"),
#   endif

#   ifdef XMRIG_ALGO_CN_FEMTO
    ALGO_ALIAS_AUTO(CN_UPX2),       ALGO_ALIAS(CN_UPX2,         "cryptonight/upx2"),
                                    ALGO_ALIAS(CN_UPX2,         "cn-extremelite/upx2"),
                                    ALGO_ALIAS(CN_UPX2,         "cryptonight-upx/2"),
#   endif

#   ifdef XMRIG_ALGO_CN_GPU
    ALGO_ALIAS_AUTO(CN_GPU),        ALGO_ALIAS(CN_GPU,          "cryptonight/gpu"),
                                    ALGO_ALIAS(CN_GPU,          "cryptonight_gpu"),
#   endif

#   ifdef XMRIG_ALGO_RANDOMX
    ALGO_ALIAS_AUTO(RX_0),          ALGO_ALIAS(RX_0,            "randomx/0"),
                                    ALGO_ALIAS(RX_0,            "randomx/test"),
                                    ALGO_ALIAS(RX_0,            "rx/test"),
                                    ALGO_ALIAS(RX_0,            "randomx"),
                                    ALGO_ALIAS(RX_0,            "rx"),
    ALGO_ALIAS_AUTO(RX_WOW),        ALGO_ALIAS(RX_WOW,          "randomx/wow"),
                                    ALGO_ALIAS(RX_WOW,          "randomwow"),
    ALGO_ALIAS_AUTO(RX_ARQ),        ALGO_ALIAS(RX_ARQ,          "randomx/arq"),
                                    ALGO_ALIAS(RX_ARQ,          "randomarq"),
    ALGO_ALIAS_AUTO(RX_XEQ),        ALGO_ALIAS(RX_XEQ,          "randomx/xeq"),
                                    ALGO_ALIAS(RX_XEQ,          "randomxeq"),
    ALGO_ALIAS_AUTO(RX_GRAFT),      ALGO_ALIAS(RX_GRAFT,        "randomx/graft"),
                                    ALGO_ALIAS(RX_GRAFT,        "randomgraft"),
    ALGO_ALIAS_AUTO(RX_SFX),        ALGO_ALIAS(RX_SFX,          "randomx/sfx"),
                                    ALGO_ALIAS(RX_SFX,          "randomsfx"),
    ALGO_ALIAS_AUTO(RX_KEVA),       ALGO_ALIAS(RX_KEVA,         "randomx/keva"),
                                    ALGO_ALIAS(RX_KEVA,         "randomkeva"),
#   endif

#   ifdef XMRIG_ALGO_ARGON2
    ALGO_ALIAS_AUTO(AR2_CHUKWA),    ALGO_ALIAS(AR2_CHUKWA,      "chukwa"),
    ALGO_ALIAS_AUTO(AR2_CHUKWA_V2), ALGO_ALIAS(AR2_CHUKWA,      "chukwav2"),
    ALGO_ALIAS_AUTO(AR2_WRKZ),      ALGO_ALIAS(AR2_WRKZ,        "argon2/wrkz"),
#   endif

#   ifdef XMRIG_ALGO_KAWPOW
    ALGO_ALIAS_AUTO(KAWPOW_RVN),    ALGO_ALIAS(KAWPOW_RVN,      "kawpow/rvn"),
#   endif

#   ifdef XMRIG_ALGO_RANDOMX
    ALGO_ALIAS_AUTO(RX_XLA),        ALGO_ALIAS(RX_XLA,          "Panthera"),
#   endif

#   ifdef XMRIG_ALGO_GHOSTRIDER
    ALGO_ALIAS_AUTO(GHOSTRIDER_RTM), ALGO_ALIAS(GHOSTRIDER_RTM, "ghostrider/rtm"),
                                     ALGO_ALIAS(GHOSTRIDER_RTM, "gr"),
    ALGO_ALIAS_AUTO(FLEX_KCN), ALGO_ALIAS(FLEX_KCN, "flex/kcn"),
                               ALGO_ALIAS(FLEX_KCN, "flex"),
#   endif
};


} /* namespace xmrig */


xmrig::Algorithm::Algorithm(const rapidjson::Value &value) :
    m_id(parse(value.GetString()))
{
}


xmrig::Algorithm::Algorithm(uint32_t id) :
    m_id(kAlgorithmNames.count(id) ? static_cast<Id>(id) : INVALID)
{
}


const char *xmrig::Algorithm::name() const
{
    if (!isValid()) {
        return kINVALID;
    }

    assert(kAlgorithmNames.count(m_id));
    const auto it = kAlgorithmNames.find(m_id);

    return it != kAlgorithmNames.end() ? it->second : kINVALID;
}


rapidjson::Value xmrig::Algorithm::toJSON() const
{
    using namespace rapidjson;

    return isValid() ? Value(StringRef(name())) : Value(kNullType);
}


rapidjson::Value xmrig::Algorithm::toJSON(rapidjson::Document &) const
{
    return toJSON();
}


xmrig::Algorithm::Id xmrig::Algorithm::parse(const char *name)
{
    if (name == nullptr || strlen(name) < 1) {
        return INVALID;
    }

    const auto it = kAlgorithmAliases.find(name);

    return it != kAlgorithmAliases.end() ? it->second : INVALID;
}


size_t xmrig::Algorithm::count()
{
    return kAlgorithmNames.size();
}


std::vector<xmrig::Algorithm> xmrig::Algorithm::all(const std::function<bool(const Algorithm &algo)> &filter)
{
    static const std::vector<Id> order = {
        CN_0, CN_1, CN_2, CN_R, CN_FAST, CN_HALF, CN_XAO, CN_RTO, CN_RWZ, CN_ZLS, CN_DOUBLE, CN_CCX,
        CN_LITE_0, CN_LITE_1,
        CN_HEAVY_0, CN_HEAVY_TUBE, CN_HEAVY_XHV,
        CN_PICO_0, CN_PICO_TLO,
        CN_UPX2,
        CN_GPU,
        RX_0, RX_WOW, RX_ARQ, RX_XEQ, RX_GRAFT, RX_SFX, RX_KEVA,
        RX_XLA,
        AR2_CHUKWA, AR2_CHUKWA_V2, AR2_WRKZ,
        KAWPOW_RVN,
        GHOSTRIDER_RTM,
        FLEX_KCN
    };

    Algorithms out;
    out.reserve(count());

    for (const Id algo : order) {
        if (kAlgorithmNames.count(algo) && (!filter || filter(algo))) {
            out.emplace_back(algo);
        }
    }

    return out;
}
