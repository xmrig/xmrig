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

#ifndef XMRIG_ALGORITHM_H
#define XMRIG_ALGORITHM_H


#include <functional>
#include <vector>


#include "3rdparty/rapidjson/fwd.h"


namespace xmrig {


class Algorithm
{
public:
    // Changes in following file may required if this enum changed:
    //
    // src/backend/opencl/cl/cn/algorithm.cl
    //
    // Id encoding:
    // 1 byte: family
    // 1 byte: L3 memory as power of 2 (if applicable).
    // 1 byte: L2 memory for RandomX algorithms as power of 2, base variant for CryptoNight algorithms or 0x00.
    // 1 byte: extra variant (coin) id.
    enum Id : uint32_t {
        INVALID         = 0,
        CN_0            = 0x63150000,   // "cn/0"             CryptoNight (original).
        CN_1            = 0x63150100,   // "cn/1"             CryptoNight variant 1 also known as Monero7 and CryptoNightV7.
        CN_2            = 0x63150200,   // "cn/2"             CryptoNight variant 2.
        CN_R            = 0x63150272,   // "cn/r"             CryptoNightR (Monero's variant 4).
        CN_FAST         = 0x63150166,   // "cn/fast"          CryptoNight variant 1 with half iterations.
        CN_HALF         = 0x63150268,   // "cn/half"          CryptoNight variant 2 with half iterations (Masari/Torque).
        CN_XAO          = 0x63150078,   // "cn/xao"           CryptoNight variant 0 (modified, Alloy only).
        CN_RTO          = 0x63150172,   // "cn/rto"           CryptoNight variant 1 (modified, Arto only).
        CN_RWZ          = 0x63150277,   // "cn/rwz"           CryptoNight variant 2 with 3/4 iterations and reversed shuffle operation (Graft).
        CN_ZLS          = 0x6315027a,   // "cn/zls"           CryptoNight variant 2 with 3/4 iterations (Zelerius).
        CN_DOUBLE       = 0x63150264,   // "cn/double"        CryptoNight variant 2 with double iterations (X-CASH).
        CN_CCX          = 0x63150063,   // "cn/ccx"           Conceal (CCX)
        CN_LITE_0       = 0x63140000,   // "cn-lite/0"        CryptoNight-Lite variant 0.
        CN_LITE_1       = 0x63140100,   // "cn-lite/1"        CryptoNight-Lite variant 1.
        CN_HEAVY_0      = 0x63160000,   // "cn-heavy/0"       CryptoNight-Heavy (4 MB).
        CN_HEAVY_TUBE   = 0x63160172,   // "cn-heavy/tube"    CryptoNight-Heavy (modified, TUBE only).
        CN_HEAVY_XHV    = 0x63160068,   // "cn-heavy/xhv"     CryptoNight-Heavy (modified, Haven Protocol only).
        CN_PICO_0       = 0x63120200,   // "cn-pico"          CryptoNight-Pico
        CN_PICO_TLO     = 0x63120274,   // "cn-pico/tlo"      CryptoNight-Pico (TLO)
        CN_UPX2         = 0x63110200,   // "cn/upx2"          Uplexa (UPX2)
        CN_GR_0         = 0x63130100,   // "cn/dark"          GhostRider
        CN_GR_1         = 0x63130101,   // "cn/dark-lite"     GhostRider
        CN_GR_2         = 0x63150102,   // "cn/fast"          GhostRider
        CN_GR_3         = 0x63140103,   // "cn/lite"          GhostRider
        CN_GR_4         = 0x63120104,   // "cn/turtle"        GhostRider
        CN_GR_5         = 0x63120105,   // "cn/turtle-lite"   GhostRider
        GHOSTRIDER_RTM  = 0x6c150000,   // "ghostrider"       GhostRider
        RX_0            = 0x72151200,   // "rx/0"             RandomX (reference configuration).
        RX_WOW          = 0x72141177,   // "rx/wow"           RandomWOW (Wownero).
        RX_ARQ          = 0x72121061,   // "rx/arq"           RandomARQ (Arqma).
        RX_GRAFT        = 0x72151267,   // "rx/graft"         RandomGRAFT (Graft).
        RX_SFX          = 0x72151273,   // "rx/sfx"           RandomSFX (Safex Cash).
        RX_YADA         = 0x72151279,   // "rx/yada"          RandomYada (YadaCoin).
        AR2_CHUKWA      = 0x61130000,   // "argon2/chukwa"    Argon2id (Chukwa).
        AR2_CHUKWA_V2   = 0x61140000,   // "argon2/chukwav2"  Argon2id (Chukwa v2).
        AR2_WRKZ        = 0x61120000,   // "argon2/wrkz"      Argon2id (WRKZ)
        KAWPOW_RVN      = 0x6b0f0000,   // "kawpow/rvn"       KawPow (RVN)
    };

    enum Family : uint32_t {
        UNKNOWN         = 0,
        CN_ANY          = 0x63000000,
        CN              = 0x63150000,
        CN_LITE         = 0x63140000,
        CN_HEAVY        = 0x63160000,
        CN_PICO         = 0x63120000,
        CN_FEMTO        = 0x63110000,
        RANDOM_X        = 0x72000000,
        ARGON2          = 0x61000000,
        KAWPOW          = 0x6b000000,
        GHOSTRIDER      = 0x6c000000
    };

    static const char *kINVALID;
    static const char *kCN;
    static const char *kCN_0;
    static const char *kCN_1;
    static const char *kCN_2;
    static const char *kCN_R;
    static const char *kCN_FAST;
    static const char *kCN_HALF;
    static const char *kCN_XAO;
    static const char *kCN_RTO;
    static const char *kCN_RWZ;
    static const char *kCN_ZLS;
    static const char *kCN_DOUBLE;
    static const char *kCN_CCX;

#   ifdef XMRIG_ALGO_CN_LITE
    static const char *kCN_LITE;
    static const char *kCN_LITE_0;
    static const char *kCN_LITE_1;
#   endif

#   ifdef XMRIG_ALGO_CN_HEAVY
    static const char *kCN_HEAVY;
    static const char *kCN_HEAVY_0;
    static const char *kCN_HEAVY_TUBE;
    static const char *kCN_HEAVY_XHV;
#   endif

#   ifdef XMRIG_ALGO_CN_PICO
    static const char *kCN_PICO;
    static const char *kCN_PICO_0;
    static const char *kCN_PICO_TLO;
#   endif

#   ifdef XMRIG_ALGO_CN_FEMTO
    static const char *kCN_UPX2;
#   endif

#   ifdef XMRIG_ALGO_RANDOMX
    static const char *kRX;
    static const char *kRX_0;
    static const char *kRX_WOW;
    static const char *kRX_ARQ;
    static const char *kRX_GRAFT;
    static const char *kRX_SFX;
    static const char *kRX_YADA;
#   endif

#   ifdef XMRIG_ALGO_ARGON2
    static const char *kAR2;
    static const char *kAR2_CHUKWA;
    static const char *kAR2_CHUKWA_V2;
    static const char *kAR2_WRKZ;
#   endif

#   ifdef XMRIG_ALGO_KAWPOW
    static const char *kKAWPOW;
    static const char *kKAWPOW_RVN;
#   endif

#   ifdef XMRIG_ALGO_GHOSTRIDER
    static const char* kGHOSTRIDER;
    static const char* kGHOSTRIDER_RTM;
#   endif

    inline Algorithm() = default;
    inline Algorithm(const char *algo) : m_id(parse(algo))  {}
    inline Algorithm(Id id) : m_id(id)                      {}
    Algorithm(const rapidjson::Value &value);
    Algorithm(uint32_t id);

    static inline constexpr bool isCN(Id id)                { return (id & 0xff000000) == CN_ANY; }
    static inline constexpr Id base(Id id)                  { return isCN(id) ? static_cast<Id>(CN_0 | (id & 0xff00)) : INVALID; }
    static inline constexpr size_t l2(Id id)                { return family(id) == RANDOM_X ? (1U << ((id >> 8) & 0xff)) : 0U; }
    static inline constexpr size_t l3(Id id)                { return 1ULL << ((id >> 16) & 0xff); }
    static inline constexpr uint32_t family(Id id)          { return id & (isCN(id) ? 0xffff0000 : 0xff000000); }

    inline bool isCN() const                                { return isCN(m_id); }
    inline bool isEqual(const Algorithm &other) const       { return m_id == other.m_id; }
    inline bool isValid() const                             { return m_id != INVALID && family() > UNKNOWN; }
    inline Id base() const                                  { return base(m_id); }
    inline Id id() const                                    { return m_id; }
    inline size_t l2() const                                { return l2(m_id); }
    inline uint32_t family() const                          { return family(m_id); }
    inline uint32_t minIntensity() const                    { return ((m_id == GHOSTRIDER_RTM) ? 8 : 1); };
    inline uint32_t maxIntensity() const                    { return isCN() ? 5 : ((m_id == GHOSTRIDER_RTM) ? 8 : 1); };

    inline size_t l3() const                                { return l3(m_id); }

    inline bool operator!=(Algorithm::Id id) const          { return m_id != id; }
    inline bool operator!=(const Algorithm &other) const    { return !isEqual(other); }
    inline bool operator==(Algorithm::Id id) const          { return m_id == id; }
    inline bool operator==(const Algorithm &other) const    { return isEqual(other); }
    inline operator Algorithm::Id() const                   { return m_id; }

    const char *name() const;
    rapidjson::Value toJSON() const;
    rapidjson::Value toJSON(rapidjson::Document &doc) const;

    static Id parse(const char *name);
    static size_t count();
    static std::vector<Algorithm> all(const std::function<bool(const Algorithm &algo)> &filter = nullptr);

private:
    Id m_id = INVALID;
};


using Algorithms = std::vector<Algorithm>;


} /* namespace xmrig */


#endif /* XMRIG_ALGORITHM_H */
