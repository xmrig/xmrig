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

#ifndef XMRIG_ALGORITHM_H
#define XMRIG_ALGORITHM_H


#include <vector>


namespace xmrig {


class Algorithm
{
public:
    enum Id : int {
        INVALID = -1,
        CN_0,          // "cn/0"             Original CryptoNight
        CN_1,          // "cn/1"             CryptoNight variant 1 also known as Monero7 and CryptoNightV7
        CN_2,          // "cn/2"             CryptoNight variant 2
        CN_R,          // "cn/r"             CryptoNightR (Monero's variant 4)
        CN_WOW,        // "cn/wow"           CryptoNightR (Wownero)
        CN_FAST,       // "cn/fast"          CryptoNight variant 1 with half iterations
        CN_HALF,       // "cn/half"          CryptoNight variant 2 with half iterations (Masari/Stellite)
        CN_XAO,        // "cn/xao"           Modified CryptoNight variant 0 (Alloy only)
        CN_RTO,        // "cn/rto"           Modified CryptoNight variant 1 (Arto only)
        CN_RWZ,        // "cn/rwz"           CryptoNight variant 2 with 3/4 iterations and reversed shuffle operation (Graft)
        CN_ZLS,        // "cn/zls"           CryptoNight variant 2 with 3/4 iterations (Zelerius)
        CN_DOUBLE,     // "cn/double"        CryptoNight variant 2 with double iterations (X-CASH)
#       ifdef XMRIG_ALGO_CN_GPU
        CN_GPU,        // "cn/gpu"           CryptoNight-GPU (Ryo)
#       endif
#       ifdef XMRIG_ALGO_CN_LITE
        CN_LITE_0,     // "cn-lite/0"        CryptoNight-Lite (1 MB) variant 0
        CN_LITE_1,     // "cn-lite/1"        CryptoNight-Lite (1 MB) variant 1
#       endif
#       ifdef XMRIG_ALGO_CN_HEAVY
        CN_HEAVY_0,    // "cn-heavy/0"       CryptoNight-Heavy (4 MB)
        CN_HEAVY_TUBE, // "cn-heavy/tube"    Modified CryptoNight-Heavy (TUBE only)
        CN_HEAVY_XHV,  // "cn-heavy/xhv"     Modified CryptoNight-Heavy (Haven Protocol only)
#       endif
#       ifdef XMRIG_ALGO_CN_PICO
        CN_PICO_0,     // "cn-pico"          CryptoNight Turtle (TRTL)
#       endif
        MAX
    };

    enum Family : int {
        UNKNOWN,
        CN,
        CN_LITE,
        CN_HEAVY,
        CN_PICO
    };

    inline Algorithm()                                     {}
    inline Algorithm(const char *algo) : m_id(parse(algo)) {}
    inline Algorithm(Id id) : m_id(id)                     {}

    inline bool isEqual(const Algorithm &other) const { return m_id == other.m_id; }
    inline bool isValid() const                       { return m_id != INVALID; }
    inline const char *name() const                   { return name(false); }
    inline const char *shortName() const              { return name(true); }
    inline Family family() const                      { return family(m_id); }
    inline Id id() const                              { return m_id; }

    inline bool operator!=(const Algorithm &other) const  { return !isEqual(other); }
    inline bool operator==(const Algorithm &other) const  { return isEqual(other); }
    inline operator Algorithm::Id() const                 { return m_id; }

    static Family family(Id id);
    static Id parse(const char *name);

private:
    const char *name(bool shortName) const;

    Id m_id = INVALID;
};


typedef std::vector<xmrig::Algorithm> Algorithms;


} /* namespace xmrig */


#endif /* XMRIG_ALGORITHM_H */
