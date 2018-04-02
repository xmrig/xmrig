/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
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

#ifndef __XMRIG_H__
#define __XMRIG_H__


namespace xmrig
{


enum Algo {
    CRYPTONIGHT,       /* CryptoNight (Monero) */
    CRYPTONIGHT_LITE,  /* CryptoNight-Lite (AEON) */
    CRYPTONIGHT_HEAVY, /* CryptoNight-Heavy (SUMO) */
};


//--av=1 For CPUs with hardware AES.
//--av=2 Lower power mode (double hash) of 1.
//--av=3 Software AES implementation.
//--av=4 Lower power mode (double hash) of 3.
enum AlgoVariant {
    AV_AUTO,        // --av=0 Automatic mode.
    AV_SINGLE,      // --av=1  Single hash mode
    AV_DOUBLE,      // --av=2  Double hash mode
    AV_SINGLE_SOFT, // --av=3  Single hash mode (Software AES)
    AV_DOUBLE_SOFT, // --av=4  Double hash mode (Software AES)
    AV_TRIPLE,      // --av=5  Triple hash mode
    AV_QUAD,        // --av=6  Quard hash mode
    AV_PENTA,       // --av=7  Penta hash mode
    AV_TRIPLE_SOFT, // --av=8  Triple hash mode (Software AES)
    AV_QUAD_SOFT,   // --av=9  Quard hash mode  (Software AES)
    AV_PENTA_SOFT,  // --av=10 Penta hash mode  (Software AES)
    AV_MAX
};


enum Variant {
    VARIANT_AUTO = -1, // Autodetect
    VARIANT_NONE = 0,  // Original CryptoNight
    VARIANT_V1   = 1   // Monero v7 PoW
};


} /* namespace xmrig */


#endif /* __XMRIG_H__ */
