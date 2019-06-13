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

#ifndef XMRIG_XMRIG_H
#define XMRIG_XMRIG_H


namespace xmrig
{


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


enum AlgoVerify {
    VERIFY_HW_AES   = 1,
    VERIFY_SOFT_AES = 2
};


enum AesMode {
    AES_AUTO,
    AES_HW,
    AES_SOFT
};


enum OclVendor {
    OCL_VENDOR_UNKNOWN = -2,
    OCL_VENDOR_MANUAL  = -1,
    OCL_VENDOR_AMD     = 0,
    OCL_VENDOR_NVIDIA  = 1,
    OCL_VENDOR_INTEL   = 2
};


enum Assembly {
    ASM_NONE,
    ASM_AUTO,
    ASM_INTEL,
    ASM_RYZEN,
    ASM_BULLDOZER,
    ASM_MAX
};


} /* namespace xmrig */


#endif /* XMRIG_XMRIG_H */
