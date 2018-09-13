/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
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

#ifndef __CRYPTONIGHT_CONSTANTS_H__
#define __CRYPTONIGHT_CONSTANTS_H__


#include <stdint.h>


#include "common/xmrig.h"


namespace xmrig
{

constexpr const size_t   CRYPTONIGHT_MEMORY       = 2 * 1024 * 1024;
constexpr const uint32_t CRYPTONIGHT_MASK         = 0x1FFFF0;
constexpr const uint32_t CRYPTONIGHT_ITER         = 0x80000;
constexpr const uint32_t CRYPTONIGHT_MSR_ITER     = 0x40000;
constexpr const uint32_t CRYPTONIGHT_XAO_ITER     = 0x100000;

constexpr const size_t   CRYPTONIGHT_LITE_MEMORY  = 1 * 1024 * 1024;
constexpr const uint32_t CRYPTONIGHT_LITE_MASK    = 0xFFFF0;
constexpr const uint32_t CRYPTONIGHT_LITE_ITER    = 0x40000;

constexpr const size_t   CRYPTONIGHT_HEAVY_MEMORY = 4 * 1024 * 1024;
constexpr const uint32_t CRYPTONIGHT_HEAVY_MASK   = 0x3FFFF0;
constexpr const uint32_t CRYPTONIGHT_HEAVY_ITER   = 0x40000;


template<Algo ALGO> inline constexpr size_t cn_select_memory()           { return 0; }
template<> inline constexpr size_t cn_select_memory<CRYPTONIGHT>()       { return CRYPTONIGHT_MEMORY; }
template<> inline constexpr size_t cn_select_memory<CRYPTONIGHT_LITE>()  { return CRYPTONIGHT_LITE_MEMORY; }
template<> inline constexpr size_t cn_select_memory<CRYPTONIGHT_HEAVY>() { return CRYPTONIGHT_HEAVY_MEMORY; }


inline size_t cn_select_memory(Algo algorithm)
{
    switch(algorithm)
    {
    case CRYPTONIGHT:
        return CRYPTONIGHT_MEMORY;

    case CRYPTONIGHT_LITE:
        return CRYPTONIGHT_LITE_MEMORY;

    case CRYPTONIGHT_HEAVY:
        return CRYPTONIGHT_HEAVY_MEMORY;

    default:
        break;
    }

    return 0;
}


template<Algo ALGO> inline constexpr uint32_t cn_select_mask()           { return 0; }
template<> inline constexpr uint32_t cn_select_mask<CRYPTONIGHT>()       { return CRYPTONIGHT_MASK; }
template<> inline constexpr uint32_t cn_select_mask<CRYPTONIGHT_LITE>()  { return CRYPTONIGHT_LITE_MASK; }
template<> inline constexpr uint32_t cn_select_mask<CRYPTONIGHT_HEAVY>() { return CRYPTONIGHT_HEAVY_MASK; }


inline uint32_t cn_select_mask(Algo algorithm)
{
    switch(algorithm)
    {
    case CRYPTONIGHT:
        return CRYPTONIGHT_MASK;

    case CRYPTONIGHT_LITE:
        return CRYPTONIGHT_LITE_MASK;

    case CRYPTONIGHT_HEAVY:
        return CRYPTONIGHT_HEAVY_MASK;

    default:
        break;
    }

    return 0;
}


template<Algo ALGO, Variant variant> inline constexpr uint32_t cn_select_iter()        { return 0; }
template<> inline constexpr uint32_t cn_select_iter<CRYPTONIGHT, VARIANT_0>()          { return CRYPTONIGHT_ITER; }
template<> inline constexpr uint32_t cn_select_iter<CRYPTONIGHT, VARIANT_1>()          { return CRYPTONIGHT_ITER; }
template<> inline constexpr uint32_t cn_select_iter<CRYPTONIGHT, VARIANT_2>()          { return CRYPTONIGHT_ITER; }
template<> inline constexpr uint32_t cn_select_iter<CRYPTONIGHT, VARIANT_XTL>()        { return CRYPTONIGHT_ITER; }
template<> inline constexpr uint32_t cn_select_iter<CRYPTONIGHT, VARIANT_MSR>()        { return CRYPTONIGHT_MSR_ITER; }
template<> inline constexpr uint32_t cn_select_iter<CRYPTONIGHT, VARIANT_XAO>()        { return CRYPTONIGHT_XAO_ITER; }
template<> inline constexpr uint32_t cn_select_iter<CRYPTONIGHT, VARIANT_RTO>()        { return CRYPTONIGHT_ITER; }
template<> inline constexpr uint32_t cn_select_iter<CRYPTONIGHT_LITE, VARIANT_0>()     { return CRYPTONIGHT_LITE_ITER; }
template<> inline constexpr uint32_t cn_select_iter<CRYPTONIGHT_LITE, VARIANT_1>()     { return CRYPTONIGHT_LITE_ITER; }
template<> inline constexpr uint32_t cn_select_iter<CRYPTONIGHT_HEAVY, VARIANT_0>()    { return CRYPTONIGHT_HEAVY_ITER; }
template<> inline constexpr uint32_t cn_select_iter<CRYPTONIGHT_HEAVY, VARIANT_XHV>()  { return CRYPTONIGHT_HEAVY_ITER; }
template<> inline constexpr uint32_t cn_select_iter<CRYPTONIGHT_HEAVY, VARIANT_TUBE>() { return CRYPTONIGHT_HEAVY_ITER; }


inline uint32_t cn_select_iter(Algo algorithm, Variant variant)
{
    switch (variant) {
    case VARIANT_MSR:
        return CRYPTONIGHT_MSR_ITER;

    case VARIANT_RTO:
        return CRYPTONIGHT_XAO_ITER;

    default:
        break;
    }

    switch(algorithm)
    {
    case CRYPTONIGHT:
        return CRYPTONIGHT_ITER;

    case CRYPTONIGHT_LITE:
        return CRYPTONIGHT_LITE_ITER;

    case CRYPTONIGHT_HEAVY:
        return CRYPTONIGHT_HEAVY_ITER;

    default:
        break;
    }

    return 0;
}


template<Variant variant> inline constexpr Variant cn_base_variant() { return VARIANT_0; }
template<> inline constexpr Variant cn_base_variant<VARIANT_0>()     { return VARIANT_0; }
template<> inline constexpr Variant cn_base_variant<VARIANT_1>()     { return VARIANT_1; }
template<> inline constexpr Variant cn_base_variant<VARIANT_TUBE>()  { return VARIANT_1; }
template<> inline constexpr Variant cn_base_variant<VARIANT_XTL>()   { return VARIANT_1; }
template<> inline constexpr Variant cn_base_variant<VARIANT_MSR>()   { return VARIANT_1; }
template<> inline constexpr Variant cn_base_variant<VARIANT_XHV>()   { return VARIANT_0; }
template<> inline constexpr Variant cn_base_variant<VARIANT_XAO>()   { return VARIANT_0; }
template<> inline constexpr Variant cn_base_variant<VARIANT_RTO>()   { return VARIANT_1; }
template<> inline constexpr Variant cn_base_variant<VARIANT_2>()     { return VARIANT_2; }


} /* namespace xmrig */


#endif /* __CRYPTONIGHT_CONSTANTS_H__ */
