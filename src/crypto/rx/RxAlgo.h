/* XMRig
 * Copyright (c) 2018-2019 tevador     <tevador@gmail.com>
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

#ifndef XMRIG_RX_ALGO_H
#define XMRIG_RX_ALGO_H


#include <cstddef>
#include <cstdint>


#include "base/crypto/Algorithm.h"


struct RandomX_ConfigurationBase;


namespace xmrig
{


class RxAlgo
{
public:
    static Algorithm::Id apply(Algorithm::Id algorithm);
    static const RandomX_ConfigurationBase *base(Algorithm::Id algorithm);
    static uint32_t programCount(Algorithm::Id algorithm);
    static uint32_t programIterations(Algorithm::Id algorithm);
    static uint32_t programSize(Algorithm::Id algorithm);
    static uint32_t version(Algorithm::Id algorithm);

    static inline Algorithm::Id id(Algorithm::Id algorithm)
    {
        if (algorithm == Algorithm::RX_SFX) {
            return Algorithm::RX_0;
        }

        return algorithm;
    }
};


} /* namespace xmrig */


#endif /* XMRIG_RX_ALGO_H */
