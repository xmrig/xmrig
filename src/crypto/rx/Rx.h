/* xmlcore
 * Copyright (c) 2018-2019 tevador     <tevador@gmail.com>
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 xmlcore       <https://github.com/xmlcore>, <support@xmlcore.com>
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

#ifndef xmlcore_RX_H
#define xmlcore_RX_H


#include <cstdint>
#include <utility>
#include <vector>


#include "crypto/common/HugePagesInfo.h"


namespace xmlcore
{


class Algorithm;
class CpuConfig;
class CpuThread;
class IRxListener;
class Job;
class RxConfig;
class RxDataset;


class Rx
{
public:
    static HugePagesInfo hugePages();
    static RxDataset *dataset(const Job &job, uint32_t nodeId);
    static void destroy();
    static void init(IRxListener *listener);
    template<typename T> static bool init(const T &seed, const RxConfig &config, const CpuConfig &cpu);
    template<typename T> static bool isReady(const T &seed);

#   ifdef xmlcore_FEATURE_MSR
    static bool isMSR();
#   else
    static constexpr bool isMSR()   { return false; }
#   endif
};


} /* namespace xmlcore */


#endif /* xmlcore_RX_H */
