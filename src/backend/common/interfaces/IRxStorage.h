/* XMRig
 * Copyright (c) 2018-2019 tevador     <tevador@gmail.com>
 * Copyright (c) 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_IRXSTORAGE_H
#define XMRIG_IRXSTORAGE_H


#include "base/tools/Object.h"
#include "crypto/common/HugePagesInfo.h"
#include "crypto/rx/RxConfig.h"


#include <cstdint>
#include <utility>


namespace xmrig {


class Job;
class RxDataset;
class RxSeed;


class IRxStorage
{
public:
    XMRIG_DISABLE_COPY_MOVE(IRxStorage)

    IRxStorage()            = default;
    virtual ~IRxStorage()   = default;

    virtual bool isAllocated() const                                                                                            = 0;
    virtual HugePagesInfo hugePages() const                                                                                     = 0;
    virtual RxDataset *dataset(const Job &job, uint32_t nodeId) const                                                           = 0;
    virtual void init(const RxSeed &seed, uint32_t threads, bool hugePages, bool oneGbPages, RxConfig::Mode mode, int priority) = 0;
};


} /* namespace xmrig */


#endif // XMRIG_IRXSTORAGE_H
