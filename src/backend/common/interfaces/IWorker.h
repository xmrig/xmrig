/* XMRig
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

#ifndef XMRIG_IWORKER_H
#define XMRIG_IWORKER_H


#include "base/tools/Object.h"


#include <cstdint>
#include <cstddef>


namespace xmrig {


class Job;
class VirtualMemory;


class IWorker
{
public:
    XMRIG_DISABLE_COPY_MOVE(IWorker)

    IWorker()           = default;
    virtual ~IWorker()  = default;

    virtual bool selfTest()                                                                         = 0;
    virtual const VirtualMemory *memory() const                                                     = 0;
    virtual size_t id() const                                                                       = 0;
    virtual size_t intensity() const                                                                = 0;
    virtual void hashrateData(uint64_t &hashCount, uint64_t &timeStamp, uint64_t &rawHashes) const  = 0;
    virtual void jobEarlyNotification(const Job &job)                                               = 0;
    virtual void start()                                                                            = 0;
};


} // namespace xmrig


#endif // XMRIG_IWORKER_H
