/* XMRig
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

#ifndef XMRIG_ISTRATEGY_H
#define XMRIG_ISTRATEGY_H


#include <cstdint>


#include "base/tools/Object.h"


namespace xmrig {


class Algorithm;
class IClient;
class JobResult;
class ProxyUrl;


class IStrategy
{
public:
    XMRIG_DISABLE_COPY_MOVE(IStrategy)

    IStrategy()             = default;
    virtual ~IStrategy()    = default;

    virtual bool isActive() const                      = 0;
    virtual IClient *client() const                    = 0;
    virtual int64_t submit(const JobResult &result)    = 0;
    virtual void connect()                             = 0;
    virtual void resume()                              = 0;
    virtual void setAlgo(const Algorithm &algo)        = 0;
    virtual void setProxy(const ProxyUrl &proxy)       = 0;
    virtual void stop()                                = 0;
    virtual void tick(uint64_t now)                    = 0;
};


} /* namespace xmrig */


#endif // XMRIG_ISTRATEGY_H
