/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2018 XMRig       <support@xmrig.com>
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

#ifndef XMRIG_CPUINFO_H
#define XMRIG_CPUINFO_H


#include <stddef.h>
#include <stdint.h>


#include "common/xmrig.h"


namespace xmrig {


class ICpuInfo
{
public:
    virtual ~ICpuInfo() {}

    virtual bool hasAES() const                                               = 0;
    virtual bool isSupported() const                                          = 0;
    virtual bool isX64() const                                                = 0;
    virtual const char *brand() const                                         = 0;
    virtual int32_t cores() const                                             = 0;
    virtual int32_t L2() const                                                = 0;
    virtual int32_t L3() const                                                = 0;
    virtual int32_t nodes() const                                             = 0;
    virtual int32_t sockets() const                                           = 0;
    virtual int32_t threads() const                                           = 0;
    virtual size_t optimalThreadsCount(size_t memSize, int maxCpuUsage) const = 0;
    virtual xmrig::Assembly assembly() const                                  = 0;
};


} /* namespace xmrig */


#endif // XMRIG_CPUINFO_H
