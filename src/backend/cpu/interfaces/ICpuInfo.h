/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2019 XMRig       <support@xmrig.com>
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


#include "backend/cpu/CpuThreads.h"
#include "crypto/common/Assembly.h"
#include "crypto/common/Algorithm.h"


namespace xmrig {


class ICpuInfo
{
public:
    enum Vendor {
        VENDOR_UNKNOWN,
        VENDOR_INTEL,
        VENDOR_AMD
    };

    virtual ~ICpuInfo() = default;

#   if defined(__x86_64__) || defined(_M_AMD64) || defined (__arm64__) || defined (__aarch64__)
    inline constexpr static bool isX64() { return true; }
#   else
    inline constexpr static bool isX64() { return false; }
#   endif

    virtual Assembly::Id assembly() const                                           = 0;
    virtual bool hasAES() const                                                     = 0;
    virtual bool hasAVX2() const                                                    = 0;
    virtual bool hasOneGbPages() const                                              = 0;
    virtual const char *backend() const                                             = 0;
    virtual const char *brand() const                                               = 0;
    virtual CpuThreads threads(const Algorithm &algorithm, uint32_t limit) const    = 0;
    virtual size_t cores() const                                                    = 0;
    virtual size_t L2() const                                                       = 0;
    virtual size_t L3() const                                                       = 0;
    virtual size_t nodes() const                                                    = 0;
    virtual size_t packages() const                                                 = 0;
    virtual size_t threads() const                                                  = 0;
    virtual Vendor vendor() const                                                   = 0;
};


} /* namespace xmrig */


#endif // XMRIG_CPUINFO_H
