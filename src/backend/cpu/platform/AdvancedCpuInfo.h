/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <support@xmrig.com>
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

#ifndef XMRIG_ADVANCEDCPUINFO_H
#define XMRIG_ADVANCEDCPUINFO_H


#include "backend/cpu/platform/BasicCpuInfo.h"


namespace xmrig {


class AdvancedCpuInfo : public BasicCpuInfo
{
public:
    AdvancedCpuInfo();

protected:
    CpuThreads threads(const Algorithm &algorithm, uint32_t limit) const override;

    inline const char *backend() const override     { return m_backend; }
    inline size_t cores() const override            { return m_cores; }
    inline size_t L2() const override               { return m_L2; }
    inline size_t L3() const override               { return m_L3; }
    inline size_t packages() const override         { return m_packages; }
    inline size_t threads() const override          { return m_threads; }

private:
    bool m_L2_exclusive   = false;
    char m_backend[32]{};
    size_t m_cores        = 0;
    size_t m_L2           = 0;
    size_t m_L3           = 0;
    size_t m_packages     = 1;
};


} /* namespace xmrig */


#endif /* XMRIG_ADVANCEDCPUINFO_H */
