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

#ifndef XMRIG_ADVANCEDCPUINFO_H
#define XMRIG_ADVANCEDCPUINFO_H


#include "common/interfaces/ICpuInfo.h"


namespace xmrig {


class AdvancedCpuInfo : public ICpuInfo
{
public:
    AdvancedCpuInfo();

protected:
    size_t optimalThreadsCount(size_t memSize, int maxCpuUsage) const override;

    inline Assembly assembly() const override       { return m_assembly; }
    inline bool hasAES() const override             { return m_aes; }
    inline bool hasAVX2() const override            { return m_avx2; }
    inline bool isSupported() const override        { return true; }
    inline const char *brand() const override       { return m_brand; }
    inline int32_t cores() const override           { return m_cores; }
    inline int32_t L2() const override              { return m_L2; }
    inline int32_t L3() const override              { return m_L3; }
    inline int32_t nodes() const override           { return -1; }
    inline int32_t sockets() const override         { return m_sockets; }
    inline int32_t threads() const override         { return m_threads; }

#   if defined(__x86_64__) || defined(_M_AMD64)
    inline bool isX64() const override { return true; }
#   else
    inline bool isX64() const override { return false; }
#   endif

private:
    Assembly m_assembly;
    bool m_aes;
    bool m_avx2;
    bool m_L2_exclusive;
    char m_brand[64];
    int32_t m_cores;
    int32_t m_L2;
    int32_t m_L3;
    int32_t m_sockets;
    int32_t m_threads;
};


} /* namespace xmrig */


#endif /* XMRIG_ADVANCEDCPUINFO_H */
