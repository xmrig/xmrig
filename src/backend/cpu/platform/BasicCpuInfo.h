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

#ifndef XMRIG_BASICCPUINFO_H
#define XMRIG_BASICCPUINFO_H


#include "backend/cpu/interfaces/ICpuInfo.h"


namespace xmrig {


class BasicCpuInfo : public ICpuInfo
{
public:
    BasicCpuInfo();

protected:
    const char *backend() const override;
    CpuThreads threads(const Algorithm &algorithm, uint32_t limit) const override;

    inline Assembly::Id assembly() const override   { return m_assembly; }
    inline bool hasAES() const override             { return m_aes; }
    inline bool hasAVX2() const override            { return m_avx2; }
    inline bool hasOneGbPages() const override      { return m_pdpe1gb; }
    inline const char *brand() const override       { return m_brand; }
    inline size_t cores() const override            { return 0; }
    inline size_t L2() const override               { return 0; }
    inline size_t L3() const override               { return 0; }
    inline size_t nodes() const override            { return 0; }
    inline size_t packages() const override         { return 1; }
    inline size_t threads() const override          { return m_threads; }
    inline Vendor vendor() const override           { return m_vendor; }

protected:
    char m_brand[64 + 6]{};
    size_t m_threads;

private:
    Assembly m_assembly     = Assembly::NONE;
    bool m_aes              = false;
    const bool m_avx2       = false;
    const bool m_pdpe1gb    = false;
    Vendor m_vendor         = VENDOR_UNKNOWN;
};


} /* namespace xmrig */


#endif /* XMRIG_BASICCPUINFO_H */
