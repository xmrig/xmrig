/* XMRig
 * Copyright (c) 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig       <support@xmrig.com>
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


#include <bitset>


namespace xmrig {


class BasicCpuInfo : public ICpuInfo
{
public:
    BasicCpuInfo();

protected:
    const char *backend() const override;
    CpuThreads threads(const Algorithm &algorithm, uint32_t limit) const override;
    rapidjson::Value toJSON(rapidjson::Document &doc) const override;

    inline Arch arch() const override                           { return m_arch; }
    inline Assembly::Id assembly() const override               { return m_assembly; }
    inline bool has(Flag flag) const override                   { return m_flags.test(flag); }
    inline bool hasAES() const override                         { return has(FLAG_AES); }
    inline bool hasAVX() const override                         { return has(FLAG_AVX); }
    inline bool hasAVX2() const override                        { return has(FLAG_AVX2); }
    inline bool hasBMI2() const override                        { return has(FLAG_BMI2); }
    inline bool hasCatL3() const override                       { return has(FLAG_CAT_L3); }
    inline bool hasOneGbPages() const override                  { return has(FLAG_PDPE1GB); }
    inline bool hasXOP() const override                         { return has(FLAG_XOP); }
    inline bool isVM() const override                           { return has(FLAG_VM); }
    inline bool jccErratum() const override                     { return m_jccErratum; }
    inline const char *brand() const override                   { return m_brand; }
    inline const std::vector<int32_t> &units() const override   { return m_units; }
    inline MsrMod msrMod() const override                       { return m_msrMod; }
    inline size_t cores() const override                        { return 0; }
    inline size_t L2() const override                           { return 0; }
    inline size_t L3() const override                           { return 0; }
    inline size_t nodes() const override                        { return 0; }
    inline size_t packages() const override                     { return 1; }
    inline size_t threads() const override                      { return m_threads; }
    inline Vendor vendor() const override                       { return m_vendor; }

protected:
    Arch m_arch             = ARCH_UNKNOWN;
    bool m_jccErratum       = false;
    char m_brand[64 + 6]{};
    size_t m_threads;
    std::vector<int32_t> m_units;
    Vendor m_vendor         = VENDOR_UNKNOWN;

private:
#   ifndef XMRIG_ARM
    uint32_t m_procInfo     = 0;
    uint32_t m_family       = 0;
    uint32_t m_model        = 0;
    uint32_t m_stepping     = 0;
#   endif

    Assembly m_assembly     = Assembly::NONE;
    MsrMod m_msrMod         = MSR_MOD_NONE;
    std::bitset<FLAG_MAX> m_flags;
};


} /* namespace xmrig */


#endif /* XMRIG_BASICCPUINFO_H */
