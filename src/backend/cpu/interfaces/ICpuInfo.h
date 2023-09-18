/* XMRig
 * Copyright (c) 2018-2023 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2023 XMRig       <support@xmrig.com>
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
#include "base/crypto/Algorithm.h"
#include "base/tools/Object.h"
#include "crypto/common/Assembly.h"


#ifdef XMRIG_FEATURE_HWLOC
using hwloc_const_bitmap_t  = const struct hwloc_bitmap_s *;
using hwloc_topology_t      = struct hwloc_topology *;
#endif


namespace xmrig {


class ICpuInfo
{
public:
    XMRIG_DISABLE_COPY_MOVE(ICpuInfo)

    enum Vendor : uint32_t {
        VENDOR_UNKNOWN,
        VENDOR_INTEL,
        VENDOR_AMD
    };

    enum Arch : uint32_t {
        ARCH_UNKNOWN,
        ARCH_ZEN,
        ARCH_ZEN_PLUS,
        ARCH_ZEN2,
        ARCH_ZEN3,
        ARCH_ZEN4
    };

    enum MsrMod : uint32_t {
        MSR_MOD_NONE,
        MSR_MOD_RYZEN_17H,
        MSR_MOD_RYZEN_19H,
        MSR_MOD_RYZEN_19H_ZEN4,
        MSR_MOD_INTEL,
        MSR_MOD_CUSTOM,
        MSR_MOD_MAX
    };

#   define MSR_NAMES_LIST "none", "ryzen_17h", "ryzen_19h", "ryzen_19h_zen4", "intel", "custom"

    enum Flag : uint32_t {
        FLAG_AES,
        FLAG_VAES,
        FLAG_AVX,
        FLAG_AVX2,
        FLAG_AVX512F,
        FLAG_BMI2,
        FLAG_OSXSAVE,
        FLAG_PDPE1GB,
        FLAG_SSE2,
        FLAG_SSSE3,
        FLAG_SSE41,
        FLAG_XOP,
        FLAG_POPCNT,
        FLAG_CAT_L3,
        FLAG_VM,
        FLAG_MAX
    };

    ICpuInfo()          = default;
    virtual ~ICpuInfo() = default;

#   if defined(__x86_64__) || defined(_M_AMD64) || defined (__arm64__) || defined (__aarch64__)
    inline constexpr static bool is64bit() { return true; }
#   else
    inline constexpr static bool is64bit() { return false; }
#   endif

    virtual Arch arch() const                                                       = 0;
    virtual Assembly::Id assembly() const                                           = 0;
    virtual bool has(Flag feature) const                                            = 0;
    virtual bool hasAES() const                                                     = 0;
    virtual bool hasVAES() const                                                    = 0;
    virtual bool hasAVX() const                                                     = 0;
    virtual bool hasAVX2() const                                                    = 0;
    virtual bool hasBMI2() const                                                    = 0;
    virtual bool hasCatL3() const                                                   = 0;
    virtual bool hasOneGbPages() const                                              = 0;
    virtual bool hasXOP() const                                                     = 0;
    virtual bool isVM() const                                                       = 0;
    virtual bool jccErratum() const                                                 = 0;
    virtual const char *backend() const                                             = 0;
    virtual const char *brand() const                                               = 0;
    virtual const std::vector<int32_t> &units() const                               = 0;
    virtual CpuThreads threads(const Algorithm &algorithm, uint32_t limit) const    = 0;
    virtual MsrMod msrMod() const                                                   = 0;
    virtual rapidjson::Value toJSON(rapidjson::Document &doc) const                 = 0;
    virtual size_t cores() const                                                    = 0;
    virtual size_t L2() const                                                       = 0;
    virtual size_t L3() const                                                       = 0;
    virtual size_t nodes() const                                                    = 0;
    virtual size_t packages() const                                                 = 0;
    virtual size_t threads() const                                                  = 0;
    virtual Vendor vendor() const                                                   = 0;
    virtual uint32_t model() const                                                  = 0;

#   ifdef XMRIG_FEATURE_HWLOC
    virtual bool membind(hwloc_const_bitmap_t nodeset)                              = 0;
    virtual const std::vector<uint32_t> &nodeset() const                            = 0;
    virtual hwloc_topology_t topology() const                                       = 0;
#   endif
};


} // namespace xmrig


#endif // XMRIG_CPUINFO_H
