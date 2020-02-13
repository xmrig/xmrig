/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_OCLDEVICE_H
#define XMRIG_OCLDEVICE_H


#include "backend/common/misc/PciTopology.h"
#include "backend/opencl/wrappers/OclVendor.h"
#include "base/tools/String.h"

#include <algorithm>
#include <vector>


using cl_device_id      = struct _cl_device_id *;
using cl_platform_id    = struct _cl_platform_id *;


namespace xmrig {


class Algorithm;
class OclThreads;


class OclDevice
{
public:
    enum Type {
        Unknown,
        Baffin,
        Polaris,
        Lexa,
        Vega_10,
        Vega_20,
        Raven,
        Navi_10,
        Navi_12,
        Navi_14
    };

    OclDevice() = delete;
    OclDevice(uint32_t index, cl_device_id id, cl_platform_id platform);

    String printableName() const;
    uint32_t clock() const;
    void generate(const Algorithm &algorithm, OclThreads &threads) const;

    inline bool isValid() const                 { return m_id != nullptr && m_platform != nullptr; }
    inline cl_device_id id() const              { return m_id; }
    inline const PciTopology &topology() const  { return m_topology; }
    inline const String &board() const          { return m_board.isNull() ? m_name : m_board; }
    inline const String &name() const           { return m_name; }
    inline const String &vendor() const         { return m_vendor; }
    inline OclVendor vendorId() const           { return m_vendorId; }
    inline Type type() const                    { return m_type; }
    inline uint32_t computeUnits() const        { return m_computeUnits; }
    inline size_t freeMemSize() const           { return std::min(maxMemAllocSize(), globalMemSize()); }
    inline size_t globalMemSize() const         { return m_globalMemory; }
    inline size_t maxMemAllocSize() const       { return m_maxMemoryAlloc; }
    inline uint32_t index() const               { return m_index; }

#   ifdef XMRIG_FEATURE_API
    void toJSON(rapidjson::Value &out, rapidjson::Document &doc) const;
#   endif

private:
    cl_device_id m_id               = nullptr;
    cl_platform_id m_platform       = nullptr;
    const String m_board;
    const String m_name;
    const String m_vendor;
    const size_t m_maxMemoryAlloc   = 0;
    const size_t m_globalMemory     = 0;
    const uint32_t m_computeUnits   = 1;
    const uint32_t m_index          = 0;
    OclVendor m_vendorId            = OCL_VENDOR_UNKNOWN;
    PciTopology m_topology;
    Type m_type                     = Unknown;
};


} // namespace xmrig


#endif /* XMRIG_OCLDEVICE_H */
