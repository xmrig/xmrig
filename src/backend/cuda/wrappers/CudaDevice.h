/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2019 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_CUDADEVICE_H
#define XMRIG_CUDADEVICE_H


#include "backend/common/misc/PciTopology.h"
#include "base/tools/String.h"


using nvid_ctx      = struct nvid_ctx;
using nvmlDevice_t  = struct nvmlDevice_st *;


namespace xmrig {


class Algorithm;
class CudaThreads;


class CudaDevice
{
public:
    CudaDevice() = delete;
    CudaDevice(const CudaDevice &other) = delete;
    CudaDevice(CudaDevice &&other) noexcept;
    CudaDevice(uint32_t index, int32_t bfactor, int32_t bsleep);
    ~CudaDevice();

    size_t freeMemSize() const;
    size_t globalMemSize() const;
    uint32_t clock() const;
    uint32_t computeCapability(bool major = true) const;
    uint32_t memoryClock() const;
    uint32_t smx() const;
    void generate(const Algorithm &algorithm, CudaThreads &threads) const;

    inline bool isValid() const                     { return m_ctx != nullptr; }
    inline const PciTopology &topology() const      { return m_topology; }
    inline const String &name() const               { return m_name; }
    inline uint32_t arch() const                    { return (computeCapability(true) * 10) + computeCapability(false); }
    inline uint32_t index() const                   { return m_index; }

#   ifdef XMRIG_FEATURE_NVML
    inline nvmlDevice_t nvmlDevice() const          { return m_nvmlDevice; }
    inline void setNvmlDevice(nvmlDevice_t device)  { m_nvmlDevice = device; }
#   endif

#   ifdef XMRIG_FEATURE_API
    void toJSON(rapidjson::Value &out, rapidjson::Document &doc) const;
#   endif

    CudaDevice &operator=(const CudaDevice &other)  = delete;
    CudaDevice &operator=(CudaDevice &&other)       = delete;

private:
    const uint32_t m_index          = 0;
    nvid_ctx *m_ctx                 = nullptr;
    PciTopology m_topology;
    String m_name;

#   ifdef XMRIG_FEATURE_NVML
    nvmlDevice_t m_nvmlDevice       = nullptr;
#   endif
};


} // namespace xmrig


#endif /* XMRIG_CUDADEVICE_H */
