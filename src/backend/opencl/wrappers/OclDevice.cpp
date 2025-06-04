/* XMRig
 * Copyright (c) 2021      Spudz76     <https://github.com/Spudz76>
 * Copyright (c) 2018-2024 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2024 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#include "backend/opencl/wrappers/OclDevice.h"
#include "3rdparty/fmt/core.h"
#include "3rdparty/rapidjson/document.h"
#include "backend/opencl/OclGenerator.h"
#include "backend/opencl/OclThreads.h"
#include "backend/opencl/wrappers/OclLib.h"
#include "base/io/log/Log.h"


#ifdef XMRIG_FEATURE_ADL
#   include "backend/opencl/wrappers/AdlLib.h"
#endif


#include <algorithm>
#include <map>


namespace xmrig {


struct topology_amd {
    cl_uint type;
    cl_char unused[17];
    cl_char bus;
    cl_char device;
    cl_char function;
};


#ifdef XMRIG_ALGO_RANDOMX
extern bool ocl_generic_rx_generator(const OclDevice &device, const Algorithm &algorithm, OclThreads &threads);
#endif

#ifdef XMRIG_ALGO_KAWPOW
extern bool ocl_generic_kawpow_generator(const OclDevice& device, const Algorithm& algorithm, OclThreads& threads);
#endif

extern bool ocl_vega_cn_generator(const OclDevice &device, const Algorithm &algorithm, OclThreads &threads);
extern bool ocl_generic_cn_generator(const OclDevice &device, const Algorithm &algorithm, OclThreads &threads);

#ifdef XMRIG_ALGO_CN_GPU
extern bool ocl_generic_cn_gpu_generator(const OclDevice &device, const Algorithm &algorithm, OclThreads &threads);
#endif


static ocl_gen_config_fun generators[] = {
#   ifdef XMRIG_ALGO_RANDOMX
    ocl_generic_rx_generator,
#   endif
#   ifdef XMRIG_ALGO_KAWPOW
    ocl_generic_kawpow_generator,
#   endif
    ocl_vega_cn_generator,
    ocl_generic_cn_generator,
#   ifdef XMRIG_ALGO_CN_GPU
    ocl_generic_cn_gpu_generator,
#   endif
};


static OclVendor getPlatformVendorId(const String &vendor, const String &extensions)
{
    if (extensions.contains("cl_amd_") || vendor.contains("Advanced Micro Devices") || vendor.contains("AMD")) {
        return OCL_VENDOR_AMD;
    }

    if (extensions.contains("cl_nv_") || vendor.contains("NVIDIA")) {
        return OCL_VENDOR_NVIDIA;
    }

    if (extensions.contains("cl_intel_") || vendor.contains("Intel")) {
        return OCL_VENDOR_INTEL;
    }

#   ifdef XMRIG_OS_APPLE
    if (extensions.contains("cl_APPLE_") || vendor.contains("Apple")) {
        return OCL_VENDOR_APPLE;
    }
#   endif

    return OCL_VENDOR_UNKNOWN;
}


static OclVendor getVendorId(const String &vendor)
{
    if (vendor.contains("Advanced Micro Devices") || vendor.contains("AMD")) {
        return OCL_VENDOR_AMD;
    }

    if (vendor.contains("NVIDIA")) {
        return OCL_VENDOR_NVIDIA;
    }

    if (vendor.contains("Intel")) {
        return OCL_VENDOR_INTEL;
    }

#   ifdef XMRIG_OS_APPLE
    if (vendor.contains("Apple")) {
        return OCL_VENDOR_APPLE;
    }
#   endif

    return OCL_VENDOR_UNKNOWN;
}


} // namespace xmrig


xmrig::OclDevice::OclDevice(uint32_t index, cl_device_id id, cl_platform_id platform) :
    m_id(id),
    m_platform(platform),
    m_platformVendor(OclLib::getString(platform, CL_PLATFORM_VENDOR)),
    m_name(OclLib::getString(id, CL_DEVICE_NAME)),
    m_vendor(OclLib::getString(id, CL_DEVICE_VENDOR)),
    m_extensions(OclLib::getString(id, CL_DEVICE_EXTENSIONS)),
    m_maxMemoryAlloc(OclLib::getUlong(id, CL_DEVICE_MAX_MEM_ALLOC_SIZE)),
    m_globalMemory(OclLib::getUlong(id, CL_DEVICE_GLOBAL_MEM_SIZE)),
    m_computeUnits(OclLib::getUint(id, CL_DEVICE_MAX_COMPUTE_UNITS, 1)),
    m_index(index)
{
    m_vendorId  = getVendorId(m_vendor);
    m_platformVendorId = getPlatformVendorId(m_platformVendor, m_extensions);
    m_type      = getType(m_name);

    if (m_extensions.contains("cl_amd_device_attribute_query")) {
        topology_amd topology{};
        if (OclLib::getDeviceInfo(id, CL_DEVICE_TOPOLOGY_AMD, sizeof(topology), &topology) == CL_SUCCESS && topology.type == CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD) {
            m_topology = { topology.bus, topology.device, topology.function };
        }

        m_board = OclLib::getString(id, CL_DEVICE_BOARD_NAME_AMD);
    }
    else if (m_extensions.contains("cl_nv_device_attribute_query")) {
        cl_uint bus = 0;
        if (OclLib::getDeviceInfo(id, CL_DEVICE_PCI_BUS_ID_NV, sizeof(bus), &bus) == CL_SUCCESS) {
            cl_uint slot  = OclLib::getUint(id, CL_DEVICE_PCI_SLOT_ID_NV);
            m_topology = { bus, (slot >> 3) & 0xff, slot & 7 };
        }
    }
}


xmrig::String xmrig::OclDevice::printableName() const
{
    if (m_board.isNull()) {
        return fmt::format(GREEN_BOLD("{}"), m_name).c_str();
    }

    return fmt::format(GREEN_BOLD("{}") " (" CYAN_BOLD("{}") ")", m_board, m_name).c_str();
}


uint32_t xmrig::OclDevice::clock() const
{
    return OclLib::getUint(id(), CL_DEVICE_MAX_CLOCK_FREQUENCY);
}


void xmrig::OclDevice::generate(const Algorithm &algorithm, OclThreads &threads) const
{
    for (auto fn : generators) {
        if (fn(*this, algorithm, threads)) {
            return;
        }
    }
}


#ifdef XMRIG_FEATURE_API
void xmrig::OclDevice::toJSON(rapidjson::Value &out, rapidjson::Document &doc) const
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    out.AddMember("board",       board().toJSON(doc), allocator);
    out.AddMember("name",        name().toJSON(doc), allocator);
    out.AddMember("bus_id",      topology().toString().toJSON(doc), allocator);
    out.AddMember("cu",          computeUnits(), allocator);
    out.AddMember("global_mem",  static_cast<uint64_t>(globalMemSize()), allocator);

#   ifdef XMRIG_FEATURE_ADL
    if (AdlLib::isReady()) {
        auto data = AdlLib::health(*this);

        Value health(kObjectType);
        health.AddMember("temperature", data.temperature, allocator);
        health.AddMember("power",       data.power, allocator);
        health.AddMember("clock",       data.clock, allocator);
        health.AddMember("mem_clock",   data.memClock, allocator);
        health.AddMember("rpm",         data.rpm, allocator);

        out.AddMember("health", health, allocator);
    }
#   endif
}
#endif


#ifndef XMRIG_OS_APPLE
xmrig::OclDevice::Type xmrig::OclDevice::getType(const String &name)
{
    static std::map<const char *, OclDevice::Type> types = {
        { "gfx900",     Vega_10 },
        { "gfx901",     Vega_10 },
        { "gfx902",     Raven },
        { "gfx903",     Raven },
        { "gfx906",     Vega_20 },
        { "gfx907",     Vega_20 },
        { "gfx1010",    Navi_10 },
        { "gfx1011",    Navi_12 },
        { "gfx1012",    Navi_14 },
        { "gfx1030",    Navi_21 },
        { "gfx804",     Lexa },
        { "Baffin",     Baffin },
        { "Ellesmere",  Ellesmere },
        { "gfx803",     Polaris },
        { "polaris",    Polaris },
    };

    for (auto &kv : types) {
        if (name.contains(kv.first)) {
            return kv.second;
        }
    }

    return OclDevice::Unknown;
}
#endif
