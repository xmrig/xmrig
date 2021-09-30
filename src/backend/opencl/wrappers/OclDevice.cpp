/* XMRig
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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
#include "3rdparty/rapidjson/document.h"
#include "backend/opencl/OclGenerator.h"
#include "backend/opencl/OclThreads.h"
#include "backend/opencl/wrappers/OclLib.h"
#include "base/io/log/Log.h"


#ifdef XMRIG_FEATURE_ADL
#   include "backend/opencl/wrappers/AdlLib.h"
#endif


#include <algorithm>


// NOLINTNEXTLINE(modernize-use-using)
typedef union
{
    struct { cl_uint type; cl_uint data[5]; } raw;
    struct { cl_uint type; cl_char unused[17]; cl_char bus; cl_char device; cl_char function; } pcie;
} topology_amd;


namespace xmrig {


#ifdef XMRIG_ALGO_RANDOMX
extern bool ocl_generic_rx_generator(const OclDevice &device, const Algorithm &algorithm, OclThreads &threads);
#endif

#ifdef XMRIG_ALGO_ASTROBWT
extern bool ocl_generic_astrobwt_generator(const OclDevice& device, const Algorithm& algorithm, OclThreads& threads);
#endif

#ifdef XMRIG_ALGO_KAWPOW
extern bool ocl_generic_kawpow_generator(const OclDevice& device, const Algorithm& algorithm, OclThreads& threads);
#endif

extern bool ocl_vega_cn_generator(const OclDevice &device, const Algorithm &algorithm, OclThreads &threads);
extern bool ocl_generic_cn_generator(const OclDevice &device, const Algorithm &algorithm, OclThreads &threads);


static ocl_gen_config_fun generators[] = {
#   ifdef XMRIG_ALGO_RANDOMX
    ocl_generic_rx_generator,
#   endif
#   ifdef XMRIG_ALGO_ASTROBWT
    ocl_generic_astrobwt_generator,
#   endif
#   ifdef XMRIG_ALGO_KAWPOW
    ocl_generic_kawpow_generator,
#   endif
    ocl_vega_cn_generator,
    ocl_generic_cn_generator
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

    if (extensions.contains("cl_APPLE_") || vendor.contains("Apple")) {
        return OCL_VENDOR_APPLE;
    }

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

    if (vendor.contains("Apple")) {
        return OCL_VENDOR_APPLE;
    }

    return OCL_VENDOR_UNKNOWN;
}


static OclDevice::Type getType(const String &name, const OclVendor platformVendorId)
{
    if (platformVendorId == OCL_VENDOR_APPLE) {
        // Apple Platform: uses product names, not gfx# or codenames
        if (name.contains("AMD Radeon")) {
            if (name.contains(" 450 ") ||
                name.contains(" 455 ") ||
                name.contains(" 460 ")) {
                return OclDevice::Baffin;
            }

            if (name.contains(" 555 ") || name.contains(" 555X ") ||
                name.contains(" 560 ") || name.contains(" 560X ") ||
                name.contains(" 570 ") || name.contains(" 570X ") ||
                name.contains(" 575 ") || name.contains(" 575X ")) {
                return OclDevice::Polaris;
            }

            if (name.contains(" 580 ") || name.contains(" 580X ")) {
                return OclDevice::Ellesmere;
            }

            if (name.contains(" Vega ")) {
                if (name.contains(" 48 ") ||
                    name.contains(" 56 ") ||
                    name.contains(" 64 ") ||
                    name.contains(" 64X ")) {
                    return OclDevice::Vega_10;
                }
                if (name.contains(" 16 ") ||
                    name.contains(" 20 ") ||
                    name.contains(" II ")) {
                    return OclDevice::Vega_20;
                }
            }

            if (name.contains(" 5700 ") || name.contains(" W5700X ")) {
                return OclDevice::Navi_10;
            }

            if (name.contains(" 5600 ") || name.contains(" 5600M ")) {
                return OclDevice::Navi_12;
            }

            if (name.contains(" 5300 ") || name.contains(" 5300M ") ||
                name.contains(" 5500 ") || name.contains(" 5500M ")) {
                return OclDevice::Navi_14;
            }

            if (name.contains(" W6800 ") || name.contains(" W6900X ")) {
                return OclDevice::Navi_21;
            }
        }
    }

    if (name == "gfx900" || name == "gfx901") {
        return OclDevice::Vega_10;
    }

    if (name == "gfx902" || name == "gfx903") {
        return OclDevice::Raven;
    }

    if (name == "gfx906" || name == "gfx907") {
        return OclDevice::Vega_20;
    }

    if (name == "gfx1010") {
        return OclDevice::Navi_10;
    }

    if (name == "gfx1011") {
        return OclDevice::Navi_12;
    }

    if (name == "gfx1012") {
        return OclDevice::Navi_14;
    }

    if (name == "gfx1030") {
        return OclDevice::Navi_21;
    }

    if (name == "gfx804") {
        return OclDevice::Lexa;
    }

    if (name == "Baffin") {
        return OclDevice::Baffin;
    }

    if (name.contains("Ellesmere")) {
        return OclDevice::Ellesmere;
    }

    if (name == "gfx803" || name.contains("polaris")) {
        return OclDevice::Polaris;
    }

    return OclDevice::Unknown;
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
    m_type      = getType(m_name, m_platformVendorId);

    if (m_extensions.contains("cl_amd_device_attribute_query")) {
        topology_amd topology;

        if (OclLib::getDeviceInfo(id, CL_DEVICE_TOPOLOGY_AMD, sizeof(topology), &topology, nullptr) == CL_SUCCESS && topology.raw.type == CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD) {
            m_topology = PciTopology(static_cast<uint32_t>(topology.pcie.bus), static_cast<uint32_t>(topology.pcie.device), static_cast<uint32_t>(topology.pcie.function));
        }
        m_board = OclLib::getString(id, CL_DEVICE_BOARD_NAME_AMD);
    }
    else if (m_extensions.contains("cl_nv_device_attribute_query")) {
        cl_uint bus = 0;
        if (OclLib::getDeviceInfo(id, CL_DEVICE_PCI_BUS_ID_NV, sizeof (bus), &bus, nullptr) == CL_SUCCESS) {
            cl_uint slot  = OclLib::getUint(id, CL_DEVICE_PCI_SLOT_ID_NV);
            m_topology = PciTopology(bus, (slot >> 3) & 0xff, slot & 7);
        }
    }
}


xmrig::String xmrig::OclDevice::printableName() const
{
    const size_t size = m_board.size() + m_name.size() + 64;
    char *buf         = new char[size]();

    if (m_board.isNull()) {
        snprintf(buf, size, GREEN_BOLD("%s"), m_name.data());
    }
    else {
        snprintf(buf, size, GREEN_BOLD("%s") " (" CYAN_BOLD("%s") ")", m_board.data(), m_name.data());
    }

    return buf;
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
