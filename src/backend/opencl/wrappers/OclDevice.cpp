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

#ifdef XMRIG_ALGO_CN_GPU
extern bool ocl_generic_cn_gpu_generator(const OclDevice &device, const Algorithm &algorithm, OclThreads &threads);
#endif


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
    ocl_generic_cn_generator,
#   ifdef XMRIG_ALGO_CN_GPU
    ocl_generic_cn_gpu_generator,
#   endif
};


static OclVendor getVendorId(const String &vendor)
{
    if (vendor.contains("Advanced Micro Devices") || vendor.contains("AMD")) {
        return OCL_VENDOR_AMD;
    }

    if (vendor.contains("NVIDIA")) {
        return  OCL_VENDOR_NVIDIA;
    }

    if (vendor.contains("Intel")) {
        return OCL_VENDOR_INTEL;
    }

    return OCL_VENDOR_UNKNOWN;
}


static OclDevice::Type getType(const String &name)
{
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

    if (name == "gfx804") {
        return OclDevice::Lexa;
    }

    if (name == "Baffin") {
        return OclDevice::Baffin;
    }

    if (name == "gfx803" || name.contains("polaris") || name == "Ellesmere") {
        return OclDevice::Polaris;
    }

    return OclDevice::Unknown;
}


} // namespace xmrig


xmrig::OclDevice::OclDevice(uint32_t index, cl_device_id id, cl_platform_id platform) :
    m_id(id),
    m_platform(platform),
    m_board(OclLib::getString(id, 0x4038 /* CL_DEVICE_BOARD_NAME_AMD */)),
    m_name(OclLib::getString(id, CL_DEVICE_NAME)),
    m_vendor(OclLib::getString(id, CL_DEVICE_VENDOR)),
    m_maxMemoryAlloc(OclLib::getUlong(id, CL_DEVICE_MAX_MEM_ALLOC_SIZE)),
    m_globalMemory(OclLib::getUlong(id, CL_DEVICE_GLOBAL_MEM_SIZE)),
    m_computeUnits(OclLib::getUint(id, CL_DEVICE_MAX_COMPUTE_UNITS, 1)),
    m_index(index)
{
    m_vendorId  = getVendorId(m_vendor);
    m_type      = getType(m_name);

    if (m_vendorId == OCL_VENDOR_AMD) {
        topology_amd topology;

        if (OclLib::getDeviceInfo(id, 0x4037 /* CL_DEVICE_TOPOLOGY_AMD */, sizeof(topology), &topology, nullptr) == CL_SUCCESS && topology.raw.type == 1) {
            m_topology = PciTopology(static_cast<uint32_t>(topology.pcie.bus), static_cast<uint32_t>(topology.pcie.device), static_cast<uint32_t>(topology.pcie.function));
        }
    }
    else if (m_vendorId == OCL_VENDOR_NVIDIA) {
        cl_uint bus = 0;
        if (OclLib::getDeviceInfo(id, 0x4008 /* CL_DEVICE_PCI_BUS_ID_NV */, sizeof (bus), &bus, nullptr) == CL_SUCCESS) {
            cl_uint slot  = OclLib::getUint(id, 0x4009 /* CL_DEVICE_PCI_SLOT_ID_NV */);
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
