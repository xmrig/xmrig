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


#include <algorithm>


#include "backend/opencl/OclThreads.h"
#include "backend/opencl/wrappers/OclDevice.h"
#include "backend/opencl/wrappers/OclLib.h"
#include "base/io/log/Log.h"
#include "crypto/cn/CnAlgo.h"
#include "crypto/common/Algorithm.h"
#include "rapidjson/document.h"


typedef union
{
    struct { cl_uint type; cl_uint data[5]; } raw;
    struct { cl_uint type; cl_char unused[17]; cl_char bus; cl_char device; cl_char function; } pcie;
} topology_amd;


namespace xmrig {


constexpr const size_t oneMiB = 1024u * 1024u;


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


static inline bool isCNv2(const Algorithm &algorithm)
{
    return algorithm.family() == Algorithm::CN && CnAlgo<>::base(algorithm) == Algorithm::CN_2;
}


} // namespace xmrig


xmrig::OclDevice::OclDevice(uint32_t index, cl_device_id id, cl_platform_id platform) :
    m_id(id),
    m_platform(platform),
    m_board(OclLib::getDeviceString(id, 0x4038 /* CL_DEVICE_BOARD_NAME_AMD */)),
    m_name(OclLib::getDeviceString(id, CL_DEVICE_NAME)),
    m_vendor(OclLib::getDeviceString(id, CL_DEVICE_VENDOR)),
    m_computeUnits(OclLib::getDeviceUint(id, CL_DEVICE_MAX_COMPUTE_UNITS, 1)),
    m_index(index)
{
    m_vendorId  = getVendorId(m_vendor);
    m_type      = getType(m_name);

    if (m_vendorId == OCL_VENDOR_AMD) {
        topology_amd topology;

        if (OclLib::getDeviceInfo(id, 0x4037 /* CL_DEVICE_TOPOLOGY_AMD */, sizeof(topology), &topology, nullptr) == CL_SUCCESS && topology.raw.type == 1) {
            m_topology    = true;
            m_pciTopology = PciTopology(static_cast<uint32_t>(topology.pcie.bus), static_cast<uint32_t>(topology.pcie.device), static_cast<uint32_t>(topology.pcie.function));
        }
    }
    else if (m_vendorId == OCL_VENDOR_NVIDIA) {
        cl_uint bus = 0;
        if (OclLib::getDeviceInfo(id, 0x4008 /* CL_DEVICE_PCI_BUS_ID_NV */, sizeof (bus), &bus, nullptr) == CL_SUCCESS) {
            m_topology    = true;
            cl_uint slot  = OclLib::getDeviceUint(id, 0x4009 /* CL_DEVICE_PCI_SLOT_ID_NV */);
            m_pciTopology = PciTopology(bus, (slot >> 3) & 0xff, slot & 7);
        }
    }
}


size_t xmrig::OclDevice::freeMem() const
{
    return std::min<size_t>(OclLib::getDeviceUlong(id(), CL_DEVICE_MAX_MEM_ALLOC_SIZE), globalMem());
}


size_t xmrig::OclDevice::globalMem() const
{
    return OclLib::getDeviceUlong(id(), CL_DEVICE_GLOBAL_MEM_SIZE);
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
    return OclLib::getDeviceUint(id(), CL_DEVICE_MAX_CLOCK_FREQUENCY);
}


void xmrig::OclDevice::generate(const Algorithm &algorithm, OclThreads &threads) const
{
    uint32_t intensity = getIntensity(algorithm);
    if (intensity == 0) {
        return;
    }

    const uint32_t worksize     = getWorksize(algorithm);
    const uint32_t stridedIndex = getStridedIndex(algorithm);
    const uint32_t memChunk     = getMemChunk(algorithm);
    const uint32_t threadCount  = ((globalMem() - intensity * 2 * algorithm.l3()) > 128 * oneMiB) ? 2 : 1;

    threads.add(OclThread(index(), intensity, worksize, stridedIndex, memChunk, threadCount, algorithm));
}


uint32_t xmrig::OclDevice::getIntensity(const Algorithm &algorithm) const
{
    if(m_type == Raven) {
        return 0;
    }

    const uint32_t maxIntensity = getPossibleIntensity(algorithm);

    if (m_type == Vega_10) {
        if (algorithm.family() == Algorithm::CN_HEAVY && m_computeUnits && maxIntensity > 976) {
            return 976;
        }

        return maxIntensity / m_computeUnits * m_computeUnits;
    }

    uint32_t intensity = (maxIntensity / (8 * m_computeUnits)) * m_computeUnits * 8;
    if (intensity == 0) {
        return 0;
    }

    if (m_vendorId == OCL_VENDOR_AMD && (m_type == Lexa || m_type == Baffin || m_computeUnits <= 16)) {
        intensity /= 2;

        if (algorithm.family() == Algorithm::CN_HEAVY) {
            intensity /= 2;
        }
    }

    return intensity;
}


uint32_t xmrig::OclDevice::getMaxThreads(const Algorithm &algorithm) const
{
    if (m_vendorId == OCL_VENDOR_NVIDIA && (m_name.contains("P100") || m_name.contains("V100"))) {
        return 40000u;
    }

    const uint32_t ratio = (algorithm.l3() <= oneMiB) ? 2u : 1u;
    if (m_type == Vega_10 || m_type == Vega_20) {
        if (computeUnits() == 56 && isCNv2(algorithm)) {
            return 1792u;
        }

        return ratio * 2024u;
    }

    return ratio * 1000u;
}


uint32_t xmrig::OclDevice::getMemChunk(const Algorithm &algorithm) const
{
    if ((m_type == Vega_10 || m_type == Vega_20) && (algorithm.family() == Algorithm::CN_PICO || isCNv2(algorithm))) {
        return 1;
    }

    return 2;
}


uint32_t xmrig::OclDevice::getPossibleIntensity(const Algorithm &algorithm) const
{
    const uint32_t maxThreads   = getMaxThreads(algorithm);
    const size_t minFreeMem     = (maxThreads == 40000u ? 512u : 128u) * oneMiB;
    const size_t availableMem   = freeMem() - minFreeMem;
    const size_t perThread      = algorithm.l3() + 224u;
    const auto maxIntensity     = static_cast<uint32_t>(availableMem / perThread);

    return std::min<uint32_t>(maxThreads, maxIntensity);
}


uint32_t xmrig::OclDevice::getStridedIndex(const Algorithm &algorithm) const
{
    if (m_vendorId == OCL_VENDOR_NVIDIA) {
        return 0;
    }

    if (algorithm.family() == Algorithm::CN_PICO || isCNv2(algorithm)) {
        return 2;
    }

    return 1;
}


uint32_t xmrig::OclDevice::getWorksize(const Algorithm &algorithm) const
{
    if (m_type == Vega_10 || m_type == Vega_20) {
        if (algorithm.family() == Algorithm::CN_PICO) {
            return 64;
        }

        if (isCNv2(algorithm)) {
            return 16;
        }
    }

    return 8;
}
