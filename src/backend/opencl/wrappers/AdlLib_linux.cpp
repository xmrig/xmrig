/* XMRig
 * Copyright 2008-2018 Advanced Micro Devices, Inc.
 * Copyright 2018-2020 SChernykh                    <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig                        <https://github.com/xmrig>, <support@xmrig.com>
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


#include "backend/opencl/wrappers/AdlLib.h"
#include "backend/opencl/wrappers/OclDevice.h"


#include <fstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>


namespace xmrig {


bool AdlLib::m_initialized          = false;
bool AdlLib::m_ready                = false;
static const std::string kPrefix    = "/sys/bus/pci/drivers/amdgpu/";


static inline bool sysfs_is_file(const char *path)
{
    struct stat sb;

    return stat(path, &sb) == 0 && ((sb.st_mode & S_IFMT) == S_IFREG);
}


static inline std::string sysfs_prefix(const PciTopology &topology)
{
    std::string path = kPrefix + "0000:" + topology.toString().data() + "/hwmon/hwmon";

    if (sysfs_is_file((path + "2/name").c_str())) {
        return path + "2/";
    }

    if (sysfs_is_file((path + "3/name").c_str())) {
        return path + "3/";
    }

    return {};
}


uint32_t sysfs_read(const std::string &path)
{
    std::ifstream file(path);
    if (!file.is_open()) {
        return 0;
    }

    uint32_t value = 0;
    file >> value;

    return value;
}


} // namespace xmrig


bool xmrig::AdlLib::init()
{
    if (!m_initialized) {
        m_ready       = dlopen() && load();
        m_initialized = true;
    }

    return m_ready;
}


const char *xmrig::AdlLib::lastError() noexcept
{
    return nullptr;
}


void xmrig::AdlLib::close()
{
}


AdlHealth xmrig::AdlLib::health(const OclDevice &device)
{
    if (!isReady() || device.vendorId() != OCL_VENDOR_AMD) {
        return {};
    }

    const auto prefix = sysfs_prefix(device.topology());
    if (prefix.empty()) {
        return {};
    }

    AdlHealth health;
    health.clock        = sysfs_read(prefix + "freq1_input") / 1000000;
    health.memClock     = sysfs_read(prefix + "freq2_input") / 1000000;
    health.power        = sysfs_read(prefix + "power1_average") / 1000000;
    health.rpm          = sysfs_read(prefix + "fan1_input");
    health.temperature  = sysfs_read(prefix + "temp2_input") / 1000;

    return health;
}


bool xmrig::AdlLib::dlopen()
{
    struct stat sb;
    if (stat(kPrefix.c_str(), &sb) == -1) {
        return false;
    }

    return (sb.st_mode & S_IFMT) == S_IFDIR;
}


bool xmrig::AdlLib::load()
{
    return true;
}
