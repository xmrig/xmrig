/* XMRig
 * Copyright (c) 2008-2018 Advanced Micro Devices, Inc.
 * Copyright (c) 2020      Patrick Bollinger            <https://github.com/pjbollinger>
 * Copyright (c) 2018-2021 SChernykh                    <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig                        <https://github.com/xmrig>, <support@xmrig.com>
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
#include "3rdparty/fmt/core.h"
#include "backend/opencl/wrappers/OclDevice.h"


#include <dirent.h>
#include <fstream>
#include <map>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>


namespace xmrig {


bool AdlLib::m_initialized          = false;
bool AdlLib::m_ready                = false;
static const std::string kPrefix    = "/sys/bus/pci/drivers/amdgpu/";
static std::map<PciTopology, std::string> hwmon_cache;


static inline bool sysfs_is_file(const char *path)
{
    struct stat sb;

    return stat(path, &sb) == 0 && ((sb.st_mode & S_IFMT) == S_IFREG);
}


static inline int dir_filter(const struct dirent *dirp)
{
    return strlen(dirp->d_name) > 5 ? 1 : 0;
}


static bool sysfs_is_amdgpu(const char *path, char *buf, const char *filename)
{
    strcpy(buf, filename);

    if (!sysfs_is_file(path)) {
        return false;
    }

    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }

    std::string name;
    std::getline(file, name);

    return name == "amdgpu";
}


static uint32_t sysfs_read(const char *path, char *buf, const char *filename)
{
    strcpy(buf, filename);

    std::ifstream file(path);
    if (!file.is_open()) {
        return 0;
    }

    uint32_t value = 0;
    file >> value;

    return value;
}


static size_t sysfs_prefix(char path[PATH_MAX], const PciTopology &topology)
{
    const auto it = hwmon_cache.find(topology);
    if (it != hwmon_cache.end()) {
        strcpy(path, it->second.data());

        return it->second.size();
    }

    char *base = fmt::format_to(path, "{}0000:{}/hwmon/", kPrefix, topology.toString());
    *base      = '\0';
    char *end  = nullptr;

    struct dirent **namelist;
    int n = scandir(path, &namelist, dir_filter, nullptr);
    if (n < 0) {
        return {};
    }

    while (n--) {
        if (!end) {
            char *tmp = fmt::format_to(base, "{}/", namelist[n]->d_name);
            end       = (sysfs_is_amdgpu(path, tmp, "name") && (sysfs_read(path, tmp, "temp1_input") || sysfs_read(path, tmp, "power1_average"))) ? tmp : nullptr;
        }

        free(namelist[n]);
    }

    free(namelist);

    if (end) {
        *end = '\0';
        hwmon_cache.insert({ topology, path });

        return end - path;
    }

    return 0;
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


// https://dri.freedesktop.org/docs/drm/gpu/amdgpu.html#gpu-power-thermal-controls-and-monitoring
AdlHealth xmrig::AdlLib::health(const OclDevice &device)
{
    if (!isReady() || device.vendorId() != OCL_VENDOR_AMD) {
        return {};
    }

    static char path[PATH_MAX]{};

    char *buf = path + sysfs_prefix(path, device.topology());
    if (buf == path) {
        return {};
    }

    AdlHealth health;
    health.clock        = sysfs_read(path, buf, "freq1_input") / 1000000;
    health.memClock     = sysfs_read(path, buf, "freq2_input") / 1000000;
    health.power        = sysfs_read(path, buf, "power1_average") / 1000000;
    health.rpm          = sysfs_read(path, buf, "fan1_input");
    health.temperature  = sysfs_read(path, buf, "temp2_input") / 1000;

    if (!health.temperature) {
        health.temperature = sysfs_read(path, buf, "temp1_input") / 1000;
    }

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
