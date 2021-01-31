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

#include "crypto/common/LinuxMemory.h"
#include "3rdparty/fmt/core.h"
#include "crypto/common/VirtualMemory.h"


#include <algorithm>
#include <fstream>
#include <mutex>
#include <string>


namespace xmrig {


static std::mutex mutex;
constexpr size_t twoMiB = 2U * 1024U * 1024U;
constexpr size_t oneGiB = 1024U * 1024U * 1024U;


static inline std::string sysfs_path(uint32_t node, size_t hugePageSize, bool nr)
{
    return fmt::format("/sys/devices/system/node/node{}/hugepages/hugepages-{}kB/{}_hugepages", node, hugePageSize / 1024, nr ? "nr" : "free");
}


static inline bool write_nr_hugepages(uint32_t node, size_t hugePageSize, uint64_t count)   { return LinuxMemory::write(sysfs_path(node, hugePageSize, true).c_str(), count); }
static inline int64_t free_hugepages(uint32_t node, size_t hugePageSize)                    { return LinuxMemory::read(sysfs_path(node, hugePageSize, false).c_str()); }
static inline int64_t nr_hugepages(uint32_t node, size_t hugePageSize)                      { return LinuxMemory::read(sysfs_path(node, hugePageSize, true).c_str()); }


} // namespace xmrig


bool xmrig::LinuxMemory::reserve(size_t size, uint32_t node, size_t hugePageSize)
{
    std::lock_guard<std::mutex> lock(mutex);

    const size_t required = VirtualMemory::align(size, hugePageSize) / hugePageSize;

    const auto available = free_hugepages(node, hugePageSize);
    if (available < 0 || static_cast<size_t>(available) >= required) {
        return false;
    }

    return write_nr_hugepages(node, hugePageSize, std::max<size_t>(nr_hugepages(node, hugePageSize), 0) + (required - available));
}


bool xmrig::LinuxMemory::write(const char *path, uint64_t value)
{
    std::ofstream file(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        return false;
    }

    file << value;
    file.flush();

    return true;
}


int64_t xmrig::LinuxMemory::read(const char *path)
{
    std::ifstream file(path);
    if (!file.is_open()) {
        return -1;
    }

    uint64_t value = 0;
    file >> value;

    return value;
}
