/* XMRig
 * Copyright (c) 2018-2025 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2025 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


static bool sysfs_write(const std::string &path, uint64_t value)
{
    std::ofstream file(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        return false;
    }

    file << value;
    file.flush();

    return true;
}


static int64_t sysfs_read(const std::string &path)
{
    std::ifstream file(path);
    if (!file.is_open()) {
        return -1;
    }

    uint64_t value = 0;
    file >> value;

    return value;
}


static std::string sysfs_path(uint32_t node, size_t hugePageSize, bool nr)
{
    return fmt::format("/sys/devices/system/node/node{}/hugepages/hugepages-{}kB/{}_hugepages", node, hugePageSize / 1024, nr ? "nr" : "free");
}


static std::string sysfs_path(size_t hugePageSize, bool nr)
{
    return fmt::format("/sys/kernel/mm/hugepages/hugepages-{}kB/{}_hugepages", hugePageSize / 1024, nr ? "nr" : "free");
}


static bool write_nr_hugepages(uint32_t node, size_t hugePageSize, uint64_t count)
{
    if (sysfs_write(sysfs_path(node, hugePageSize, true), count)) {
        return true;
    }

    return sysfs_write(sysfs_path(hugePageSize, true), count);
}


static int64_t sysfs_read_hugepages(uint32_t node, size_t hugePageSize, bool nr)
{
    const int64_t value = sysfs_read(sysfs_path(node, hugePageSize, nr));
    if (value >= 0) {
        return value;
    }

    return sysfs_read(sysfs_path(hugePageSize, nr));
}


static inline int64_t free_hugepages(uint32_t node, size_t hugePageSize)                    { return sysfs_read_hugepages(node, hugePageSize, false); }
static inline int64_t nr_hugepages(uint32_t node, size_t hugePageSize)                      { return sysfs_read_hugepages(node, hugePageSize, true); }


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
