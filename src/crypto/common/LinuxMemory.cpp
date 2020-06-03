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


//#include <iostream>

#include "crypto/common/LinuxMemory.h"
#include "base/io/log/Log.h"
#include "crypto/common/VirtualMemory.h"
#include "backend/cpu/Cpu.h"


#include <algorithm>
#include <fstream>
#include <string>
#include <mutex>


namespace xmrig {


static std::mutex mutex;
constexpr size_t twoMiB = 2U * 1024U * 1024U;
constexpr size_t oneGiB = 1024U * 1024U * 1024U;


static inline std::string sysfs_path(uint32_t node, bool oneGbPages, bool nr)
{
    return "/sys/devices/system/node/node" + std::to_string(node) + "/hugepages/hugepages-" + (oneGbPages ? "1048576" : "2048") + "kB/" + (nr ? "nr" : "free") + "_hugepages";
}


static inline bool write_nr_hugepages(uint32_t node, bool oneGbPages, uint64_t count)    { return LinuxMemory::write(sysfs_path(node, oneGbPages, true).c_str(), count); }
static inline int64_t free_hugepages(uint32_t node, bool oneGbPages)                     { return LinuxMemory::read(sysfs_path(node, oneGbPages, false).c_str()); }
static inline int64_t nr_hugepages(uint32_t node, bool oneGbPages)                       { return LinuxMemory::read(sysfs_path(node, oneGbPages, true).c_str()); }


} // namespace xmrig


bool xmrig::LinuxMemory::reserve(size_t size, uint32_t node, bool oneGbPages)
{
    std::lock_guard<std::mutex> lock(mutex);

    const size_t pageSize = oneGbPages ? oneGiB : twoMiB;
    const size_t required = VirtualMemory::align(size, pageSize) / pageSize;

    const auto available = free_hugepages(node, oneGbPages);
    if (available < 0 || static_cast<size_t>(available) >= required) {
        return false;
    }

    return write_nr_hugepages(node, oneGbPages, std::max<size_t>(nr_hugepages(node, oneGbPages), 0) + (required - available));
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
