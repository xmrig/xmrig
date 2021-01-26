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


#include "hw/msr/Msr.h"
#include "3rdparty/fmt/core.h"
#include "backend/cpu/Cpu.h"
#include "base/io/log/Log.h"


#include <array>
#include <cctype>
#include <cinttypes>
#include <cstdio>
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>


namespace xmrig {


static int msr_open(int32_t cpu, int flags)
{
    const auto name = fmt::format("/dev/cpu/{}/msr", cpu < 0 ? Cpu::info()->units().front() : cpu);

    return open(name.c_str(), flags);
}


class MsrPrivate
{
public:
    bool available = true;
};


} // namespace xmrig


xmrig::Msr::Msr() : d_ptr(new MsrPrivate())
{
    if (system("/sbin/modprobe msr allow_writes=on > /dev/null 2>&1") != 0) {
        LOG_WARN("%s " YELLOW_BOLD("msr kernel module is not available"), Msr::tag());

        d_ptr->available = false;
    }
}


xmrig::Msr::~Msr()
{
    delete d_ptr;
}


bool xmrig::Msr::isAvailable() const
{
    return d_ptr->available;
}


bool xmrig::Msr::write(Callback &&callback)
{
    const auto &units = Cpu::info()->units();

    for (int32_t pu : units) {
        if (!callback(pu)) {
            return false;
        }
    }

    return true;
}


bool xmrig::Msr::rdmsr(uint32_t reg, int32_t cpu, uint64_t &value) const
{
    const int fd = msr_open(cpu, O_RDONLY);

    if (fd < 0) {
        return false;
    }

    const bool success = pread(fd, &value, sizeof value, reg) == sizeof value;
    close(fd);

    return success;
}


bool xmrig::Msr::wrmsr(uint32_t reg, uint64_t value, int32_t cpu)
{
    const int fd = msr_open(cpu, O_WRONLY);

    if (fd < 0) {
        return false;
    }

    const bool success = pwrite(fd, &value, sizeof value, reg) == sizeof value;

    close(fd);

    return success;
}
