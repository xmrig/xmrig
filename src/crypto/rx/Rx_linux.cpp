/* XMRig
 * Copyright 2010      Jeff Garzik              <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler                   <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones              <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466                 <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee                <jayddee246@gmail.com>
 * Copyright 2017-2019 XMR-Stak                 <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett              <https://github.com/vtnerd>
 * Copyright 2018-2019 tevador                  <tevador@gmail.com>
 * Copyright 2018-2019 SChernykh                <https://github.com/SChernykh>
 * Copyright 2000      Transmeta Corporation    <https://github.com/intel/msr-tools>
 * Copyright 2004-2008 H. Peter Anvin           <https://github.com/intel/msr-tools>
 * Copyright 2016-2019 XMRig                    <https://github.com/xmrig>, <support@xmrig.com>
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


#include "crypto/rx/Rx.h"
#include "backend/cpu/Cpu.h"
#include "base/io/log/Log.h"
#include "base/tools/Chrono.h"
#include "crypto/rx/RxConfig.h"


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


enum MsrMod : uint32_t {
    MSR_MOD_NONE,
    MSR_MOD_RYZEN,
    MSR_MOD_INTEL,
    MSR_MOD_MAX
};


static const char *tag                                      = YELLOW_BG_BOLD(WHITE_BOLD_S " msr ") " ";
static const std::array<const char *, MSR_MOD_MAX> modNames = { nullptr, "Ryzen", "Intel" };


static inline int dir_filter(const struct dirent *dirp)
{
    return isdigit(dirp->d_name[0]) ? 1 : 0;
}


static bool wrmsr_on_cpu(uint32_t reg, uint32_t cpu, uint64_t value)
{
    char msr_file_name[64]{};

    sprintf(msr_file_name, "/dev/cpu/%d/msr", cpu);
    int fd = open(msr_file_name, O_WRONLY);
    if (fd < 0) {
        return false;
    }

    const bool success = pwrite(fd, &value, sizeof value, reg) == sizeof value;

    close(fd);

    return success;
}


static bool wrmsr_on_all_cpus(uint32_t reg, uint64_t value)
{
    struct dirent **namelist;
    int dir_entries = scandir("/dev/cpu", &namelist, dir_filter, 0);
    int errors      = 0;

    while (dir_entries--) {
        if (!wrmsr_on_cpu(reg, strtoul(namelist[dir_entries]->d_name, nullptr, 10), value)) {
            ++errors;
        }

        free(namelist[dir_entries]);
    }

    free(namelist);

    if (errors) {
        LOG_WARN(CLEAR "%s" YELLOW_BOLD_S "cannot set MSR 0x%08" PRIx32 " to 0x%08" PRIx64, tag, reg, value);
    }

    return errors == 0;
}


static bool wrmsr_modprobe()
{
    if (system("/sbin/modprobe msr > /dev/null 2>&1") != 0) {
        LOG_WARN(CLEAR "%s" YELLOW_BOLD_S "msr kernel module is not available", tag);

        return false;
    }

    return true;
}


} // namespace xmrig


void xmrig::Rx::osInit(const RxConfig &config)
{
    if (config.wrmsr() < 0) {
        return;
    }

    MsrMod mod = MSR_MOD_NONE;
    if (Cpu::info()->assembly() == Assembly::RYZEN) {
        mod = MSR_MOD_RYZEN;
    }
    else if (Cpu::info()->vendor() == ICpuInfo::VENDOR_INTEL) {
        mod = MSR_MOD_INTEL;
    }

    if (mod == MSR_MOD_NONE) {
        return;
    }

    const uint64_t ts = Chrono::steadyMSecs();

    if (!wrmsr_modprobe()) {
        return;
    }

    if (mod == MSR_MOD_RYZEN) {
        wrmsr_on_all_cpus(0xC0011020, 0);
        wrmsr_on_all_cpus(0xC0011021, 0x40);
        wrmsr_on_all_cpus(0xC0011022, 0x510000);
        wrmsr_on_all_cpus(0xC001102b, 0x1808cc16);
    }
    else if (mod == MSR_MOD_INTEL) {
        wrmsr_on_all_cpus(0x1a4, config.wrmsr());
    }

    LOG_NOTICE(CLEAR "%s" GREEN_BOLD_S "register values for %s has been set successfully" BLACK_BOLD(" (%" PRIu64 " ms)"), tag, modNames[mod], Chrono::steadyMSecs() - ts);
}
