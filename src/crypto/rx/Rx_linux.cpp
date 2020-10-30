/* XMRig
 * Copyright 2010      Jeff Garzik              <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler                   <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones              <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466                 <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee                <jayddee246@gmail.com>
 * Copyright 2017-2019 XMR-Stak                 <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett              <https://github.com/vtnerd>
 * Copyright 2018-2019 tevador                  <tevador@gmail.com>
 * Copyright 2000      Transmeta Corporation    <https://github.com/intel/msr-tools>
 * Copyright 2004-2008 H. Peter Anvin           <https://github.com/intel/msr-tools>
 * Copyright 2018-2020 SChernykh                <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig                    <https://github.com/xmrig>, <support@xmrig.com>
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
#include "backend/cpu/CpuThread.h"
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
#include <signal.h>
#include <ucontext.h>


namespace xmrig {


static const char *tag = YELLOW_BG_BOLD(WHITE_BOLD_S " msr     ") " ";
static MsrItems savedState;


static inline int dir_filter(const struct dirent *dirp)
{
    return isdigit(dirp->d_name[0]) ? 1 : 0;
}


bool rdmsr_on_cpu(uint32_t reg, uint32_t cpu, uint64_t &value)
{
    char msr_file_name[64]{};

    sprintf(msr_file_name, "/dev/cpu/%u/msr", cpu);
    int fd = open(msr_file_name, O_RDONLY);
    if (fd < 0) {
        return false;
    }

    const bool success = pread(fd, &value, sizeof value, reg) == sizeof value;

    close(fd);

    return success;
}


static MsrItem rdmsr(uint32_t reg)
{
    uint64_t value = 0;
    if (!rdmsr_on_cpu(reg, 0, value)) {
        LOG_WARN(CLEAR "%s" YELLOW_BOLD_S "cannot read MSR 0x%08" PRIx32, tag, reg);

        return {};
    }

    return { reg, value };
}


static uint64_t get_masked_value(uint64_t old_value, uint64_t new_value, uint64_t mask)
{
    return (new_value & mask) | (old_value & ~mask);
}


static bool wrmsr_on_cpu(uint32_t reg, uint32_t cpu, uint64_t value, uint64_t mask)
{
    // If a bit in mask is set to 1, use new value, otherwise use old value
    if (mask != MsrItem::kNoMask) {
        uint64_t old_value;
        if (rdmsr_on_cpu(reg, cpu, old_value)) {
            value = get_masked_value(old_value, value, mask);
        }
    }

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


template<typename T>
static bool wrmsr_on_all_cpus(uint32_t reg, uint64_t value, uint64_t mask, T&& callback)
{
    struct dirent **namelist;
    int dir_entries = scandir("/dev/cpu", &namelist, dir_filter, 0);
    int errors      = 0;

    while (dir_entries--) {
        if (!callback(reg, strtoul(namelist[dir_entries]->d_name, nullptr, 10), value, mask)) {
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
    if (system("/sbin/modprobe msr allow_writes=on > /dev/null 2>&1") != 0) {
        LOG_WARN(CLEAR "%s" YELLOW_BOLD_S "msr kernel module is not available", tag);

        return false;
    }

    return true;
}


static bool wrmsr(const MsrItems& preset, const std::vector<CpuThread>& threads, bool cache_qos, bool save)
{
    if (!wrmsr_modprobe()) {
        return false;
    }

    if (save) {
        for (const auto &i : preset) {
            auto item = rdmsr(i.reg());
            LOG_VERBOSE(CLEAR "%s" CYAN_BOLD("0x%08" PRIx32) CYAN(":0x%016" PRIx64) CYAN_BOLD(" -> 0x%016" PRIx64), tag, i.reg(), item.value(), get_masked_value(item.value(), i.value(), i.mask()));

            if (item.isValid()) {
                savedState.emplace_back(item);
            }
        }
    }

    for (const auto &i : preset) {
        if (!wrmsr_on_all_cpus(i.reg(), i.value(), i.mask(), [](uint32_t reg, uint32_t cpu, uint64_t value, uint64_t mask) { return wrmsr_on_cpu(reg, cpu, value, mask); })) {
            return false;
        }
    }

    const uint32_t n = Cpu::info()->threads();

    // Which CPU cores will have access to the full L3 cache
    std::vector<bool> cacheEnabled(n, false);
    bool cacheQoSDisabled = threads.empty();

    for (const CpuThread& t : threads) {
        // If some thread has no affinity or wrong affinity, disable cache QoS
        if ((t.affinity() < 0) || (t.affinity() >= n)) {
            cacheQoSDisabled = true;
            if (cache_qos) {
                LOG_WARN(CLEAR "%s" YELLOW_BOLD_S "Cache QoS can only be enabled when all mining threads have affinity set", tag);
            }
            break;
        }

        cacheEnabled[t.affinity()] = true;
    }

    if (cache_qos && !Cpu::info()->hasCatL3()) {
        if (!threads.empty()) {
            LOG_WARN(CLEAR "%s" YELLOW_BOLD_S "This CPU doesn't support cat_l3, cache QoS is unavailable", tag);
        }
        cache_qos = false;
    }

    bool result = true;

    if (cache_qos) {
        result = wrmsr_on_all_cpus(0xC8F, 0, MsrItem::kNoMask, [&cacheEnabled, cacheQoSDisabled](uint32_t, uint32_t cpu, uint64_t, uint64_t) {
            if (cacheQoSDisabled || (cpu >= cacheEnabled.size()) || cacheEnabled[cpu]) {
                // Assign Class Of Service 0 to current CPU core (default, full L3 cache available)
                if (!wrmsr_on_cpu(0xC8F, cpu, 0, MsrItem::kNoMask)) {
                    return false;
                }
            }
            else {
                // Disable L3 cache for Class Of Service 1
                if (!wrmsr_on_cpu(0xC91, cpu, 0, MsrItem::kNoMask)) {
                    // Some CPUs don't let set it to all zeros
                    if (!wrmsr_on_cpu(0xC91, cpu, 1, MsrItem::kNoMask)) {
                        return false;
                    }
                }

                // Assign Class Of Service 1 to current CPU core
                if (!wrmsr_on_cpu(0xC8F, cpu, 1ULL << 32, MsrItem::kNoMask)) {
                    return false;
                }
            }
            return true;
        });
    }

    return result;
}


#ifdef XMRIG_FIX_RYZEN
static thread_local std::pair<const void*, const void*> mainLoopBounds = { nullptr, nullptr };

static void MainLoopHandler(int sig, siginfo_t *info, void *ucontext)
{
    ucontext_t *ucp = (ucontext_t*) ucontext;

    LOG_VERBOSE(YELLOW_BOLD("%s at %p"), (sig == SIGSEGV) ? "SIGSEGV" : "SIGILL", ucp->uc_mcontext.gregs[REG_RIP]);

    void* p = reinterpret_cast<void*>(ucp->uc_mcontext.gregs[REG_RIP]);
    const std::pair<const void*, const void*>& loopBounds = mainLoopBounds;

    if ((loopBounds.first <= p) && (p < loopBounds.second)) {
        ucp->uc_mcontext.gregs[REG_RIP] = reinterpret_cast<size_t>(loopBounds.second);
    }
    else {
        abort();
    }
}

void Rx::setMainLoopBounds(const std::pair<const void*, const void*>& bounds)
{
    mainLoopBounds = bounds;
}
#endif


} // namespace xmrig


bool xmrig::Rx::msrInit(const RxConfig &config, const std::vector<CpuThread> &threads)
{
    const auto &preset = config.msrPreset();
    if (preset.empty()) {
        return false;
    }

    const uint64_t ts = Chrono::steadyMSecs();

    if (wrmsr(preset, threads, config.cacheQoS(), config.rdmsr())) {
        LOG_NOTICE(CLEAR "%s" GREEN_BOLD_S "register values for \"%s\" preset has been set successfully" BLACK_BOLD(" (%" PRIu64 " ms)"), tag, config.msrPresetName(), Chrono::steadyMSecs() - ts);

        return true;
    }


    LOG_ERR(CLEAR "%s" RED_BOLD_S "FAILED TO APPLY MSR MOD, HASHRATE WILL BE LOW", tag);

    return false;
}


void xmrig::Rx::msrDestroy()
{
    if (savedState.empty()) {
        return;
    }

    const uint64_t ts = Chrono::steadyMSecs();

    if (!wrmsr(savedState, std::vector<CpuThread>(), true, false)) {
        LOG_ERR(CLEAR "%s" RED_BOLD_S "failed to restore initial state" BLACK_BOLD(" (%" PRIu64 " ms)"), tag, Chrono::steadyMSecs() - ts);
    }
}


void xmrig::Rx::setupMainLoopExceptionFrame()
{
#   ifdef XMRIG_FIX_RYZEN
    struct sigaction act = {};
    act.sa_sigaction = MainLoopHandler;
    act.sa_flags = SA_RESTART | SA_SIGINFO;
    sigaction(SIGSEGV, &act, nullptr);
    sigaction(SIGILL, &act, nullptr);
#   endif
}
