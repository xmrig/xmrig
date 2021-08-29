/* XMRig
 * Copyright (c) 2018-2019 tevador     <tevador@gmail.com>
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


#include "crypto/rx/RxMsr.h"
#include "backend/cpu/Cpu.h"
#include "backend/cpu/CpuThread.h"
#include "base/io/log/Log.h"
#include "base/tools/Chrono.h"
#include "crypto/rx/RxConfig.h"
#include "hw/msr/Msr.h"


#include <algorithm>
#include <set>


namespace xmrig {


bool RxMsr::m_cacheQoS      = false;
bool RxMsr::m_enabled       = false;
bool RxMsr::m_initialized   = false;


static MsrItems items;


#ifdef XMRIG_OS_WIN
static constexpr inline int32_t get_cpu(int32_t)        { return -1; }
#else
static constexpr inline int32_t get_cpu(int32_t cpu)    { return cpu; }
#endif


static bool wrmsr(const MsrItems &preset, const std::vector<CpuThread> &threads, bool cache_qos, bool save)
{
    auto msr = Msr::get();
    if (!msr) {
        return false;
    }

    if (save) {
        items.reserve(preset.size());

        for (const auto &i : preset) {
            auto item = msr->read(i.reg());
            if (!item.isValid()) {
                items.clear();

                return false;
            }

            LOG_VERBOSE("%s " CYAN_BOLD("0x%08" PRIx32) CYAN(":0x%016" PRIx64) CYAN_BOLD(" -> 0x%016" PRIx64), Msr::tag(), i.reg(), item.value(), MsrItem::maskedValue(item.value(), i.value(), i.mask()));

            items.emplace_back(item);
        }
    }

    // Which CPU cores will have access to the full L3 cache
    std::set<int32_t> cacheEnabled;
    bool cacheQoSDisabled = threads.empty();

    if (cache_qos) {
        const auto &units = Cpu::info()->units();

        for (const auto &t : threads) {
            const auto affinity = static_cast<int32_t>(t.affinity());

            // If some thread has no affinity or wrong affinity, disable cache QoS
            if (affinity < 0 || std::find(units.begin(), units.end(), affinity) == units.end()) {
                cacheQoSDisabled = true;

                LOG_WARN("%s " YELLOW_BOLD("cache QoS can only be enabled when all mining threads have affinity set"), Msr::tag());
                break;
            }

            cacheEnabled.insert(affinity);
        }
    }

    return msr->write([&msr, &preset, cache_qos, &cacheEnabled, cacheQoSDisabled](int32_t cpu) {
        for (const auto &item : preset) {
            if (!msr->write(item, get_cpu(cpu))) {
                return false;
            }
        }

        if (!cache_qos) {
            return true;
        }

        // Assign Class Of Service 0 to current CPU core (default, full L3 cache available)
        if (cacheQoSDisabled || cacheEnabled.count(cpu)) {
            return msr->write(0xC8F, 0, get_cpu(cpu));
        }

        // Disable L3 cache for Class Of Service 1
        if (!msr->write(0xC91, 0, get_cpu(cpu))) {
            // Some CPUs don't let set it to all zeros
            if (!msr->write(0xC91, 1, get_cpu(cpu))) {
                return false;
            }
        }

        // Assign Class Of Service 1 to current CPU core
        return msr->write(0xC8F, 1ULL << 32, get_cpu(cpu));
    });
}


} // namespace xmrig


bool xmrig::RxMsr::init(const RxConfig &config, const std::vector<CpuThread> &threads)
{
    if (isInitialized()) {
        return isEnabled();
    }

    m_initialized = true;
    m_enabled     = false;

    const auto &preset = config.msrPreset();
    if (preset.empty()) {
        return false;
    }

    const uint64_t ts = Chrono::steadyMSecs();
    m_cacheQoS        = config.cacheQoS();

    if (m_cacheQoS && !Cpu::info()->hasCatL3()) {
        LOG_WARN("%s " YELLOW_BOLD("this CPU doesn't support cat_l3, cache QoS is unavailable"), Msr::tag());

        m_cacheQoS = false;
    }

    if ((m_enabled = wrmsr(preset, threads, m_cacheQoS, config.rdmsr()))) {
        LOG_NOTICE("%s " GREEN_BOLD("register values for \"%s\" preset have been set successfully") BLACK_BOLD(" (%" PRIu64 " ms)"), Msr::tag(), config.msrPresetName(), Chrono::steadyMSecs() - ts);
    }
    else {
        LOG_ERR("%s " RED_BOLD("FAILED TO APPLY MSR MOD, HASHRATE WILL BE LOW"), Msr::tag());
    }

    return isEnabled();
}


void xmrig::RxMsr::destroy()
{
    if (!isInitialized()) {
        return;
    }

    m_initialized = false;
    m_enabled     = false;

    if (items.empty()) {
        return;
    }

    const uint64_t ts = Chrono::steadyMSecs();

    if (!wrmsr(items, std::vector<CpuThread>(), m_cacheQoS, false)) {
        LOG_ERR("%s " RED_BOLD("failed to restore initial state" BLACK_BOLD(" (%" PRIu64 " ms)")), Msr::tag(), Chrono::steadyMSecs() - ts);
    }
}
