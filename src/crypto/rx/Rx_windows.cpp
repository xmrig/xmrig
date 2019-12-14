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
#include "backend/common/Tags.h"
#include "backend/cpu/Cpu.h"
#include "base/io/log/Log.h"
#include "base/kernel/Platform.h"
#include "crypto/rx/RxConfig.h"

#include <Windows.h>
#include <string>
#include <thread>

#define SERVICE_NAME L"WinRing0_1_2_0"

static SC_HANDLE hManager;
static SC_HANDLE hService;

static bool uninstall_driver()
{
    bool result = true;
    DWORD err;
    SERVICE_STATUS serviceStatus;
    if (!ControlService(hService, SERVICE_CONTROL_STOP, &serviceStatus)) {
        err = GetLastError();
        LOG_ERR("Failed to stop WinRing0 driver, error %u", err);
        result = false;
    }
    if (!DeleteService(hService)) {
        err = GetLastError();
        LOG_ERR("Failed to remove WinRing0 driver, error %u", err);
        result = false;
    }
    return result;
}

static HANDLE install_driver()
{
    DWORD err = 0;

    hManager = OpenSCManager(nullptr, nullptr, SC_MANAGER_ALL_ACCESS);
    if (!hManager) {
        err = GetLastError();
        LOG_ERR("Failed to open service control manager, error %u", err);
        return 0;
    }

    std::vector<wchar_t> dir;
    dir.resize(MAX_PATH);
    do {
        dir.resize(dir.size() * 2);
        DWORD len = GetModuleFileNameW(NULL, dir.data(), dir.size());
        err = GetLastError();
    } while (err == ERROR_INSUFFICIENT_BUFFER);

    if (err != ERROR_SUCCESS) {
        LOG_ERR("Failed to get path to driver, error %u", err);
        return 0;
    }

    for (auto it = dir.end(); it != dir.begin(); --it) {
        if ((*it == L'\\') || (*it == L'/')) {
            ++it;
            *it = L'\0';
            break;
        }
    }

    std::wstring driverPath = dir.data();
    driverPath += L"WinRing0x64.sys";

    hService = OpenServiceW(hManager, SERVICE_NAME, SERVICE_ALL_ACCESS);
    if (hService) {
        if (!uninstall_driver()) {
            return 0;
        }
        CloseServiceHandle(hService);
        hService = 0;
    }
    else {
        err = GetLastError();
        if (err != ERROR_SERVICE_DOES_NOT_EXIST) {
            LOG_ERR("Failed to open WinRing0 driver, error %u", err);
            return 0;
        }
    }

    hService = CreateServiceW(hManager, SERVICE_NAME, SERVICE_NAME, SERVICE_ALL_ACCESS, SERVICE_KERNEL_DRIVER, SERVICE_DEMAND_START, SERVICE_ERROR_NORMAL, driverPath.c_str(), nullptr, nullptr, nullptr, nullptr, nullptr);
    if (!hService) {
        LOG_ERR("Failed to install WinRing0 driver, error %u", err);
    }

    if (!StartService(hService, 0, nullptr)) {
        err = GetLastError();
        if (err != ERROR_SERVICE_ALREADY_RUNNING) {
            LOG_ERR("Failed to start WinRing0 driver, error %u", err);
            return 0;
        }
    }

    HANDLE hDriver = CreateFileW(L"\\\\.\\" SERVICE_NAME, GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (!hDriver) {
        err = GetLastError();
        LOG_ERR("Failed to connect to WinRing0 driver, error %u", err);
        return 0;
    }

    return hDriver;
}


#define IOCTL_WRITE_MSR CTL_CODE(40000, 0x822, METHOD_BUFFERED, FILE_ANY_ACCESS)

static bool wrmsr(HANDLE hDriver, uint32_t reg, uint64_t value) {
    struct {
        uint32_t reg;
        uint32_t value[2];
    } input;
    static_assert(sizeof(input) == 12, "Invalid struct size for WinRing0 driver");

    input.reg = reg;
    *((uint64_t*)input.value) = value;

    DWORD output;
    DWORD k;
    if (!DeviceIoControl(hDriver, IOCTL_WRITE_MSR, &input, sizeof(input), &output, sizeof(output), &k, nullptr)) {
        const DWORD err = GetLastError();
        LOG_WARN(CLEAR "%s" YELLOW_BOLD_S "cannot set MSR 0x%08" PRIx32 " to 0x%08" PRIx64, xmrig::rx_tag(), reg, value);
        return false;
    }

    return true;
}


void xmrig::Rx::osInit(const RxConfig &config)
{
    if ((config.wrmsr() < 0) || !ICpuInfo::isX64()) {
        return;
    }

    const char* msr_mod_variant = (Cpu::info()->assembly() == Assembly::RYZEN) ? "Ryzen" :
                                 ((Cpu::info()->vendor() == ICpuInfo::VENDOR_INTEL) ? "Intel" : nullptr);

    if (!msr_mod_variant) {
        return;
    }

    LOG_INFO(CLEAR "%s" GREEN_BOLD_S "MSR mod: loading WinRing0 driver", xmrig::rx_tag());

    HANDLE hDriver = install_driver();
    if (!hDriver) {
        if (hService) {
            uninstall_driver();
            CloseServiceHandle(hService);
        }
        if (hManager) {
            CloseServiceHandle(hManager);
        }
        return;
    }

    LOG_INFO(CLEAR "%s" GREEN_BOLD_S "MSR mod: setting MSR register values for %s", xmrig::rx_tag(), msr_mod_variant);

    std::thread wrmsr_thread([hDriver, &config]() {
        for (uint32_t i = 0, n = std::thread::hardware_concurrency(); i < n; ++i) {
            Platform::setThreadAffinity(i);
            if (Cpu::info()->assembly() == Assembly::RYZEN) {
                wrmsr(hDriver, 0xC0011020, 0);
                wrmsr(hDriver, 0xC0011021, 0x40);
                wrmsr(hDriver, 0xC0011022, 0x510000);
                wrmsr(hDriver, 0xC001102b, 0x1808cc16);
            }
            else if (Cpu::info()->vendor() == ICpuInfo::VENDOR_INTEL) {
                wrmsr(hDriver, 0x1a4, config.wrmsr());
            }
        }
    });
    wrmsr_thread.join();

    CloseHandle(hDriver);

    uninstall_driver();

    CloseServiceHandle(hService);
    CloseServiceHandle(hManager);

    LOG_INFO(CLEAR "%s" GREEN_BOLD_S "MSR mod: all done, WinRing0 driver unloaded", xmrig::rx_tag());
}
