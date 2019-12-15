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
 * Copyright 2007-2009 hiyohiyo                 <https://openlibsys.org>, <hiyohiyo@crystalmark.info>
 * Copyright 2018-2019 SChernykh                <https://github.com/SChernykh>
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
#include "base/kernel/Platform.h"
#include "base/tools/Chrono.h"
#include "crypto/rx/RxConfig.h"


#include <Windows.h>
#include <array>
#include <string>
#include <thread>


#define SERVICE_NAME L"WinRing0_1_2_0"


namespace xmrig {


enum MsrMod : uint32_t {
    MSR_MOD_NONE,
    MSR_MOD_RYZEN,
    MSR_MOD_INTEL,
    MSR_MOD_MAX
};


static const char *tag                                      = YELLOW_BG_BOLD(WHITE_BOLD_S " msr ") " ";
static const std::array<const char *, MSR_MOD_MAX> modNames = { nullptr, "Ryzen", "Intel" };


static SC_HANDLE hManager;
static SC_HANDLE hService;


static bool wrmsr_uninstall_driver()
{
    if (!hService) {
        return true;
    }

    bool result = true;
    SERVICE_STATUS serviceStatus;

    if (!ControlService(hService, SERVICE_CONTROL_STOP, &serviceStatus)) {
        LOG_ERR(CLEAR "%s" RED_S "failed to stop WinRing0 driver, error %u", tag, GetLastError());
        result = false;
    }

    if (!DeleteService(hService)) {
        LOG_ERR(CLEAR "%s" RED_S "failed to remove WinRing0 driver, error %u", tag, GetLastError());
        result = false;
    }

    CloseServiceHandle(hService);
    hService = nullptr;

    return result;
}


static HANDLE wrmsr_install_driver()
{
    DWORD err = 0;

    hManager = OpenSCManager(nullptr, nullptr, SC_MANAGER_ALL_ACCESS);
    if (!hManager) {
        err = GetLastError();

        if (err == ERROR_ACCESS_DENIED) {
            LOG_WARN(CLEAR "%s" YELLOW_BOLD_S "to write MSR registers Administrator privileges required.", tag);
        }
        else {
            LOG_ERR(CLEAR "%s" RED_S "failed to open service control manager, error %u", tag, err);
        }

        return nullptr;
    }

    std::vector<wchar_t> dir;
    dir.resize(MAX_PATH);
    do {
        dir.resize(dir.size() * 2);
        GetModuleFileNameW(nullptr, dir.data(), dir.size());
        err = GetLastError();
    } while (err == ERROR_INSUFFICIENT_BUFFER);

    if (err != ERROR_SUCCESS) {
        LOG_ERR(CLEAR "%s" RED_S "failed to get path to driver, error %u", tag, err);
        return nullptr;
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
    if (hService && !wrmsr_uninstall_driver()) {
        return nullptr;
    }

    hService = CreateServiceW(hManager, SERVICE_NAME, SERVICE_NAME, SERVICE_ALL_ACCESS, SERVICE_KERNEL_DRIVER, SERVICE_DEMAND_START, SERVICE_ERROR_NORMAL, driverPath.c_str(), nullptr, nullptr, nullptr, nullptr, nullptr);
    if (!hService) {
        LOG_ERR(CLEAR "%s" RED_S "failed to install WinRing0 driver, error %u", tag, GetLastError());

        return nullptr;
    }

    if (!StartService(hService, 0, nullptr)) {
        err = GetLastError();
        if (err != ERROR_SERVICE_ALREADY_RUNNING) {
            LOG_ERR(CLEAR "%s" RED_S "failed to start WinRing0 driver, error %u", tag, err);

            CloseServiceHandle(hService);
            hService = nullptr;

            return nullptr;
        }
    }

    HANDLE hDriver = CreateFileW(L"\\\\.\\" SERVICE_NAME, GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (!hDriver) {
        LOG_ERR(CLEAR "%s" RED_S "failed to connect to WinRing0 driver, error %u", tag, GetLastError());

        return nullptr;
    }

    return hDriver;
}


#define IOCTL_WRITE_MSR CTL_CODE(40000, 0x822, METHOD_BUFFERED, FILE_ANY_ACCESS)


static bool wrmsr(HANDLE hDriver, uint32_t reg, uint64_t value) {
    struct {
        uint32_t reg = 0;
        uint32_t value[2]{};
    } input;

    static_assert(sizeof(input) == 12, "Invalid struct size for WinRing0 driver");

    input.reg = reg;
    *((uint64_t*)input.value) = value;

    DWORD output;
    DWORD k;

    if (!DeviceIoControl(hDriver, IOCTL_WRITE_MSR, &input, sizeof(input), &output, sizeof(output), &k, nullptr)) {
        LOG_WARN(CLEAR "%s" YELLOW_BOLD_S "cannot set MSR 0x%08" PRIx32 " to 0x%08" PRIx64, tag, reg, value);

        return false;
    }

    return true;
}


} // namespace xmrig


void xmrig::Rx::osInit(const RxConfig &config)
{
    if ((config.wrmsr() < 0) || !ICpuInfo::isX64()) {
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

    HANDLE hDriver = wrmsr_install_driver();
    if (!hDriver) {
        wrmsr_uninstall_driver();

        if (hManager) {
            CloseServiceHandle(hManager);
        }

        return;
    }

    std::thread wrmsr_thread([hDriver, mod, &config]() {
        for (uint32_t i = 0, n = Cpu::info()->threads(); i < n; ++i) {
            Platform::setThreadAffinity(i);

            if (mod == MSR_MOD_RYZEN) {
                wrmsr(hDriver, 0xC0011020, 0);
                wrmsr(hDriver, 0xC0011021, 0x40);
                wrmsr(hDriver, 0xC0011022, 0x510000);
                wrmsr(hDriver, 0xC001102b, 0x1808cc16);
            }
            else if (mod == MSR_MOD_INTEL) {
                wrmsr(hDriver, 0x1a4, config.wrmsr());
            }
        }
    });

    wrmsr_thread.join();

    CloseHandle(hDriver);

    wrmsr_uninstall_driver();
    CloseServiceHandle(hManager);

    LOG_NOTICE(CLEAR "%s" GREEN_BOLD_S "register values for %s has been set successfully" BLACK_BOLD(" (%" PRIu64 " ms)"), tag, modNames[mod], Chrono::steadyMSecs() - ts);
}
