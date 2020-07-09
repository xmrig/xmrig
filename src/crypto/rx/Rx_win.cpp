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
#include "base/io/log/Log.h"
#include "base/kernel/Platform.h"
#include "base/tools/Chrono.h"
#include "crypto/rx/RxConfig.h"


#include <windows.h>
#include <array>
#include <string>
#include <thread>


#define SERVICE_NAME L"WinRing0_1_2_0"


namespace xmrig {


static bool reuseDriver = false;
static const char *tag  = YELLOW_BG_BOLD(WHITE_BOLD_S " msr     ") " ";
static MsrItems savedState;


static SC_HANDLE hManager;
static SC_HANDLE hService;


static bool wrmsr_uninstall_driver()
{
    if (!hService) {
        return true;
    }

    bool result = true;

    if (!reuseDriver) {
        SERVICE_STATUS serviceStatus;

        if (!ControlService(hService, SERVICE_CONTROL_STOP, &serviceStatus)) {
            result = false;
        }

        if (!DeleteService(hService)) {
            LOG_ERR(CLEAR "%s" RED_S "failed to remove WinRing0 driver, error %u", tag, GetLastError());
            result = false;
        }
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
    if (hService) {
        LOG_WARN(CLEAR "%s" YELLOW("service ") YELLOW_BOLD("WinRing0_1_2_0") YELLOW(" is already exists"), tag);

        SERVICE_STATUS status;
        const auto rc = QueryServiceStatus(hService, &status);

        if (rc && Log::isVerbose()) {
            DWORD dwBytesNeeded;

            QueryServiceConfigA(hService, nullptr, 0, &dwBytesNeeded);
            if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
                std::vector<BYTE> buffer(dwBytesNeeded);
                auto config = reinterpret_cast<LPQUERY_SERVICE_CONFIGA>(buffer.data());

                if (QueryServiceConfigA(hService, config, buffer.size(), &dwBytesNeeded)) {
                    LOG_INFO(CLEAR "%s" YELLOW("service path: ") YELLOW_BOLD("\"%s\""), tag, config->lpBinaryPathName);
                }
            }
        }

        if (rc && status.dwCurrentState == SERVICE_RUNNING) {
            reuseDriver = true;
        }
        else if (!wrmsr_uninstall_driver()) {
            return nullptr;
        }
    }

    if (!reuseDriver) {
        hService = CreateServiceW(hManager, SERVICE_NAME, SERVICE_NAME, SERVICE_ALL_ACCESS, SERVICE_KERNEL_DRIVER, SERVICE_DEMAND_START, SERVICE_ERROR_NORMAL, driverPath.c_str(), nullptr, nullptr, nullptr, nullptr, nullptr);
        if (!hService) {
            LOG_ERR(CLEAR "%s" RED_S "failed to install WinRing0 driver, error %u", tag, GetLastError());

            return nullptr;
        }

        if (!StartService(hService, 0, nullptr)) {
            err = GetLastError();
            if (err != ERROR_SERVICE_ALREADY_RUNNING) {
                if (err == ERROR_FILE_NOT_FOUND) {
                    LOG_ERR(CLEAR "%s" RED("failed to start WinRing0 driver: ") RED_BOLD("\"WinRing0x64.sys not found\""), tag);
                }
                else {
                    LOG_ERR(CLEAR "%s" RED_S "failed to start WinRing0 driver, error %u", tag, err);
                }

                wrmsr_uninstall_driver();

                return nullptr;
            }
        }
    }

    HANDLE hDriver = CreateFileW(L"\\\\.\\" SERVICE_NAME, GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (!hDriver) {
        LOG_ERR(CLEAR "%s" RED_S "failed to connect to WinRing0 driver, error %u", tag, GetLastError());

        return nullptr;
    }

    return hDriver;
}


#define IOCTL_READ_MSR  CTL_CODE(40000, 0x821, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define IOCTL_WRITE_MSR CTL_CODE(40000, 0x822, METHOD_BUFFERED, FILE_ANY_ACCESS)


static bool rdmsr(HANDLE driver, uint32_t reg, uint64_t &value)
{
    DWORD size = 0;

    return DeviceIoControl(driver, IOCTL_READ_MSR, &reg, sizeof(reg), &value, sizeof(value), &size, nullptr) != 0;
}


static MsrItem rdmsr(HANDLE driver, uint32_t reg)
{
    uint64_t value = 0;
    if (!rdmsr(driver, reg, value)) {
        LOG_WARN(CLEAR "%s" YELLOW_BOLD_S "cannot read MSR 0x%08" PRIx32, tag, reg);

        return {};
    }

    return { reg, value };
}


static uint64_t get_masked_value(uint64_t old_value, uint64_t new_value, uint64_t mask)
{
    return (new_value & mask) | (old_value & ~mask);
}


static bool wrmsr(HANDLE driver, uint32_t reg, uint64_t value, uint64_t mask)
{
    struct {
        uint32_t reg = 0;
        uint32_t value[2]{};
    } input;

    static_assert(sizeof(input) == 12, "Invalid struct size for WinRing0 driver");

    // If a bit in mask is set to 1, use new value, otherwise use old value
    if (mask != MsrItem::kNoMask) {
        uint64_t old_value;
        if (rdmsr(driver, reg, old_value)) {
            value = get_masked_value(old_value, value, mask);
        }
    }

    input.reg = reg;
    *(reinterpret_cast<uint64_t*>(input.value)) = value;

    DWORD output;
    DWORD k;

    if (!DeviceIoControl(driver, IOCTL_WRITE_MSR, &input, sizeof(input), &output, sizeof(output), &k, nullptr)) {
        LOG_WARN(CLEAR "%s" YELLOW_BOLD_S "cannot set MSR 0x%08" PRIx32 " to 0x%08" PRIx64, tag, reg, value);

        return false;
    }

    return true;
}


static bool wrmsr(const MsrItems &preset, bool save)
{
    bool success = true;

    HANDLE driver = wrmsr_install_driver();
    if (!driver) {
        wrmsr_uninstall_driver();

        if (hManager) {
            CloseServiceHandle(hManager);
        }

        return false;
    }

    if (save) {
        for (const auto &i : preset) {
            auto item = rdmsr(driver, i.reg());
            LOG_VERBOSE(CLEAR "%s" CYAN_BOLD("0x%08" PRIx32) CYAN(":0x%016" PRIx64) CYAN_BOLD(" -> 0x%016" PRIx64), tag, i.reg(), item.value(), get_masked_value(item.value(), i.value(), i.mask()));

            if (item.isValid()) {
                savedState.emplace_back(item);
            }
        }
    }

    std::thread wrmsr_thread([driver, &preset, &success]() {
        for (uint32_t i = 0, n = Cpu::info()->threads(); i < n; ++i) {
            if (!Platform::setThreadAffinity(i)) {
                continue;
            }

            for (const auto &i : preset) {
                success = wrmsr(driver, i.reg(), i.value(), i.mask());
            }

            if (!success) {
                break;
            }
        }
    });

    wrmsr_thread.join();

    CloseHandle(driver);

    wrmsr_uninstall_driver();
    CloseServiceHandle(hManager);

    return success;
}


#ifdef XMRIG_FIX_RYZEN
static thread_local std::pair<const void*, const void*> mainLoopBounds = { nullptr, nullptr };

static LONG WINAPI MainLoopHandler(_EXCEPTION_POINTERS *ExceptionInfo)
{
    if (ExceptionInfo->ExceptionRecord->ExceptionCode == 0xC0000005) {
        const char* accessType;
        switch (ExceptionInfo->ExceptionRecord->ExceptionInformation[0]) {
        case 0: accessType = "read"; break;
        case 1: accessType = "write"; break;
        case 8: accessType = "DEP violation"; break;
        default: accessType = "unknown"; break;
        }
        LOG_VERBOSE(YELLOW_BOLD("[THREAD %u] Access violation at 0x%p: %s at address 0x%p"), GetCurrentThreadId(), ExceptionInfo->ExceptionRecord->ExceptionAddress, accessType, ExceptionInfo->ExceptionRecord->ExceptionInformation[1]);
    }
    else {
        LOG_VERBOSE(YELLOW_BOLD("[THREAD %u] Exception 0x%08X at 0x%p"), GetCurrentThreadId(), ExceptionInfo->ExceptionRecord->ExceptionCode, ExceptionInfo->ExceptionRecord->ExceptionAddress);
    }

    void* p = reinterpret_cast<void*>(ExceptionInfo->ContextRecord->Rip);
    const std::pair<const void*, const void*>& loopBounds = mainLoopBounds;

    if ((loopBounds.first <= p) && (p < loopBounds.second)) {
        ExceptionInfo->ContextRecord->Rip = reinterpret_cast<DWORD64>(loopBounds.second);
        return EXCEPTION_CONTINUE_EXECUTION;
    }

    return EXCEPTION_CONTINUE_SEARCH;
}

void Rx::setMainLoopBounds(const std::pair<const void*, const void*>& bounds)
{
    mainLoopBounds = bounds;
}
#endif


} // namespace xmrig


void xmrig::Rx::msrInit(const RxConfig &config)
{
    const auto &preset = config.msrPreset();
    if (preset.empty()) {
        return;
    }

    const uint64_t ts = Chrono::steadyMSecs();

    if (wrmsr(preset, config.rdmsr())) {
        LOG_NOTICE(CLEAR "%s" GREEN_BOLD_S "register values for \"%s\" preset has been set successfully" BLACK_BOLD(" (%" PRIu64 " ms)"), tag, config.msrPresetName(), Chrono::steadyMSecs() - ts);
    }
    else {
        LOG_ERR(CLEAR "%s" RED_BOLD_S "FAILED TO APPLY MSR MOD, HASHRATE WILL BE LOW", tag);
    }
}


void xmrig::Rx::msrDestroy()
{
    if (savedState.empty()) {
        return;
    }

    const uint64_t ts = Chrono::steadyMSecs();

    if (!wrmsr(savedState, false)) {
        LOG_ERR(CLEAR "%s" RED_BOLD_S "failed to restore initial state" BLACK_BOLD(" (%" PRIu64 " ms)"), tag, Chrono::steadyMSecs() - ts);
    }
}


void xmrig::Rx::setupMainLoopExceptionFrame()
{
#   ifdef XMRIG_FIX_RYZEN
    AddVectoredExceptionHandler(1, MainLoopHandler);
#   endif
}
