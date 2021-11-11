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
#include "backend/cpu/Cpu.h"
#include "base/io/log/Log.h"
#include "base/kernel/Platform.h"


#include <string>
#include <thread>
#include <vector>
#include <windows.h>


#define SERVICE_NAME    L"WinRing0_1_2_0"
#define IOCTL_READ_MSR  CTL_CODE(40000, 0x821, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define IOCTL_WRITE_MSR CTL_CODE(40000, 0x822, METHOD_BUFFERED, FILE_ANY_ACCESS)


namespace xmrig {


static const wchar_t *kServiceName = SERVICE_NAME;


class MsrPrivate
{
public:
    bool uninstall()
    {
        if (driver != INVALID_HANDLE_VALUE) {
            CloseHandle(driver);
        }

        if (!service) {
            return true;
        }

        bool result = true;

        if (!reuse) {
            SERVICE_STATUS serviceStatus;

            if (!ControlService(service, SERVICE_CONTROL_STOP, &serviceStatus)) {
                result = false;
            }

            if (!DeleteService(service)) {
                LOG_ERR("%s " RED("failed to remove WinRing0 driver, error %u"), Msr::tag(), GetLastError());
                result = false;
            }
        }

        CloseServiceHandle(service);
        service = nullptr;

        return result;
    }


    bool reuse          = false;
    HANDLE driver       = INVALID_HANDLE_VALUE;
    SC_HANDLE manager   = nullptr;
    SC_HANDLE service   = nullptr;
};


} // namespace xmrig


xmrig::Msr::Msr() : d_ptr(new MsrPrivate())
{
    DWORD err = 0;

    d_ptr->manager = OpenSCManager(nullptr, nullptr, SC_MANAGER_ALL_ACCESS);
    if (!d_ptr->manager) {
        if ((err = GetLastError()) == ERROR_ACCESS_DENIED) {
            LOG_WARN("%s " YELLOW_BOLD("to access MSR registers Administrator privileges required."), tag());
        }
        else {
            LOG_ERR("%s " RED("failed to open service control manager, error %u"), tag(), err);
        }

        return;
    }

    std::vector<wchar_t> dir;

    do {
        dir.resize(dir.empty() ? MAX_PATH : dir.size() * 2);
        GetModuleFileNameW(nullptr, dir.data(), dir.size());
        err = GetLastError();
    } while (err == ERROR_INSUFFICIENT_BUFFER);

    if (err != ERROR_SUCCESS) {
        LOG_ERR("%s " RED("failed to get path to driver, error %u"), tag(), err);
        return;
    }

    for (auto it = dir.end() - 1; it != dir.begin(); --it) {
        if ((*it == L'\\') || (*it == L'/')) {
            ++it;
            *it = L'\0';
            break;
        }
    }

    const std::wstring path = std::wstring(dir.data()) + L"WinRing0x64.sys";

    d_ptr->service = OpenServiceW(d_ptr->manager, kServiceName, SERVICE_ALL_ACCESS);
    if (d_ptr->service) {
        LOG_WARN("%s " YELLOW("service ") YELLOW_BOLD("WinRing0_1_2_0") YELLOW(" already exists"), tag());

        SERVICE_STATUS status;
        const auto rc = QueryServiceStatus(d_ptr->service, &status);

        if (rc) {
            DWORD dwBytesNeeded = 0;

            QueryServiceConfigA(d_ptr->service, nullptr, 0, &dwBytesNeeded);
            if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
                std::vector<BYTE> buffer(dwBytesNeeded);
                auto config = reinterpret_cast<LPQUERY_SERVICE_CONFIGA>(buffer.data());

                if (QueryServiceConfigA(d_ptr->service, config, buffer.size(), &dwBytesNeeded)) {
                    LOG_INFO("%s " YELLOW("service path: ") YELLOW_BOLD("\"%s\""), tag(), config->lpBinaryPathName);
                }
            }
        }

        if (rc && status.dwCurrentState == SERVICE_RUNNING) {
            d_ptr->reuse = true;
        }
        else if (!d_ptr->uninstall()) {
            return;
        }
    }

    d_ptr->driver = CreateFileW(L"\\\\.\\" SERVICE_NAME, GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (d_ptr->driver != INVALID_HANDLE_VALUE) {
        LOG_WARN("%s " YELLOW("service ") YELLOW_BOLD("WinRing0_1_2_0") YELLOW(" already exists, but with a different service name"), tag());
        d_ptr->reuse = true;
        return;
    }

    if (!d_ptr->reuse) {
        d_ptr->service = CreateServiceW(d_ptr->manager, kServiceName, kServiceName, SERVICE_ALL_ACCESS, SERVICE_KERNEL_DRIVER, SERVICE_DEMAND_START, SERVICE_ERROR_NORMAL, path.c_str(), nullptr, nullptr, nullptr, nullptr, nullptr);
        if (!d_ptr->service) {
            LOG_ERR("%s " RED("failed to install WinRing0 driver, error %u"), tag(), GetLastError());

            return;
        }

        if (!StartService(d_ptr->service, 0, nullptr)) {
            err = GetLastError();
            if (err != ERROR_SERVICE_ALREADY_RUNNING) {
                if (err == ERROR_FILE_NOT_FOUND) {
                    LOG_ERR("%s " RED("failed to start WinRing0 driver: ") RED_BOLD("\"WinRing0x64.sys not found\""), tag());
                }
                else {
                    LOG_ERR("%s " RED("failed to start WinRing0 driver, error %u"), tag(), err);
                }

                d_ptr->uninstall();

                return;
            }
        }
    }

    d_ptr->driver = CreateFileW(L"\\\\.\\" SERVICE_NAME, GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (d_ptr->driver == INVALID_HANDLE_VALUE) {
        LOG_ERR("%s " RED("failed to connect to WinRing0 driver, error %u"), tag(), GetLastError());;
    }
}


xmrig::Msr::~Msr()
{
    d_ptr->uninstall();

    delete d_ptr;
}


bool xmrig::Msr::isAvailable() const
{
    return d_ptr->driver != INVALID_HANDLE_VALUE;
}


bool xmrig::Msr::write(Callback &&callback)
{
    const auto &units = Cpu::info()->units();
    bool success      = false;

    std::thread thread([&callback, &units, &success]() {
        for (int32_t pu : units) {
            if (!Platform::setThreadAffinity(pu)) {
                continue;
            }

            if (!callback(pu)) {
                return;
            }
        }

        success = true;
    });

    thread.join();

    return success;
}


bool xmrig::Msr::rdmsr(uint32_t reg, int32_t cpu, uint64_t &value) const
{
    assert(cpu < 0);

    DWORD size = 0;

    return DeviceIoControl(d_ptr->driver, IOCTL_READ_MSR, &reg, sizeof(reg), &value, sizeof(value), &size, nullptr) != 0;
}


bool xmrig::Msr::wrmsr(uint32_t reg, uint64_t value, int32_t cpu)
{
    assert(cpu < 0);

    struct {
        uint32_t reg = 0;
        uint32_t value[2]{};
    } input;

    static_assert(sizeof(input) == 12, "Invalid struct size for WinRing0 driver");

    input.reg = reg;
    *(reinterpret_cast<uint64_t*>(input.value)) = value;

    DWORD output;
    DWORD k;

    return DeviceIoControl(d_ptr->driver, IOCTL_WRITE_MSR, &input, sizeof(input), &output, sizeof(output), &k, nullptr) != 0;
}
