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

#include <stdexcept>
#include <uv.h>


#include "backend/cuda/wrappers/NvmlLib.h"
#include "backend/cuda/wrappers/nvml_lite.h"
#include "base/io/log/Log.h"



namespace xmrig {


static uv_lib_t nvmlLib;


static const char *kNvmlDeviceGetClockInfo                          = "nvmlDeviceGetClockInfo";
static const char *kNvmlDeviceGetCount                              = "nvmlDeviceGetCount_v2";
static const char *kNvmlDeviceGetFanSpeed                           = "nvmlDeviceGetFanSpeed";
static const char *kNvmlDeviceGetFanSpeed_v2                        = "nvmlDeviceGetFanSpeed_v2";
static const char *kNvmlDeviceGetHandleByIndex                      = "nvmlDeviceGetHandleByIndex_v2";
static const char *kNvmlDeviceGetPciInfo                            = "nvmlDeviceGetPciInfo_v2";
static const char *kNvmlDeviceGetPowerUsage                         = "nvmlDeviceGetPowerUsage";
static const char *kNvmlDeviceGetTemperature                        = "nvmlDeviceGetTemperature";
static const char *kNvmlInit                                        = "nvmlInit_v2";
static const char *kNvmlShutdown                                    = "nvmlShutdown";
static const char *kNvmlSystemGetDriverVersion                      = "nvmlSystemGetDriverVersion";
static const char *kNvmlSystemGetNVMLVersion                        = "nvmlSystemGetNVMLVersion";
static const char *kSymbolNotFound                                  = "symbol not found";


static nvmlReturn_t (*pNvmlDeviceGetClockInfo)(nvmlDevice_t device, uint32_t type, uint32_t *clock)                         = nullptr;
static nvmlReturn_t (*pNvmlDeviceGetCount)(uint32_t *deviceCount)                                                           = nullptr;
static nvmlReturn_t (*pNvmlDeviceGetFanSpeed_v2)(nvmlDevice_t device, uint32_t fan, uint32_t *speed)                        = nullptr;
static nvmlReturn_t (*pNvmlDeviceGetFanSpeed)(nvmlDevice_t device, uint32_t *speed)                                         = nullptr;
static nvmlReturn_t (*pNvmlDeviceGetHandleByIndex)(uint32_t index, nvmlDevice_t *device)                                    = nullptr;
static nvmlReturn_t (*pNvmlDeviceGetPciInfo)(nvmlDevice_t device, nvmlPciInfo_t *pci)                                       = nullptr;
static nvmlReturn_t (*pNvmlDeviceGetPowerUsage)(nvmlDevice_t device, uint32_t *power)                                       = nullptr;
static nvmlReturn_t (*pNvmlDeviceGetTemperature)(nvmlDevice_t device, uint32_t sensorType, uint32_t *temp)                  = nullptr;
static nvmlReturn_t (*pNvmlInit)()                                                                                          = nullptr;
static nvmlReturn_t (*pNvmlShutdown)()                                                                                      = nullptr;
static nvmlReturn_t (*pNvmlSystemGetDriverVersion)(char *version, uint32_t length)                                          = nullptr;
static nvmlReturn_t (*pNvmlSystemGetNVMLVersion)(char *version, uint32_t length)                                            = nullptr;


#define DLSYM(x) if (uv_dlsym(&nvmlLib, k##x, reinterpret_cast<void**>(&p##x)) == -1) { throw std::runtime_error(kSymbolNotFound); }


bool NvmlLib::m_initialized         = false;
bool NvmlLib::m_ready               = false;
char NvmlLib::m_driverVersion[80]   = { 0 };
char NvmlLib::m_nvmlVersion[80]     = { 0 };
String NvmlLib::m_loader;


} // namespace xmrig


bool xmrig::NvmlLib::init(const char *fileName)
{
    if (!m_initialized) {
        m_loader      = fileName;
        m_ready       = dlopen() && load();
        m_initialized = true;
    }

    return m_ready;
}


const char *xmrig::NvmlLib::lastError() noexcept
{
    return uv_dlerror(&nvmlLib);
}


void xmrig::NvmlLib::close()
{
    if (m_ready) {
        pNvmlShutdown();
    }

    uv_dlclose(&nvmlLib);
}


bool xmrig::NvmlLib::assign(std::vector<CudaDevice> &devices)
{
    uint32_t count = 0;
    if (pNvmlDeviceGetCount(&count) != NVML_SUCCESS) {
        return false;
    }

    for (uint32_t i = 0; i < count; i++) {
        nvmlDevice_t nvmlDevice = nullptr;
        if (pNvmlDeviceGetHandleByIndex(i, &nvmlDevice) != NVML_SUCCESS) {
            continue;
        }

        nvmlPciInfo_t pci;
        if (pNvmlDeviceGetPciInfo(nvmlDevice, &pci) != NVML_SUCCESS) {
            continue;
        }

        for (auto &device : devices) {
            if (device.topology().bus() == pci.bus && device.topology().device() == pci.device) {
                device.setNvmlDevice(nvmlDevice);
            }
        }
    }

    return true;
}


NvmlHealth xmrig::NvmlLib::health(nvmlDevice_t device)
{
    if (!device) {
        return {};
    }

    NvmlHealth health;
    pNvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &health.temperature);
    pNvmlDeviceGetPowerUsage(device, &health.power);
    pNvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &health.clock);
    pNvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &health.memClock);

    if (health.power) {
        health.power /= 1000;
    }

    uint32_t speed = 0;

    if (pNvmlDeviceGetFanSpeed_v2) {
        uint32_t i = 0;

        while (pNvmlDeviceGetFanSpeed_v2(device, i, &speed) == NVML_SUCCESS) {
            health.fanSpeed.push_back(speed);
            ++i;
        }

    }
    else {
        pNvmlDeviceGetFanSpeed(device, &speed);

        health.fanSpeed.push_back(speed);
    }

    return health;
}


bool xmrig::NvmlLib::dlopen()
{
    if (!m_loader.isNull()) {
        return uv_dlopen(m_loader, &nvmlLib) == 0;
    }

#   ifdef _WIN32
    if (uv_dlopen("nvml.dll", &nvmlLib) == 0) {
        return true;
    }

    char path[MAX_PATH] = { 0 };
    ExpandEnvironmentStringsA("%PROGRAMFILES%\\NVIDIA Corporation\\NVSMI\\nvml.dll", path, sizeof(path));

    return uv_dlopen(path, &nvmlLib) == 0;
#   else
    return uv_dlopen("libnvidia-ml.so", &nvmlLib) == 0;
#   endif
}


bool xmrig::NvmlLib::load()
{
    try {
        DLSYM(NvmlDeviceGetClockInfo);
        DLSYM(NvmlDeviceGetCount);
        DLSYM(NvmlDeviceGetFanSpeed);
        DLSYM(NvmlDeviceGetHandleByIndex);
        DLSYM(NvmlDeviceGetPciInfo);
        DLSYM(NvmlDeviceGetPowerUsage);
        DLSYM(NvmlDeviceGetTemperature);
        DLSYM(NvmlInit);
        DLSYM(NvmlShutdown);
        DLSYM(NvmlSystemGetDriverVersion);
        DLSYM(NvmlSystemGetNVMLVersion);
    } catch (std::exception &ex) {
        return false;
    }

    uv_dlsym(&nvmlLib, kNvmlDeviceGetFanSpeed_v2, reinterpret_cast<void**>(&pNvmlDeviceGetFanSpeed_v2));

    if (pNvmlInit() != NVML_SUCCESS) {
        return false;
    }

    pNvmlSystemGetDriverVersion(m_driverVersion, sizeof(m_driverVersion));
    pNvmlSystemGetNVMLVersion(m_nvmlVersion, sizeof(m_nvmlVersion));

    return true;
}
