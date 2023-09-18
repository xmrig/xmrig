/* XMRig
 * Copyright (c) 2008-2018 Advanced Micro Devices, Inc.
 * Copyright (c) 2018-2021 SChernykh                    <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig                        <https://github.com/xmrig>, <support@xmrig.com>
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


#include "backend/opencl/wrappers/AdlLib.h"
#include "3rdparty/adl/adl_sdk.h"
#include "3rdparty/adl/adl_structures.h"
#include "backend/opencl/wrappers/OclDevice.h"


namespace xmrig {


static std::vector<AdapterInfo> adapters;
static uv_lib_t adlLib;


static const char *kSymbolNotFound                          = "symbol not found";
static const char *kADL_Main_Control_Create                 = "ADL_Main_Control_Create";
static const char *kADL_Main_Control_Destroy                = "ADL_Main_Control_Destroy";
static const char *kADL_Adapter_NumberOfAdapters_Get        = "ADL_Adapter_NumberOfAdapters_Get";
static const char *kADL_Adapter_AdapterInfo_Get             = "ADL_Adapter_AdapterInfo_Get";
static const char *kADL2_Overdrive_Caps                     = "ADL2_Overdrive_Caps";
static const char *kADL2_OverdriveN_FanControl_Get          = "ADL2_OverdriveN_FanControl_Get";
static const char *kADL2_New_QueryPMLogData_Get             = "ADL2_New_QueryPMLogData_Get";
static const char *kADL2_OverdriveN_Temperature_Get         = "ADL2_OverdriveN_Temperature_Get";
static const char *kADL2_OverdriveN_PerformanceStatus_Get   = "ADL2_OverdriveN_PerformanceStatus_Get";
static const char *kADL2_Overdrive6_CurrentPower_Get        = "ADL2_Overdrive6_CurrentPower_Get";


using ADL_MAIN_CONTROL_CREATE               = int (*)(ADL_MAIN_MALLOC_CALLBACK, int);
using ADL_MAIN_CONTROL_DESTROY              = int (*)();
using ADL_ADAPTER_NUMBEROFADAPTERS_GET      = int (*)(int *);
using ADL_ADAPTER_ADAPTERINFO_GET           = int (*)(LPAdapterInfo, int);
using ADL2_OVERDRIVE_CAPS                   = int (*)(ADL_CONTEXT_HANDLE, int, int *, int *, int *);
using ADL2_OVERDRIVEN_FANCONTROL_GET        = int (*)(ADL_CONTEXT_HANDLE, int, ADLODNFanControl *);
using ADL2_NEW_QUERYPMLOGDATA_GET           = int (*)(ADL_CONTEXT_HANDLE, int, ADLPMLogDataOutput *);
using ADL2_OVERDRIVEN_TEMPERATURE_GET       = int (*)(ADL_CONTEXT_HANDLE, int, int, int *);
using ADL2_OVERDRIVEN_PERFORMANCESTATUS_GET = int (*)(ADL_CONTEXT_HANDLE, int, ADLODNPerformanceStatus *);
using ADL2_OVERDRIVE6_CURRENTPOWER_GET      = int (*)(ADL_CONTEXT_HANDLE, int, int, int *);


ADL_MAIN_CONTROL_CREATE                 ADL_Main_Control_Create                 = nullptr;
ADL_MAIN_CONTROL_DESTROY                ADL_Main_Control_Destroy                = nullptr;
ADL_ADAPTER_NUMBEROFADAPTERS_GET        ADL_Adapter_NumberOfAdapters_Get        = nullptr;
ADL_ADAPTER_ADAPTERINFO_GET             ADL_Adapter_AdapterInfo_Get             = nullptr;
ADL2_OVERDRIVE_CAPS                     ADL2_Overdrive_Caps                     = nullptr;
ADL2_OVERDRIVEN_FANCONTROL_GET          ADL2_OverdriveN_FanControl_Get          = nullptr;
ADL2_NEW_QUERYPMLOGDATA_GET             ADL2_New_QueryPMLogData_Get             = nullptr;
ADL2_OVERDRIVEN_TEMPERATURE_GET         ADL2_OverdriveN_Temperature_Get         = nullptr;
ADL2_OVERDRIVEN_PERFORMANCESTATUS_GET   ADL2_OverdriveN_PerformanceStatus_Get   = nullptr;
ADL2_OVERDRIVE6_CURRENTPOWER_GET        ADL2_Overdrive6_CurrentPower_Get        = nullptr;


#define DLSYM(x) if (uv_dlsym(&adlLib, k##x, reinterpret_cast<void**>(&(x))) == -1) { throw std::runtime_error(kSymbolNotFound); }


bool AdlLib::m_initialized         = false;
bool AdlLib::m_ready               = false;


static void * __stdcall ADL_Main_Memory_Alloc(int iSize)
{
    return malloc(iSize); // NOLINT(cppcoreguidelines-no-malloc, hicpp-no-malloc)
}


static inline bool matchTopology(const PciTopology &topology, const AdapterInfo &adapter)
{
    return adapter.iBusNumber > -1 && topology.bus() == adapter.iBusNumber && topology.device() == adapter.iDeviceNumber && topology.function() == adapter.iFunctionNumber;
}


static void getFan_v7(const AdapterInfo &adapter, AdlHealth &health)
{
    ADLODNFanControl data;
    memset(&data, 0, sizeof(ADLODNFanControl));

    if (ADL2_OverdriveN_FanControl_Get(nullptr, adapter.iAdapterIndex, &data) == ADL_OK) {
        health.rpm = data.iCurrentFanSpeed;
    }
}


static void getTemp_v7(const AdapterInfo &adapter, AdlHealth &health)
{
    int temp = 0;
    if (ADL2_OverdriveN_Temperature_Get(nullptr, adapter.iAdapterIndex, 1, &temp) == ADL_OK) {
        health.temperature = temp / 1000;
    }
}


static void getClocks_v7(const AdapterInfo &adapter, AdlHealth &health)
{
    ADLODNPerformanceStatus data;
    memset(&data, 0, sizeof(ADLODNPerformanceStatus));

    if (ADL2_OverdriveN_PerformanceStatus_Get(nullptr, adapter.iAdapterIndex, &data) == ADL_OK) {
        health.clock    = data.iCoreClock / 100;
        health.memClock = data.iMemoryClock / 100;
    }
}


static void getPower_v7(const AdapterInfo &adapter, AdlHealth &health)
{
    int power = 0;
    if (ADL2_Overdrive6_CurrentPower_Get && ADL2_Overdrive6_CurrentPower_Get(nullptr, adapter.iAdapterIndex, 0, &power) == ADL_OK) {
        health.power = static_cast<uint32_t>(power / 256.0);
    }
}


static void getSensorsData_v8(const AdapterInfo &adapter, AdlHealth &health)
{
    if (!ADL2_New_QueryPMLogData_Get) {
        return;
    }

    ADLPMLogDataOutput data;
    memset(&data, 0, sizeof(ADLPMLogDataOutput));

    if (ADL2_New_QueryPMLogData_Get(nullptr, adapter.iAdapterIndex, &data) != ADL_OK) {
        return;
    }

    auto sensorValue = [&data](ADLSensorType type) { return data.sensors[type].supported ? data.sensors[type].value : 0; };

    health.clock        = sensorValue(PMLOG_CLK_GFXCLK);
    health.memClock     = sensorValue(PMLOG_CLK_MEMCLK);
    health.power        = sensorValue(PMLOG_ASIC_POWER);
    health.rpm          = sensorValue(PMLOG_FAN_RPM);
    health.temperature  = sensorValue(PMLOG_TEMPERATURE_HOTSPOT);
}


} // namespace xmrig


bool xmrig::AdlLib::init()
{
    if (!m_initialized) {
        m_ready       = dlopen() && load();
        m_initialized = true;
    }

    return m_ready;
}


const char *xmrig::AdlLib::lastError() noexcept
{
    return uv_dlerror(&adlLib);
}


void xmrig::AdlLib::close()
{
    if (m_ready) {
        ADL_Main_Control_Destroy();
    }

    uv_dlclose(&adlLib);
}


AdlHealth xmrig::AdlLib::health(const OclDevice &device)
{
    if (!isReady() || device.vendorId() != OCL_VENDOR_AMD) {
        return {};
    }

    int supported   = 0;
    int enabled     = 0;
    int version     = 0;
    AdlHealth health;

    for (const auto &adapter : adapters) {
        if (matchTopology(device.topology(), adapter)) {
            if (ADL2_Overdrive_Caps(nullptr, adapter.iAdapterIndex, &supported, &enabled, &version) != ADL_OK) {
                continue;
            }

            if (version == 7) {
                getFan_v7(adapter, health);
                getTemp_v7(adapter, health);
                getClocks_v7(adapter, health);
                getPower_v7(adapter, health);
            }
            else if (version == 8) {
                getSensorsData_v8(adapter, health);
            }

            break;
        }
    }

    return health;
}


bool xmrig::AdlLib::dlopen()
{
    return uv_dlopen("atiadlxx.dll", &adlLib) == 0;
}


bool xmrig::AdlLib::load()
{
    try {
        DLSYM(ADL_Main_Control_Create);
        DLSYM(ADL_Main_Control_Destroy);
        DLSYM(ADL_Adapter_NumberOfAdapters_Get);
        DLSYM(ADL_Adapter_AdapterInfo_Get);
        DLSYM(ADL2_Overdrive_Caps);
        DLSYM(ADL2_OverdriveN_FanControl_Get);
        DLSYM(ADL2_OverdriveN_Temperature_Get);
        DLSYM(ADL2_OverdriveN_PerformanceStatus_Get);
    } catch (std::exception &ex) {
        return false;
    }

    try {
        DLSYM(ADL2_Overdrive6_CurrentPower_Get);
        DLSYM(ADL2_New_QueryPMLogData_Get);
    } catch (std::exception &ex) {}

    if (ADL_Main_Control_Create(ADL_Main_Memory_Alloc, 1) != ADL_OK) {
        return false;
    }

    int count = 0;
    if (ADL_Adapter_NumberOfAdapters_Get(&count) != ADL_OK) {
        return false;
    }

    if (count == 0) {
        return false;
    }

    adapters.resize(count);
    const size_t size = sizeof(adapters[0]) * adapters.size();
    memset(adapters.data(), 0, size);

    return ADL_Adapter_AdapterInfo_Get(adapters.data(), size) == ADL_OK;
}
