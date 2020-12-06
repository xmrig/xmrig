/* XMRig
 * Copyright (c) 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include <cassert>


#include "backend/cpu/Cpu.h"
#include "3rdparty/rapidjson/document.h"


#if defined(XMRIG_FEATURE_HWLOC)
#   include "backend/cpu/platform/HwlocCpuInfo.h"
#else
#   include "backend/cpu/platform/BasicCpuInfo.h"
#endif


static xmrig::ICpuInfo *cpuInfo = nullptr;


xmrig::ICpuInfo *xmrig::Cpu::info()
{
    if (cpuInfo == nullptr) {
#       if defined(XMRIG_FEATURE_HWLOC)
        cpuInfo = new HwlocCpuInfo();
#       else
        cpuInfo = new BasicCpuInfo();
#       endif
    }

    return cpuInfo;
}


rapidjson::Value xmrig::Cpu::toJSON(rapidjson::Document &doc)
{
    return info()->toJSON(doc);
}


void xmrig::Cpu::release()
{
    delete cpuInfo;
    cpuInfo = nullptr;
}
