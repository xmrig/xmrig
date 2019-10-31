/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2019 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_NVML_LITE_H
#define XMRIG_NVML_LITE_H


#include <cstdint>


#define NVML_SUCCESS         0
#define NVML_TEMPERATURE_GPU 0
#define NVML_CLOCK_SM        1
#define NVML_CLOCK_MEM       2


using nvmlReturn_t = uint32_t;
using nvmlDevice_t = struct nvmlDevice_st *;


struct nvmlPciInfo_t
{
    char busIdLegacy[16]{};
    unsigned int domain         = 0;
    unsigned int bus            = 0;
    unsigned int device         = 0;
    unsigned int pciDeviceId    = 0;
    unsigned int pciSubSystemId = 0;

    char busId[32]{};
};


#endif /* XMRIG_NVML_LITE_H */
