/* xmr_arch64
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
 * Copyright 2017-2018 ePlus Systems Ltd. <xmr@eplus.systems>h
 *
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

#ifndef __SELFMAINTENANCE_H__
#define __SELFMAINTENANCE_H__


#include <stdint.h>


class SelfMaintenance
{
public:

    int getCPUTemperature(int pT);
    int getFileSystemAvailable();

private:
    int m_cpuSingleCoreSpeed;
    int m_cpuCoresCount;
    int m_cpuTemperatureC;
    int m_cpuTemperatureF;
};


#endif /* __SELFMAINTENANCE_H__ */
