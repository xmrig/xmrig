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
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <sys/sysinfo.h>
#include <typeinfo>
#include <thread>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <locale>
#include <limits>
#include <string>

#include "SelfMaintenance.h"

/*---------------------------------------------------------------------
* NAME       : SelfMaintenance::getCPUTemperature()
* SYNOPSIS   : Get CPU temperature working temperature (Celsius)
* DESCRIPTION:
*
---------------------------------------------------------------------*/
int SelfMaintenance::getCPUTemperature(int pT){
    using namespace std;
    stringstream   strStream;

    unsigned num_cpu = std::thread::hardware_concurrency();
    m_cpuCoresCount = num_cpu;
    //---
    ifstream cpu_freq("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq");
    ifstream cpu_temp("/sys/class/thermal/thermal_zone0/temp1");
    //---
    strStream << cpu_freq.rdbuf();
    strStream >> m_cpuSingleCoreSpeed;
    //---
    strStream.str("");
    strStream << cpu_temp.rdbuf();
    strStream >> m_cpuTemperatureC;
	//--
	return(m_cpuTemperatureC);
}

/*---------------------------------------------------------------------
* NAME       : SelfMaintenance::getFileSystemAvailable()
* SYNOPSIS   : File system available in, [0-100%]
* DESCRIPTION:
*
---------------------------------------------------------------------*/
int SelfMaintenance::getFileSystemAvailable(){
/*	    stringstream   strStream;
	    double  hdd_size;
	    double  hdd_free;
	    double  fs_level;
	    ostringstream  strConvert;
	    //---
	    struct sysinfo info;
	    sysinfo( &info );
	    //---
	    struct statvfs fsinfo;
	    statvfs("/", &fsinfo);
 */
	    //---
	    return(0);
}

