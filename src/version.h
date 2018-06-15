/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
 * Copyright 2017-     BenDr0id    <ben@graef.in>
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

#ifndef __VERSION_H__
#define __VERSION_H__

#ifdef XMRIG_CC_SERVER
#define APP_ID        "XMRigCC"
#define APP_NAME      "XMRigCC"
#define APP_DESC      "XMRigCC Command'n'Control Server"
#define APP_COPYRIGHT "Copyright (C) 2017- BenDr0id"
# else
#define APP_ID        "XMRigCC"
#define APP_NAME      "XMRigCC"
#define APP_DESC      "XMRigCC CPU miner"
#define APP_COPYRIGHT "Copyright (C) 2017- BenDr0id"
#endif
#define APP_VERSION   "1.6.4 (based on XMRig)"
#define APP_DOMAIN    ""
#define APP_SITE      "https://github.com/Bendr0id/xmrigCC"
#define APP_KIND      "cpu"

#define APP_VER_MAJOR  1
#define APP_VER_MINOR  6
#define APP_VER_BUILD  4
#define APP_VER_REV    0

#ifndef NDEBUG
	#ifndef XMRIG_NO_TLS
		#define BUILD_TYPE   "DEBUG with TLS"
	#else
		#define BUILD_TYPE   "DEBUG"	
	#endif	
#else
	#ifndef XMRIG_NO_TLS
		#define BUILD_TYPE   "RELEASE with TLS"
	#else
		#define BUILD_TYPE   "RELEASE"	
	#endif	
#endif

#ifdef _MSC_VER
#   if (_MSC_VER >= 1910)
#       define MSVC_VERSION 2017
#   elif _MSC_VER == 1900
#       define MSVC_VERSION 2015
#   elif _MSC_VER == 1800
#       define MSVC_VERSION 2013
#   elif _MSC_VER == 1700
#       define MSVC_VERSION 2012
#   elif _MSC_VER == 1600
#       define MSVC_VERSION 2010
#   else
#       define MSVC_VERSION 0
#   endif
#include <string>
#else
    #if defined(__FreeBSD__) || defined(__APPLE__)
        #include <string>
    #else
        #include <string.h>
    #endif
#endif




class Version
{
public:
    inline static std::string string()
    {
        std::string version = std::to_string(APP_VER_MAJOR) + std::string(".") + std::to_string(APP_VER_MINOR) +
                              std::string(".") + std::to_string(APP_VER_BUILD);

        return version;
    }

    inline static int code()
    {
        std::string version = std::to_string(APP_VER_MAJOR) + std::to_string(APP_VER_MINOR) + std::to_string(APP_VER_BUILD);

        return std::stoi(version);
    }
};
#endif /* __VERSION_H__ */
