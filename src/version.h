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

#ifndef XMRIG_VERSION_H
#define XMRIG_VERSION_H

#define APP_ID        "xmrig"
#define APP_NAME      "XMRig"
#define APP_DESC      "XMRig miner"
#define APP_VERSION   "6.15.3-dev"
#define APP_DOMAIN    "xmrig.com"
#define APP_SITE      "www.xmrig.com"
#define APP_COPYRIGHT "Copyright (C) 2016-2021 xmrig.com"
#define APP_KIND      "miner"

#define APP_VER_MAJOR  6
#define APP_VER_MINOR  15
#define APP_VER_PATCH  3

#ifdef _MSC_VER
#   if (_MSC_VER >= 1920)
#       define MSVC_VERSION 2019
#   elif (_MSC_VER >= 1910 && _MSC_VER < 1920)
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
#endif

#if defined(_MSC_FULL_VER)
#   define MSVC_VERSION_MAJOR ((_MSC_FULL_VER / 10000000) >> 0)
#   define MSVC_VERSION_MINOR (((_MSC_FULL_VER - (MSVC_VERSION_MAJOR * 10000000)) / 100000) >> 0)
#   define MSVC_VERSION_PATCH (_MSC_FULL_VER - (MSVC_VERSION_MAJOR * 10000000) - (MSVC_VERSION_MINOR * 100000))
#   define MSVC_VERSION_BUILD _MSC_BUILD
#endif

#if defined(__INTEL_COMPILER)
#   define __INTELC_MAJOR__ ((__INTEL_COMPILER / 100) >> 0)
#   define __INTELC_MINOR__ (((__INTEL_COMPILER - (__INTELC_MAJOR__ * 100)) / 10) >> 0)
#   define __INTELC_PATCH__ (__INTEL_COMPILER - (__INTELC_MAJOR__ * 100) - (__INTELC_MINOR__ * 10))
#endif

#endif /* XMRIG_VERSION_H */
