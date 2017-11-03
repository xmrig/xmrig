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
#define APP_ID        "xmrigCC"
#define APP_NAME      "XMRigCC"
#define APP_DESC      "XMRigCC Command'n'Control Server"
#define APP_VERSION   "1.0.7"
#define APP_COPYRIGHT "Copyright (C) 2017- BenDr0id"
# else
#define APP_ID        "xmrigCC"
#define APP_NAME      "XMRigCC"
#define APP_DESC      "XMRigCC CPU miner"
#define APP_VERSION   "2.4.2"
#define APP_COPYRIGHT "Copyright (C) 2017- BenDr0id"
#endif
#define APP_DOMAIN    ""
#define APP_SITE      "https://github.com/Bendr0id/xmrigCC"
#define APP_KIND      "cpu"

#define APP_VER_MAJOR  2
#define APP_VER_MINOR  4
#define APP_VER_BUILD  2
#define APP_VER_REV    0

#ifdef _MSC_VER
#   if (_MSC_VER == 1910 || _MSC_VER == 1911)
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

#endif /* __VERSION_H__ */
