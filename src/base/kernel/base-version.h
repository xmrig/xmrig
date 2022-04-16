/* XMRig
 * Copyright (c) 2018-2022 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2022 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_BASE_VERSION_H
#define XMRIG_BASE_VERSION_H

// The base version in the form major * 10000 + minor * 100 + patch.
#define XMRIG_BASE_VERSION 70000

#ifndef APP_DOMAIN
#   define APP_DOMAIN    "xmrig.com"
#endif

#ifndef APP_COPYRIGHT
#   define APP_COPYRIGHT "Copyright (C) 2016-2022 xmrig.com"
#endif

#define XMRIG_STRINGIFY(x) #x
#define XMRIG_TOSTRING(x) XMRIG_STRINGIFY(x)

#ifdef GIT_COMMIT_HASH
#   define XMRIG_GIT_COMMIT_HASH XMRIG_TOSTRING(GIT_COMMIT_HASH)
#else
#   define XMRIG_GIT_COMMIT_HASH "0000000"
#endif

#ifdef GIT_BRANCH
#   define XMRIG_GIT_BRANCH XMRIG_TOSTRING(GIT_BRANCH)
#   define APP_VERSION XMRIG_TOSTRING(APP_VER_MAJOR.APP_VER_MINOR.APP_VER_PATCH) "-" XMRIG_GIT_BRANCH
#else
#   define APP_VERSION XMRIG_TOSTRING(APP_VER_MAJOR.APP_VER_MINOR.APP_VER_PATCH)
#endif

#endif // XMRIG_BASE_VERSION_H
