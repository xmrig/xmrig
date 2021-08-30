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

#ifndef XMRIG_BASE_VERSION_H
#define XMRIG_BASE_VERSION_H


#define BASE_VER_MAJOR  0
#define BASE_VER_MINOR  1
#define BASE_VER_PATCH  0

#define XMRIG_STRINGIFY(x) #x
#define XMRIG_TOSTRING(x) XMRIG_STRINGIFY(x)

#define BASE_VERSION XMRIG_TOSTRING(BASE_VER_MAJOR.BASE_VER_MINOR.BASE_VER_PATCH)


#endif // XMRIG_BASE_VERSION_H
