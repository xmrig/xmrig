/* XMRig
 * Copyright (c) 2000-2002 Alan Cox     <alan@redhat.com>
 * Copyright (c) 2005-2020 Jean Delvare <jdelvare@suse.de>
 * Copyright (c) 2018-2021 SChernykh    <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig        <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_DMITOOLS_H
#define XMRIG_DMITOOLS_H


#include <cstddef>
#include <cstdint>
#include "base/tools/Alignment.h"


namespace xmrig {


struct dmi_header
{
    uint8_t type;
    uint8_t length;
    uint16_t handle;
    uint8_t *data;
};


struct u64 {
    uint32_t l;
    uint32_t h;
};


template<typename T>
inline T dmi_get(const uint8_t *data)                   { return readUnaligned(reinterpret_cast<const T *>(data)); }

template<typename T>
inline T dmi_get(const dmi_header *h, size_t offset)    { return readUnaligned(reinterpret_cast<const T *>(h->data + offset)); }


const char *dmi_string(dmi_header *dm, size_t offset);


} /* namespace xmrig */


#endif /* XMRIG_DMITOOLS_H */
