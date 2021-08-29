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


#include "hw/dmi/DmiTools.h"


#include <cstring>


namespace xmrig {


/* Replace non-ASCII characters with dots */
static void ascii_filter(char *bp, size_t len)
{
    for (size_t i = 0; i < len; i++) {
        if (bp[i] < 32 || bp[i] >= 127) {
            bp[i] = '.';
        }
    }
}


static char *_dmi_string(dmi_header *dm, uint8_t s, bool filter)
{
    char *bp = reinterpret_cast<char *>(dm->data);

    bp += dm->length;
    while (s > 1 && *bp)  {
        bp += strlen(bp);
        bp++;
        s--;
    }

    if (!*bp) {
        return nullptr;
    }

    if (filter) {
        ascii_filter(bp, strlen(bp));
    }

    return bp;
}


const char *dmi_string(dmi_header *dm, size_t offset)
{
    if (offset < 4) {
        return nullptr;
    }

    return _dmi_string(dm, dm->data[offset], true);
}


} // namespace xmrig
