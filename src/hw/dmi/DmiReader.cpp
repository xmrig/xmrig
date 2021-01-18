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


#include "hw/dmi/DmiReader.h"
#include "hw/dmi/DmiTools.h"


namespace xmrig {


static void dmi_get_header(dmi_header *h, uint8_t *data)
{
    h->type   = data[0];
    h->length = data[1];
    h->handle = dmi_get<uint16_t>(data + 2);
    h->data   = data;
}


} // namespace xmrig


bool xmrig::DmiReader::decode(uint8_t *buf)
{
    if (!buf) {
        return false;
    }

    uint8_t *data = buf;
    int i         = 0;

    while (data + 4 <= buf + m_size) {
        dmi_header h{};
        dmi_get_header(&h, data);

        if (h.length < 4 || h.type == 127) {
            break;
        }
        i++;

        uint8_t *next = data + h.length;
        while (static_cast<uint32_t>(next - buf + 1) < m_size && (next[0] != 0 || next[1] != 0)) {
            next++;
        }
        next += 2;

        if (static_cast<uint32_t>(next - buf) > m_size) {
            data = next;
            break;
        }

        switch (h.type) {
        case 2:
            m_board.decode(&h);
            break;

        case 17:
            m_memory.emplace_back(&h);
            break;

        default:
            break;
        }

        data = next;
    }

    return true;
}
