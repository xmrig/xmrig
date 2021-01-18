/* XMRig
 * Copyright (c) 2002-2006 Hugo Weber  <address@hidden>
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


#include "hw/dmi/DmiReader.h"
#include "hw/dmi/DmiTools.h"


#include <windows.h>


namespace xmrig {


/*
 * Struct needed to get the SMBIOS table using GetSystemFirmwareTable API.
 */
struct RawSMBIOSData {
    uint8_t	Used20CallingMethod;
    uint8_t	SMBIOSMajorVersion;
    uint8_t	SMBIOSMinorVersion;
    uint8_t	DmiRevision;
    uint32_t Length;
    uint8_t	SMBIOSTableData[];
};


/*
 * Counts the number of SMBIOS structures present in
 * the SMBIOS table.
 *
 * buf - Pointer that receives the SMBIOS Table address.
 *       This will be the address of the BYTE array from
 *       the RawSMBIOSData struct.
 *
 * len - The length of the SMBIOS Table pointed by buff.
 *
 * return - The number of SMBIOS strutctures.
 *
 * Remarks:
 * The SMBIOS Table Entry Point has this information,
 * however the GetSystemFirmwareTable API doesn't
 * return all fields from the Entry Point, and
 * DMIDECODE uses this value as a parameter for
 * dmi_table function. This is the reason why
 * this function was make.
 *
 * Hugo Weber address@hidden
 */
uint16_t count_smbios_structures(const uint8_t *buf, uint32_t len)
{
    uint16_t count  = 0;
    uint32_t offset = 0;

    while (offset < len) {
        offset += reinterpret_cast<const dmi_header *>(buf + offset)->length;
        count++;

        while ((*reinterpret_cast<const uint16_t *>(buf + offset) != 0) && (offset < len)) {
            offset++;
        }

        offset += 2;
    }

    return count;
}


} // namespace xmrig


bool xmrig::DmiReader::read()
{
    const uint32_t size = GetSystemFirmwareTable('RSMB', 0, nullptr, 0);
    auto smb            = reinterpret_cast<RawSMBIOSData *>(HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, size));

    if (!smb) {
        return false;
    }

    if (GetSystemFirmwareTable('RSMB', 0, smb, size) != size) {
        HeapFree(GetProcessHeap(), 0, smb);

        return false;
    }

    m_version = (smb->SMBIOSMajorVersion << 16) + (smb->SMBIOSMinorVersion << 8) + smb->DmiRevision;
    m_size    = smb->Length;
    m_count   = count_smbios_structures(smb->SMBIOSTableData, m_size);

    const bool rc = decode(smb->SMBIOSTableData);
    HeapFree(GetProcessHeap(), 0, smb);

    return rc;
}
