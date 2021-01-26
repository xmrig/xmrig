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

    return decode(smb->SMBIOSTableData, [smb]() {
        HeapFree(GetProcessHeap(), 0, smb);
    });
}
