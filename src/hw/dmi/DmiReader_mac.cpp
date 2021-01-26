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


#include <Carbon/Carbon.h>


namespace xmrig {


static int checksum(const uint8_t *buf, size_t len)
{
    uint8_t sum = 0;

    for (size_t a = 0; a < len; a++) {
        sum += buf[a];
    }

    return (sum == 0);
}


static uint8_t *dmi_table(uint32_t base, uint32_t &len, io_service_t service)
{
    CFMutableDictionaryRef properties = nullptr;
    if (IORegistryEntryCreateCFProperties(service, &properties, kCFAllocatorDefault, kNilOptions) != kIOReturnSuccess) {
        return nullptr;
    }

    CFDataRef data;
    uint8_t *buf = nullptr;

    if (CFDictionaryGetValueIfPresent(properties, CFSTR("SMBIOS"), (const void **)&data)) {
        assert(len == CFDataGetLength(data));

        len = CFDataGetLength(data);
        buf = reinterpret_cast<uint8_t *>(malloc(len));

        CFDataGetBytes(data, CFRangeMake(0, len), buf);
    }

    CFRelease(properties);

    return buf;
}


static uint8_t *smbios_decode(uint8_t *buf, uint32_t &size, uint32_t &version, io_service_t service)
{
    if (buf[0x05] > 0x20 || !checksum(buf, buf[0x05]) || memcmp(buf + 0x10, "_DMI_", 5) != 0 || !checksum(buf + 0x10, 0x0F))  {
        return nullptr;
    }

    version = ((buf[0x06] << 8) + buf[0x07]) << 8;
    size    = dmi_get<uint16_t>(buf + 0x16);

    return dmi_table(dmi_get<uint32_t>(buf + 0x18), size, service);
}

} // namespace xmrig


bool xmrig::DmiReader::read()
{
    mach_port_t port;
    IOMasterPort(MACH_PORT_NULL, &port);

    io_service_t service = IOServiceGetMatchingService(port, IOServiceMatching("AppleSMBIOS"));
    if (service == MACH_PORT_NULL) {
        return false;
    }

    CFDataRef data = reinterpret_cast<CFDataRef>(IORegistryEntryCreateCFProperty(service, CFSTR("SMBIOS-EPS"), kCFAllocatorDefault, kNilOptions));
    if (!data) {
        return false;
    }

    uint8_t buf[0x20]{};
    CFDataGetBytes(data, CFRangeMake(0, sizeof(buf)), buf);
    CFRelease(data);

    auto smb      = smbios_decode(buf, m_size, m_version, service);
    const bool rc = smb ? decode(smb) : false;

    IOObjectRelease(service);

    return rc;
}
