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


#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>


#define FLAG_NO_FILE_OFFSET     (1 << 0)


namespace xmrig {


static const char *kSysEntryFile = "/sys/firmware/dmi/tables/smbios_entry_point";
static const char *kSysTableFile = "/sys/firmware/dmi/tables/DMI";



static int myread(int fd, uint8_t *buf, size_t count, const char *prefix)
{
    ssize_t r = 1;
    size_t r2 = 0;

    while (r2 != count && r != 0) {
        r = read(fd, buf + r2, count - r2);
        if (r == -1) {
            if (errno != EINTR) {
                return -1;
            }
        }
        else {
            r2 += r;
        }
    }

    if (r2 != count) {
        return -1;
    }

    return 0;
}


/*
 * Reads all of file from given offset, up to max_len bytes.
 * A buffer of at most max_len bytes is allocated by this function, and
 * needs to be freed by the caller.
 * This provides a similar usage model to mem_chunk()
 *
 * Returns a pointer to the allocated buffer, or NULL on error, and
 * sets max_len to the length actually read.
 */
static uint8_t *read_file(off_t base, size_t *max_len, const char *filename)
{
    const int fd = open(filename, O_RDONLY);
    uint8_t *p   = nullptr;

    if (fd == -1) {
        return nullptr;
    }

    struct stat statbuf{};
    if (fstat(fd, &statbuf) == 0) {
        if (base >= statbuf.st_size) {
            goto out;
        }

        if (*max_len > static_cast<size_t>(statbuf.st_size) - base) {
            *max_len = statbuf.st_size - base;
        }
    }

    if ((p = reinterpret_cast<uint8_t *>(malloc(*max_len))) == nullptr) {
        goto out;
    }

    if (lseek(fd, base, SEEK_SET) == -1) {
        goto err_free;
    }

    if (myread(fd, p, *max_len, filename) == 0) {
        goto out;
    }

err_free:
    free(p);
    p = nullptr;

out:
    close(fd);

    return p;
}


static int checksum(const uint8_t *buf, size_t len)
{
    uint8_t sum = 0;

    for (size_t a = 0; a < len; a++) {
        sum += buf[a];
    }

    return (sum == 0);
}


static uint8_t *dmi_table(off_t base, uint32_t &len, const char *devmem, uint32_t flags)
{
    uint8_t *buf = nullptr;

    if (flags & FLAG_NO_FILE_OFFSET) {
        size_t size = len;
        buf         = read_file(0, &size, devmem);
        len         = size;
    }
    else {
        // FIXME
    }

    return buf;
}


static uint8_t *smbios3_decode(uint8_t *buf, const char *devmem, uint32_t &size, uint32_t &version, uint32_t flags)
{
    if (buf[0x06] > 0x20 || !checksum(buf, buf[0x06])) {
        return nullptr;
    }

    version          = (buf[0x07] << 16) + (buf[0x08] << 8) + buf[0x09];
    size             = dmi_get<uint32_t>(buf + 0x0C);
    const u64 offset = dmi_get<u64>(buf + 0x10);

    return dmi_table(((off_t)offset.h << 32) | offset.l, size, devmem, flags);;
}


} // namespace xmrig


bool xmrig::DmiReader::read()
{
    size_t size  = 0x20;
    uint8_t *buf = read_file(0, &size, kSysEntryFile);

    if (buf) {
        uint8_t *smb = nullptr;

        if (size >= 24 && memcmp(buf, "_SM3_", 5) == 0) {
            smb = smbios3_decode(buf, kSysTableFile, m_size, m_version, FLAG_NO_FILE_OFFSET);
        }

        if (smb) {
            decode(smb);

            free(smb);
            free(buf);

            return true;
        }
    }

    return false;
}
