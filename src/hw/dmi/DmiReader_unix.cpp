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
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstdio>

#ifdef __FreeBSD__
#   include <kenv.h>
#endif


#define FLAG_NO_FILE_OFFSET     (1 << 0)


namespace xmrig {


static const char *kMemDevice       = "/dev/mem";
static const char *kSysEntryFile    = "/sys/firmware/dmi/tables/smbios_entry_point";
static const char *kSysTableFile    = "/sys/firmware/dmi/tables/DMI";


static inline void safe_memcpy(void *dest, const void *src, size_t n)
{
#   ifdef XMRIG_ARM
    for (size_t i = 0; i < n; i++) {
        *((uint8_t *)dest + i) = *((const uint8_t *)src + i);
    }
#   else
    memcpy(dest, src, n);
#   endif
}



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


/*
 * Copy a physical memory chunk into a memory buffer.
 * This function allocates memory.
 */
static uint8_t *mem_chunk(off_t base, size_t len, const char *devmem)
{
    const int fd = open(devmem, O_RDONLY);
    uint8_t *p   = nullptr;
    uint8_t *mmp = nullptr;
    struct stat statbuf{};

#   ifdef _SC_PAGESIZE
    const off_t mmoffset = base % sysconf(_SC_PAGESIZE);
#   else
    const off_t mmoffset = base % getpagesize();
#   endif

    if (fd == -1) {
        return nullptr;
    }

    if ((p = reinterpret_cast<uint8_t *>(malloc(len))) == nullptr) {
        goto out;
    }

    if (fstat(fd, &statbuf) == -1) {
        goto err_free;
    }

    if (S_ISREG(statbuf.st_mode) && base + (off_t)len > statbuf.st_size) {
        goto err_free;
    }

    mmp = reinterpret_cast<uint8_t *>(mmap(nullptr, mmoffset + len, PROT_READ, MAP_SHARED, fd, base - mmoffset));
    if (mmp == MAP_FAILED) {
        goto try_read;
    }

    safe_memcpy(p, mmp + mmoffset, len);
    munmap(mmp, mmoffset + len);

    goto out;

try_read:
    if (lseek(fd, base, SEEK_SET) == -1) {
        goto err_free;
    }

    if (myread(fd, p, len, devmem) == 0) {
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
    if (flags & FLAG_NO_FILE_OFFSET) {
        size_t size = len;
        auto buf    = read_file(0, &size, devmem);
        len         = size;

        return buf;
    }

    return mem_chunk(base, len, devmem);
}


static uint8_t *smbios3_decode(uint8_t *buf, const char *devmem, uint32_t &size, uint32_t &version, uint32_t flags)
{
    if (buf[0x06] > 0x20 || !checksum(buf, buf[0x06])) {
        return nullptr;
    }

    version          = (buf[0x07] << 16) + (buf[0x08] << 8) + buf[0x09];
    size             = dmi_get<uint32_t>(buf + 0x0C);
    const u64 offset = dmi_get<u64>(buf + 0x10);

    return dmi_table(((off_t)offset.h << 32) | offset.l, size, devmem, flags);
}


static uint8_t *smbios_decode(uint8_t *buf, const char *devmem, uint32_t &size, uint32_t &version, uint32_t flags)
{
    if (buf[0x05] > 0x20 || !checksum(buf, buf[0x05]) || memcmp(buf + 0x10, "_DMI_", 5) != 0 || !checksum(buf + 0x10, 0x0F))  {
        return nullptr;
    }

    version = (buf[0x06] << 8) + buf[0x07];

    switch (version) {
    case 0x021F:
    case 0x0221:
        version = 0x0203;
        break;

    case 0x0233:
        version = 0x0206;
        break;
    }

    version = version << 8;
    size    = dmi_get<uint16_t>(buf + 0x16);

    return dmi_table(dmi_get<uint32_t>(buf + 0x18), size, devmem, flags);
}


static uint8_t *legacy_decode(uint8_t *buf, const char *devmem, uint32_t &size, uint32_t &version, uint32_t flags)
{
    if (!checksum(buf, 0x0F)) {
        return nullptr;
    }

    version = ((buf[0x0E] & 0xF0) << 12) + ((buf[0x0E] & 0x0F) << 8);
    size    = dmi_get<uint16_t>(buf + 0x06);

    return dmi_table(dmi_get<uint32_t>(buf + 0x08), size, devmem, flags);
}


#define EFI_NOT_FOUND   (-1)
#define EFI_NO_SMBIOS   (-2)
static off_t address_from_efi()
{
#   if defined(__linux__)
    FILE *efi_systab;
    const char *filename;
    char linebuf[64];
    off_t address = 0;
#   elif defined(__FreeBSD__)
    char addrstr[KENV_MVALLEN + 1];
#   endif

#   if defined(__linux__)
    if ((efi_systab = fopen(filename = "/sys/firmware/efi/systab", "r")) == nullptr && (efi_systab = fopen(filename = "/proc/efi/systab", "r")) == nullptr) {
        return EFI_NOT_FOUND;
    }

    address = EFI_NO_SMBIOS;
    while ((fgets(linebuf, sizeof(linebuf) - 1, efi_systab)) != nullptr) {
        char *addrp = strchr(linebuf, '=');
        *(addrp++) = '\0';
        if (strcmp(linebuf, "SMBIOS3") == 0 || strcmp(linebuf, "SMBIOS") == 0) {
            address = strtoull(addrp, nullptr, 0);
            break;
        }
    }

    fclose(efi_systab);

    return address;
#   elif defined(__FreeBSD__)
    if (kenv(KENV_GET, "hint.smbios.0.mem", addrstr, sizeof(addrstr)) == -1) {
        return EFI_NOT_FOUND;
    }

    return strtoull(addrstr, nullptr, 0);
#   endif

    return EFI_NOT_FOUND;
}


} // namespace xmrig


bool xmrig::DmiReader::read()
{
    size_t size  = 0x20;
    uint8_t *buf = read_file(0, &size, kSysEntryFile);
    uint8_t *smb = nullptr;

    if (buf) {
        smb = nullptr;

        if (size >= 24 && memcmp(buf, "_SM3_", 5) == 0) {
            smb = smbios3_decode(buf, kSysTableFile, m_size, m_version, FLAG_NO_FILE_OFFSET);
        }
        else if (size >= 31 && memcmp(buf, "_SM_", 4) == 0) {
            smb = smbios_decode(buf, kSysTableFile, m_size, m_version, FLAG_NO_FILE_OFFSET);
        }
        else if (size >= 15 && memcmp(buf, "_DMI_", 5) == 0) {
            smb = legacy_decode(buf, kSysTableFile, m_size, m_version, FLAG_NO_FILE_OFFSET);
        }

        if (smb) {
            return decode(smb, [smb, buf]() { free(smb); free(buf); });
        }

        free(buf);
    }

    const auto efi = address_from_efi();
    if (efi == EFI_NO_SMBIOS) {
        return false;
    }

    if (efi != EFI_NOT_FOUND) {
        if ((buf = mem_chunk(efi, 0x20, kMemDevice)) == nullptr) {
            return false;
        }

        smb = nullptr;

        if (memcmp(buf, "_SM3_", 5) == 0) {
            smb = smbios3_decode(buf, kMemDevice, m_size, m_version, 0);
        }
        else if (memcmp(buf, "_SM_", 4) == 0) {
            smb = smbios_decode(buf, kSysTableFile, m_size, m_version, 0);
        }

        if (smb) {
            return decode(smb, [smb, buf]() { free(smb); free(buf); });
        }

        free(buf);
    }

#   if defined(__x86_64__) || defined(_M_AMD64)
    if ((buf = mem_chunk(0xF0000, 0x10000, kMemDevice)) == nullptr) {
        return false;
    }

    smb = nullptr;

    for (off_t fp = 0; fp <= 0xFFE0; fp += 16) {
        if (memcmp(buf + fp, "_SM3_", 5) == 0) {
            smb = smbios3_decode(buf + fp, kMemDevice, m_size, m_version, 0);
        }

        if (smb) {
            return decode(smb, [smb, buf]() { free(smb); free(buf); });
        }
    }

    for (off_t fp = 0; fp <= 0xFFF0; fp += 16) {
        if (memcmp(buf + fp, "_SM_", 4) == 0 && fp <= 0xFFE0) {
            smb = smbios3_decode(buf + fp, kMemDevice, m_size, m_version, 0);
        }
        else if (!smb && memcmp(buf + fp, "_DMI_", 5) == 0) {
            smb = legacy_decode(buf + fp, kMemDevice, m_size, m_version, 0);
        }

        if (smb) {
            return decode(smb, [smb, buf]() { free(smb); free(buf); });
        }
    }

    free(buf);
#   endif


    return false;
}
