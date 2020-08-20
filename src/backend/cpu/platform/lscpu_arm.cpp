/* XMRig
 * Copyright 2018      Riku Voipio <riku.voipio@iki.fi>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <support@xmrig.com>
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


#include "base/tools/String.h"


#include <cstdio>
#include <cctype>


namespace xmrig {


struct lscpu_desc
{
    String vendor;
    String model;

    inline bool isReady() const { return !vendor.isNull() && !model.isNull(); }
};


struct id_part {
    const int id;
    const char *name;
};


struct hw_impl {
   const int id;
   const id_part *parts;
   const char *name;
};


static const id_part arm_part[] = {
    { 0x810, "ARM810" },
    { 0x920, "ARM920" },
    { 0x922, "ARM922" },
    { 0x926, "ARM926" },
    { 0x940, "ARM940" },
    { 0x946, "ARM946" },
    { 0x966, "ARM966" },
    { 0xa20, "ARM1020" },
    { 0xa22, "ARM1022" },
    { 0xa26, "ARM1026" },
    { 0xb02, "ARM11 MPCore" },
    { 0xb36, "ARM1136" },
    { 0xb56, "ARM1156" },
    { 0xb76, "ARM1176" },
    { 0xc05, "Cortex-A5" },
    { 0xc07, "Cortex-A7" },
    { 0xc08, "Cortex-A8" },
    { 0xc09, "Cortex-A9" },
    { 0xc0d, "Cortex-A17" }, /* Originally A12 */
    { 0xc0f, "Cortex-A15" },
    { 0xc0e, "Cortex-A17" },
    { 0xc14, "Cortex-R4" },
    { 0xc15, "Cortex-R5" },
    { 0xc17, "Cortex-R7" },
    { 0xc18, "Cortex-R8" },
    { 0xc20, "Cortex-M0" },
    { 0xc21, "Cortex-M1" },
    { 0xc23, "Cortex-M3" },
    { 0xc24, "Cortex-M4" },
    { 0xc27, "Cortex-M7" },
    { 0xc60, "Cortex-M0+" },
    { 0xd01, "Cortex-A32" },
    { 0xd03, "Cortex-A53" },
    { 0xd04, "Cortex-A35" },
    { 0xd05, "Cortex-A55" },
    { 0xd07, "Cortex-A57" },
    { 0xd08, "Cortex-A72" },
    { 0xd09, "Cortex-A73" },
    { 0xd0a, "Cortex-A75" },
    { 0xd0b, "Cortex-A76" },
    { 0xd0c, "Neoverse-N1" },
    { 0xd13, "Cortex-R52" },
    { 0xd20, "Cortex-M23" },
    { 0xd21, "Cortex-M33" },
    { 0xd4a, "Neoverse-E1" },
    { -1, nullptr },
};

static const id_part brcm_part[] = {
    { 0x0f, "Brahma B15" },
    { 0x100, "Brahma B53" },
    { 0x516, "ThunderX2" },
    { -1, nullptr },
};

static const id_part dec_part[] = {
    { 0xa10, "SA110" },
    { 0xa11, "SA1100" },
    { -1, nullptr },
};

static const id_part cavium_part[] = {
    { 0x0a0, "ThunderX" },
    { 0x0a1, "ThunderX 88XX" },
    { 0x0a2, "ThunderX 81XX" },
    { 0x0a3, "ThunderX 83XX" },
    { 0x0af, "ThunderX2 99xx" },
    { -1, nullptr },
};

static const id_part apm_part[] = {
    { 0x000, "X-Gene" },
    { -1, nullptr },
};

static const id_part qcom_part[] = {
    { 0x00f, "Scorpion" },
    { 0x02d, "Scorpion" },
    { 0x04d, "Krait" },
    { 0x06f, "Krait" },
    { 0x201, "Kryo" },
    { 0x205, "Kryo" },
    { 0x211, "Kryo" },
    { 0x800, "Falkor V1/Kryo" },
    { 0x801, "Kryo V2" },
    { 0xc00, "Falkor" },
    { 0xc01, "Saphira" },
    { -1, nullptr },
};

static const id_part samsung_part[] = {
    { 0x001, "exynos-m1" },
    { -1, nullptr },
};

static const id_part nvidia_part[] = {
    { 0x000, "Denver" },
    { 0x003, "Denver 2" },
    { -1, nullptr },
};

static const id_part marvell_part[] = {
    { 0x131, "Feroceon 88FR131" },
    { 0x581, "PJ4/PJ4b" },
    { 0x584, "PJ4B-MP" },
    { -1, nullptr },
};

static const id_part faraday_part[] = {
    { 0x526, "FA526" },
    { 0x626, "FA626" },
    { -1, nullptr },
};

static const id_part intel_part[] = {
    { 0x200, "i80200" },
    { 0x210, "PXA250A" },
    { 0x212, "PXA210A" },
    { 0x242, "i80321-400" },
    { 0x243, "i80321-600" },
    { 0x290, "PXA250B/PXA26x" },
    { 0x292, "PXA210B" },
    { 0x2c2, "i80321-400-B0" },
    { 0x2c3, "i80321-600-B0" },
    { 0x2d0, "PXA250C/PXA255/PXA26x" },
    { 0x2d2, "PXA210C" },
    { 0x411, "PXA27x" },
    { 0x41c, "IPX425-533" },
    { 0x41d, "IPX425-400" },
    { 0x41f, "IPX425-266" },
    { 0x682, "PXA32x" },
    { 0x683, "PXA930/PXA935" },
    { 0x688, "PXA30x" },
    { 0x689, "PXA31x" },
    { 0xb11, "SA1110" },
    { 0xc12, "IPX1200" },
    { -1, nullptr },
};

static const id_part hisi_part[] = {
    { 0xd01, "Kunpeng-920" }, /* aka tsv110 */
    { -1, nullptr },
};


static const hw_impl hw_implementer[] = {
    { 0x41, arm_part,     "ARM" },
    { 0x42, brcm_part,    "Broadcom" },
    { 0x43, cavium_part,  "Cavium" },
    { 0x44, dec_part,     "DEC" },
    { 0x48, hisi_part,    "HiSilicon" },
    { 0x4e, nvidia_part,  "Nvidia" },
    { 0x50, apm_part,     "APM" },
    { 0x51, qcom_part,    "Qualcomm" },
    { 0x53, samsung_part, "Samsung" },
    { 0x56, marvell_part, "Marvell" },
    { 0x66, faraday_part, "Faraday" },
    { 0x69, intel_part,   "Intel" }
};


static bool lookup(char *line, const char *pattern, String &value)
{
    if (!*line || !value.isNull()) {
        return false;
    }

    char *p;
    int len = strlen(pattern);

    if (strncmp(line, pattern, len) != 0) {
        return false;
    }

    for (p = line + len; isspace(*p); p++);

    if (*p != ':') {
        return false;
    }

    for (++p; isspace(*p); p++);

    if (!*p) {
        return false;
    }

    const char *v = p;

    len = strlen(line) - 1;
    for (p = line + len; isspace(*(p-1)); p--);
    *p = '\0';

    value = v;

    return true;
}


static bool read_basicinfo(lscpu_desc *desc)
{
    auto fp = fopen("/proc/cpuinfo", "r");
    if (!fp) {
        return false;
    }

    char buf[BUFSIZ];
    while (fgets(buf, sizeof(buf), fp) != nullptr) {
        if (!lookup(buf, "CPU implementer", desc->vendor)) {
            lookup(buf, "CPU part", desc->model);
        }

        if (desc->isReady()) {
            break;
        }
    }

    fclose(fp);

    return desc->isReady();
}


static bool arm_cpu_decode(lscpu_desc *desc)
{
    if ((strncmp(desc->vendor, "0x", 2) != 0 || strncmp(desc->model, "0x", 2) != 0)) {
        return false;
    }

    const int vendor = strtol(desc->vendor, nullptr, 0);
    const int model  = strtol(desc->model, nullptr, 0);

    for (const auto &impl : hw_implementer) {
        if (impl.id != vendor) {
            continue;
        }

        for (size_t i = 0; impl.parts[i].id != -1; ++i) {
            if (impl.parts[i].id == model) {
                desc->model = impl.parts[i].name;

                return true;
            }
        }
    }

    return false;
}


String cpu_name_arm()
{
    lscpu_desc desc;
    if (read_basicinfo(&desc) && arm_cpu_decode(&desc)) {
        return desc.model;
    }

    return {};
}


} // namespace xmrig
