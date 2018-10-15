/* XMRigCC
 * Copyright 2018-     BenDr0id    <ben@graef.in>
 *
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

#ifndef __ASM_OPTIMIZATION_H__
#define __ASM_OPTIMIZATION_H__

#include <string>
#include <algorithm>

enum AsmOptimization
{
    ASM_AUTODETECT,
    ASM_INTEL,
    ASM_RYZEN,
    ASM_NONE
};

inline std::string getAsmOptimizationName(AsmOptimization asmOptimization)
{
    switch (asmOptimization)
    {
        case ASM_INTEL:
            return "INTEL";
        case ASM_RYZEN:
            return "RYZEN";
        case ASM_NONE:
            return "OFF";
        case ASM_AUTODETECT:
        default:
            return "-1";
    }
}

inline AsmOptimization parseAsmOptimization(int optimization)
{
    AsmOptimization asmOptimization = AsmOptimization::ASM_AUTODETECT;

    switch (optimization) {
        case -1:
            asmOptimization = AsmOptimization::ASM_AUTODETECT;
            break;
        case 0:
            asmOptimization = AsmOptimization::ASM_NONE;
            break;
        case 1:
            asmOptimization = AsmOptimization::ASM_INTEL;
            break;
        case 2:
            asmOptimization = AsmOptimization::ASM_RYZEN;
            break;
        default:
            break;
    }

    return asmOptimization;
}

inline AsmOptimization parseAsmOptimization(const std::string optimization)
{
    AsmOptimization asmOptimization = AsmOptimization::ASM_AUTODETECT;

    if (optimization == "0" || optimization == "none" || optimization == "off") {
        asmOptimization = AsmOptimization::ASM_NONE;
    } else if (optimization == "1" || optimization == "intel") {
        asmOptimization = AsmOptimization::ASM_INTEL;
    } else if (optimization == "2" || optimization == "ryzen") {
        asmOptimization = AsmOptimization::ASM_RYZEN;
    }

    return asmOptimization;
}


#endif /* __ASM_OPTIMIZATION_H__ */
