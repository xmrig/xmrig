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

#ifndef __POW_VARIANT_H__
#define __POW_VARIANT_H__

#include <string>
#include <list>

enum PowVariant
{
    POW_AUTODETECT,
    POW_V0,
    POW_V1,
    POW_IPBC,
    POW_ALLOY,
    POW_XTL,
    POW_MSR,
    POW_XHV,
    POW_RTO,
    LAST_ITEM
};

inline std::string getPowVariantName(PowVariant powVariant)
{
    switch (powVariant)
    {
        case POW_V0:
            return "0";
        case POW_V1:
            return "1";
        case POW_IPBC:
            return "ipbc";
        case POW_ALLOY:
            return "alloy";
        case POW_XTL:
            return "xtl";
        case POW_MSR:
            return "msr";
        case POW_XHV:
            return "xhv";
        case POW_RTO:
            return "rto";
        case POW_AUTODETECT:
        default:
            return "-1";
    }
}

inline std::list<std::string> getSupportedPowVariants()
{
    std::list<std::string> supportedPowVariants;

    for (int variant = 0; variant != LAST_ITEM; variant++)
    {
        supportedPowVariants.push_back(getPowVariantName(static_cast<PowVariant >(variant)));
    }

    return supportedPowVariants;
}

inline PowVariant parseVariant(int variant)
{
    PowVariant powVariant = PowVariant::POW_AUTODETECT;

    switch (variant) {
        case -1:
            powVariant = PowVariant::POW_AUTODETECT;
            break;
        case 0:
            powVariant = PowVariant::POW_V0;
            break;
        case 1:
            powVariant = PowVariant::POW_V1;
            break;
        default:
            break;
    }

    return powVariant;
}


inline PowVariant parseVariant(const std::string variant)
{
    PowVariant powVariant = PowVariant::POW_AUTODETECT;

    if (variant == "0") {
        powVariant = PowVariant::POW_V0;
    } else if (variant == "1") {
        powVariant = PowVariant::POW_V1;
    } else if (variant == "ipbc" || variant == "tube") {
        powVariant = PowVariant::POW_IPBC;
    } else if (variant == "xao" || variant == "alloy") {
        powVariant = PowVariant::POW_ALLOY;
    } else if (variant == "xtl" || variant == "stellite") {
        powVariant = PowVariant::POW_XTL;
    } else if (variant == "msr" || variant == "masari") {
        powVariant = PowVariant::POW_MSR;
    } else if (variant == "xhv" || variant == "haven") {
        powVariant = PowVariant::POW_XHV;
    } else if (variant == "rto" || variant == "arto") {
        powVariant = PowVariant::POW_RTO;
    }

    return powVariant;
}


#endif /* __POW_VARIANT_H__ */
