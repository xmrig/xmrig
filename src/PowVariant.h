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
    POW_V2,
    POW_TUBE,
    POW_ALLOY,
    POW_XTL,
    POW_MSR,
    POW_XHV,
    POW_RTO,
    POW_XFH,
    POW_FAST_2,
    POW_UPX,
    POW_TURTLE,
    POW_HOSP,
    POW_WOW,
    POW_V4,
    POW_DOUBLE,
    POW_ZELERIUS,
    POW_RWZ,
    POW_UPX2,
    POW_CONCEAL,
    POW_ARGON2_CHUKWA,
    POW_ARGON2_WRKZ,
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
        case POW_V2:
            return "2";
        case POW_TUBE:
            return "tube";
        case POW_ALLOY:
            return "xao";
        case POW_XTL:
            return "xtl";
        case POW_MSR:
            return "msr";
        case POW_XHV:
            return "xhv";
        case POW_RTO:
            return "rto";
        case POW_XFH:
            return "xfh";
        case POW_FAST_2:
            return "fast2";
        case POW_UPX:
            return "upx";
        case POW_TURTLE:
            return "turtle";
        case POW_HOSP:
            return "hosp";
        case POW_WOW:
            return "wow";
        case POW_V4:
            return "r";
        case POW_DOUBLE:
            return "double";
        case POW_ZELERIUS:
            return "zls";
        case POW_RWZ:
            return "rwz";
        case POW_UPX2:
            return "upx2";
        case POW_CONCEAL:
            return "conceal";
        case POW_ARGON2_CHUKWA:
            return "chukwa";
        case POW_ARGON2_WRKZ:
            return "wrkz";
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
        case 2:
            powVariant = PowVariant::POW_V2;
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
    } else if (variant == "2") {
        powVariant = PowVariant::POW_V2;
    } else if (variant == "ipbc" || variant == "tube" || variant == "bittube") {
        powVariant = PowVariant::POW_TUBE;
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
    } else if (variant == "xfh" || variant == "freehaven" || variant == "faven") {
        powVariant = PowVariant::POW_XFH;
    } else if (variant == "xtlv9" || variant == "stellite_v9" || variant == "xtlv2" || variant == "half" || variant == "msr2" || variant == "fast2") {
        powVariant = PowVariant::POW_FAST_2;
    } else if (variant == "upx" || variant == "uplexa" || variant == "cn-upx") {
        powVariant = PowVariant::POW_UPX;
    } else if (variant == "turtle" || variant == "trtl" || variant == "pico" || variant == "turtlev2") {
        powVariant = PowVariant::POW_TURTLE;
    } else if (variant == "hosp" || variant == "hospital") {
        powVariant = PowVariant::POW_HOSP;
    } else if (variant == "wow" || variant == "wownero") {
        powVariant = PowVariant::POW_WOW;
    } else if (variant == "r" || variant == "4" || variant == "cnv4" || variant == "cnv5") {
        powVariant = PowVariant::POW_V4;
    } else if (variant == "xcash" || variant == "heavyx" || variant == "double") {
        powVariant = PowVariant::POW_DOUBLE;
    } else if (variant == "zelerius" || variant == "zls" || variant == "zlx") {
        powVariant = PowVariant::POW_ZELERIUS;
    } else if (variant == "rwz" || variant == "graft") {
        powVariant = PowVariant::POW_RWZ;
    } else if (variant == "upx2") {
        powVariant = PowVariant::POW_UPX2;
    } else if (variant == "conceal" || variant == "ccx") {
        powVariant = PowVariant::POW_CONCEAL;
    } else if (variant == "chukwa" || variant == "trtl-chukwa" || variant == "argon2-chukwa") {
        powVariant = PowVariant::POW_ARGON2_CHUKWA;
    } else if (variant == "chukwa_wrkz" || variant == "wrkz" || variant == "argon2-wrkz") {
        powVariant = PowVariant::POW_ARGON2_WRKZ;
    }

    return powVariant;
}

inline PowVariant getCNBaseVariant(PowVariant powVariant)
{
    switch (powVariant)
    {
        case POW_V1:
        case POW_XTL:
        case POW_MSR:
        case POW_RTO:
        case POW_HOSP:
        case POW_UPX:
            return POW_V1;

        case POW_V2:
        case POW_TURTLE:
        case POW_DOUBLE:
        case POW_ZELERIUS:
        case POW_RWZ:
        case POW_UPX2:
        case POW_FAST_2:
            return POW_V2;

        case POW_WOW:
        case POW_V4:
            return POW_V4;
        default:
            return POW_V0;
    }
}


#endif /* __POW_VARIANT_H__ */
