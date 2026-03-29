/* XMRig
 * Copyright (c) 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_BENCHSTATE_TEST_H
#define XMRIG_BENCHSTATE_TEST_H


#include "base/crypto/Algorithm.h"


#include <map>


namespace xmrig {


static const std::map<int, std::map<uint32_t, uint64_t> > hashCheck = {
    { Algorithm::RX_0, {
#       ifndef NDEBUG
        {    10000U, 0x4A597463865ACF0EULL },
        {    20000U, 0xC82B490C757DA738ULL },
#       endif
        {   250000U, 0x7D6054757BB08A63ULL },
        {   500000U, 0x96607546DE1F5ECCULL },
        {  1000000U, 0x898B6E0431C28A6BULL },
        {  2000000U, 0xEE9468F8B40926BCULL },
        {  3000000U, 0xC2BC5D11724813C0ULL },
        {  4000000U, 0x3A2C7B285B87F941ULL },
        {  5000000U, 0x3B5BD2C3A16B450EULL },
        {  6000000U, 0x5CD0602F20C5C7C4ULL },
        {  7000000U, 0x101DE939474B6812ULL },
        {  8000000U, 0x52B765A1B156C6ECULL },
        {  9000000U, 0x323935102AB6B45CULL },
        { 10000000U, 0xB5231262E2792B26ULL }
    }},
    { Algorithm::RX_V2, {
#       ifndef NDEBUG
        {    10000U, 0x57d2051d099613a4ULL },
        {    20000U, 0x0bae0155cc797f01ULL },
#       endif
        {   250000U, 0x18cf741a71484072ULL },
        {   500000U, 0xcd8c3e6ec31b2faeULL },
        {  1000000U, 0x88d6b8fb70cd479dULL },
        {  2000000U, 0x0e16828d236a1a63ULL },
        {  3000000U, 0x2739bdd0f25b83a6ULL },
        {  4000000U, 0x32f42d9006d2d34bULL },
        {  5000000U, 0x16d9c6286cb82251ULL },
        {  6000000U, 0x1f916ae19d6bcf07ULL },
        {  7000000U, 0x1f474f99a873948fULL },
        {  8000000U, 0x8d67e0ddf05476bbULL },
        {  9000000U, 0x3ebf37dcd5c4a215ULL },
        { 10000000U, 0x7efbddff3f30fb74ULL }
    }},
    { Algorithm::RX_WOW, {
#       ifndef NDEBUG
        {    10000U, 0x6B0918757100B338ULL },
        {    20000U, 0x0B55785C1837F41BULL },
#       endif
        {   250000U, 0xC7F712C9603E2603ULL },
        {   500000U, 0x21A0E5AAE6DA7D8DULL },
        {  1000000U, 0x0F3E5400B39EA96AULL },
        {  2000000U, 0x85944CCFA2752D1FULL },
        {  3000000U, 0x64AFFCAE991811BAULL },
        {  4000000U, 0x3E4D0B836D3B13BAULL },
        {  5000000U, 0xEB7417D621271166ULL },
        {  6000000U, 0x97FFE10C0949FFA5ULL },
        {  7000000U, 0x84CAC0F8879A4BA1ULL },
        {  8000000U, 0xA1B79F031DA2459FULL },
        {  9000000U, 0x9B65226DA873E65DULL },
        { 10000000U, 0x0F9E00C5A511C200ULL }
    }}
};


static const std::map<int, std::map<uint32_t, uint64_t> > hashCheck1T = {
    { Algorithm::RX_0, {
#       ifndef NDEBUG
        {    10000U, 0xADFC3A66F79BFE7FULL },
        {    20000U, 0x8ED578A60D55C0DBULL },
#       endif
        {   250000U, 0x90A15B799486F3EBULL },
        {   500000U, 0xAA83118FEE570F9AULL },
        {  1000000U, 0x3DF47B0A427C93D9ULL },
        {  2000000U, 0xED4D639B0AEB85C6ULL },
        {  3000000U, 0x2D4F9B4275A713C3ULL },
        {  4000000U, 0xA9EBE4888377F8D3ULL },
        {  5000000U, 0xB92F81851E180454ULL },
        {  6000000U, 0xFB9F98F63C2F1B7DULL },
        {  7000000U, 0x2CC3D7A779D5AB35ULL },
        {  8000000U, 0x2EEF833EA462F4B1ULL },
        {  9000000U, 0xC6D39EF59213A07CULL },
        { 10000000U, 0x95E6BAE68DD779CDULL }
    }},
    { Algorithm::RX_V2, {
#       ifndef NDEBUG
        {    10000, 0x90eb7c07cd9e0d90ULL },
        {    20000, 0x6523a3658d7d9930ULL },
#       endif
        {   250000, 0xf83b6d9d355ee5b1ULL },
        {   500000, 0xbea3c1bf1465e9abULL },
        {  1000000, 0x9e16f7cb56b366e1ULL },
        {  2000000, 0x3b5e671f47e15e55ULL },
        {  3000000, 0xec5819c180df03e2ULL },
        {  4000000, 0x19d31b498f86aad4ULL },
        {  5000000, 0x2487626c75cd12ccULL },
        {  6000000, 0xa323a25a5286c39aULL },
        {  7000000, 0xa123b100f3104dfcULL },
        {  8000000, 0x602db9d83bfa0ddcULL },
        {  9000000, 0x98da909e579765ddULL },
        { 10000000, 0x3a45b7247cec9895ULL }
    }},
    { Algorithm::RX_WOW, {
#       ifndef NDEBUG
        {    10000U, 0x9EC1B9B8C8C7F082ULL },
        {    20000U, 0xF1DA44FA2A20D730ULL },
#       endif
        {   250000U, 0x7B409F096C863207ULL },
        {   500000U, 0x70B7B80D15654216ULL },
        {  1000000U, 0x31301CC550306A59ULL },
        {  2000000U, 0x92F65E9E31116361ULL },
        {  3000000U, 0x7FE8DF6F43BA5285ULL },
        {  4000000U, 0xD6CDA54FE4D9BBF7ULL },
        {  5000000U, 0x73AF673E1A38E2B4ULL },
        {  6000000U, 0x81FDC5C4B45D84E4ULL },
        {  7000000U, 0xAA08CA57666DC874ULL },
        {  8000000U, 0x9DCEFB833FC875BCULL },
        {  9000000U, 0x862F051352CFCA1FULL },
        { 10000000U, 0xC403F220189E8430ULL }
    }}
};


} // namespace xmrig



#endif /* XMRIG_BENCHSTATE_TEST_H */
