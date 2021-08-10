/* XMRig
 * Copyright 2012-2013 The Cryptonote developers
 * Copyright 2014-2021 The Monero Project
 * Copyright 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_WALLETADDRESS_H
#define XMRIG_WALLETADDRESS_H


#include "base/tools/String.h"


namespace xmrig {


struct WalletAddress
{
    uint64_t tag;
    uint8_t public_spend_key[32];
    uint8_t public_view_key[32];
    uint8_t checksum[4];

    bool Decode(const String& address);
};


} /* namespace xmrig */


#endif /* XMRIG_WALLETADDRESS_H */
