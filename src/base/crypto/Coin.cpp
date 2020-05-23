/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include "base/crypto/Coin.h"
#include "3rdparty/rapidjson/document.h"


#include <cstring>


#ifdef _MSC_VER
#   define strcasecmp _stricmp
#endif


namespace xmrig {


struct CoinName
{
    const char *name;
    const Coin::Id id;
};


static CoinName const coin_names[] = {
    { "monero",     Coin::MONERO },
    { "xmr",        Coin::MONERO },
    { "arqma",      Coin::ARQMA  },
    { "arq",        Coin::ARQMA  },
    { "dero",       Coin::DERO   },
    { "keva",       Coin::KEVA   }
};


} /* namespace xmrig */



xmrig::Algorithm::Id xmrig::Coin::algorithm(uint8_t blobVersion) const
{
    switch (id()) {
    case MONERO:
        return (blobVersion >= 12) ? Algorithm::RX_0 : Algorithm::CN_R;

    case ARQMA:
        return (blobVersion >= 15) ? Algorithm::RX_ARQ : Algorithm::CN_PICO_0;

    case DERO:
        return (blobVersion >= 4) ? Algorithm::ASTROBWT_DERO : Algorithm::CN_0;

    case KEVA:
        return (blobVersion >= 11) ? Algorithm::RX_KEVA : Algorithm::CN_R;

    case INVALID:
        break;
    }

    return Algorithm::INVALID;
}



const char *xmrig::Coin::name() const
{
    for (const auto &i : coin_names) {
        if (i.id == m_id) {
            return i.name;
        }
    }

    return nullptr;
}


rapidjson::Value xmrig::Coin::toJSON() const
{
    using namespace rapidjson;

    return isValid() ? Value(StringRef(name())) : Value(kNullType);
}


xmrig::Coin::Id xmrig::Coin::parse(const char *name)
{
    if (name == nullptr || strlen(name) < 3) {
        return INVALID;
    }

    for (const auto &i : coin_names) {
        if (strcasecmp(name, i.name) == 0) {
            return i.id;
        }
    }

    return INVALID;
}
