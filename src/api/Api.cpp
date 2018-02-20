/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2018 XMRig       <support@xmrig.com>
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

#include <string.h>


#include "api/Api.h"
#include "api/ApiState.h"


ApiState *Api::m_state = nullptr;


bool Api::start()
{
    m_state = new ApiState();

    return true;
}


void Api::release()
{
    delete m_state;
}


char *Api::get(const char *url, int *status)
{
    if (!m_state) {
        return nullptr;
    }

    return m_state->get(url, status);
}


void Api::tick(const Hashrate *hashrate)
{
    if (!m_state) {
        return;
    }

    m_state->tick(hashrate);
}


void Api::tick(const NetworkState &network)
{
    if (!m_state) {
        return;
    }

    m_state->tick(network);
}
