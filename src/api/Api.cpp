/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
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


#include <cstring>

#include "api/Api.h"
#include "api/ApiState.h"


ApiState *Api::m_state = nullptr;
char Api::m_buf[4096];
uv_mutex_t Api::m_mutex;


bool Api::start()
{
    uv_mutex_init(&m_mutex);
    m_state = new ApiState();

    return true;
}


void Api::release()
{
    delete m_state;
}


const char *Api::get(const char *url, size_t *size, int *status)
{
    if (!m_state) {
        *size = 0;
        return nullptr;
    }

    uv_mutex_lock(&m_mutex);

    const char *buf = m_state->get(url, size);
    if (*size) {
        memcpy(m_buf, buf, *size);
    }
    else {
        *status = 500;
    }

    uv_mutex_unlock(&m_mutex);

    return m_buf;
}


void Api::tick(const Hashrate *hashrate)
{
    if (!m_state) {
        return;
    }

    uv_mutex_lock(&m_mutex);
    m_state->tick(hashrate);
    uv_mutex_unlock(&m_mutex);
}


void Api::tick(const NetworkState &network)
{
    if (!m_state) {
        return;
    }

    uv_mutex_lock(&m_mutex);
    m_state->tick(network);
    uv_mutex_unlock(&m_mutex);
}
