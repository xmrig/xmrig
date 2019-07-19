/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018      SChernykh   <https://github.com/SChernykh>
 * Copyright 2019      Howard Chu  <https://github.com/hyc>
 * Copyright 2016-2019 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include <assert.h>
#include <string.h>


#include "base/net/stratum/Job.h"
#include "base/tools/Buffer.h"


xmrig::Job::Job() :
    m_blob(),
    m_seedHash()
{
}


xmrig::Job::Job(bool nicehash, const Algorithm &algorithm, const String &clientId) :
    m_algorithm(algorithm),
    m_nicehash(nicehash),
    m_clientId(clientId),
    m_blob(),
    m_seedHash()
{
}


xmrig::Job::~Job()
{
}


bool xmrig::Job::isEqual(const Job &other) const
{
    return m_id == other.m_id && m_clientId == other.m_clientId && memcmp(m_blob, other.m_blob, sizeof(m_blob)) == 0;
}


bool xmrig::Job::setBlob(const char *blob)
{
    if (!blob) {
        return false;
    }

    m_size = strlen(blob);
    if (m_size % 2 != 0) {
        return false;
    }

    m_size /= 2;
    if (m_size < 76 || m_size >= sizeof(m_blob)) {
        return false;
    }

    if (!Buffer::fromHex(blob, m_size * 2, m_blob)) {
        return false;
    }

    if (*nonce() != 0 && !m_nicehash) {
        m_nicehash = true;
    }

#   ifdef XMRIG_PROXY_PROJECT
    memset(m_rawBlob, 0, sizeof(m_rawBlob));
    memcpy(m_rawBlob, blob, m_size * 2);
#   endif

    return true;
}


bool xmrig::Job::setSeedHash(const char *hash)
{
    if (!hash || (strlen(hash) != sizeof(m_seedHash) * 2)) {
        return false;
    }

#   ifdef XMRIG_PROXY_PROJECT
    m_rawSeedHash = hash;
#   endif

    return Buffer::fromHex(hash, sizeof(m_seedHash) * 2, m_seedHash);
}


bool xmrig::Job::setTarget(const char *target)
{
    if (!target) {
        return false;
    }

    const size_t len = strlen(target);

    if (len <= 8) {
        uint32_t tmp = 0;
        char str[8];
        memcpy(str, target, len);

        if (!Buffer::fromHex(str, 8, reinterpret_cast<uint8_t *>(&tmp)) || tmp == 0) {
            return false;
        }

        m_target = 0xFFFFFFFFFFFFFFFFULL / (0xFFFFFFFFULL / static_cast<uint64_t>(tmp));
    }
    else if (len <= 16) {
        m_target = 0;
        char str[16];
        memcpy(str, target, len);

        if (!Buffer::fromHex(str, 16, reinterpret_cast<uint8_t *>(&m_target)) || m_target == 0) {
            return false;
        }
    }
    else {
        return false;
    }

#   ifdef XMRIG_PROXY_PROJECT
    memset(m_rawTarget, 0, sizeof(m_rawTarget));
    memcpy(m_rawTarget, target, len);
#   endif

    m_diff = toDiff(m_target);
    return true;
}


void xmrig::Job::setDiff(uint64_t diff)
{
    m_diff   = diff;
    m_target = toDiff(diff);

#   ifdef XMRIG_PROXY_PROJECT
    Buffer::toHex(reinterpret_cast<uint8_t *>(&m_target), 8, m_rawTarget);
    m_rawTarget[16] = '\0';
#   endif
}
