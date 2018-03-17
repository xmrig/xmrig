/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2016-2018 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef __JOB_H__
#define __JOB_H__


#include <stddef.h>
#include <stdint.h>


#include "net/Id.h"
#include "xmrig.h"


class Job
{
public:
    Job();
    Job(int poolId, bool nicehash, int algo, int variant);
    ~Job();

    bool setBlob(const char *blob);
    bool setTarget(const char *target);
    void setCoin(const char *coin);
    void setVariant(int variant);

    inline bool isNicehash() const         { return m_nicehash; }
    inline bool isValid() const            { return m_size > 0 && m_diff > 0; }
    inline bool setId(const char *id)      { return m_id.setId(id); }
    inline const char *coin() const        { return m_coin; }
    inline const uint32_t *nonce() const   { return reinterpret_cast<const uint32_t*>(m_blob + 39); }
    inline const uint8_t *blob() const     { return m_blob; }
    inline const xmrig::Id &id() const     { return m_id; }
    inline int poolId() const              { return m_poolId; }
    inline int threadId() const            { return m_threadId; }
    inline int variant() const             { return (m_variant == xmrig::VARIANT_AUTO ? (m_blob[0] > 6 ? 1 : 0) : m_variant); }
    inline size_t size() const             { return m_size; }
    inline uint32_t *nonce()               { return reinterpret_cast<uint32_t*>(m_blob + 39); }
    inline uint32_t diff() const           { return (uint32_t) m_diff; }
    inline uint64_t target() const         { return m_target; }
    inline void setNicehash(bool nicehash) { m_nicehash = nicehash; }
    inline void setPoolId(int poolId)      { m_poolId = poolId; }
    inline void setThreadId(int threadId)  { m_threadId = threadId; }

    static bool fromHex(const char* in, unsigned int len, unsigned char* out);
    static inline uint32_t *nonce(uint8_t *blob)   { return reinterpret_cast<uint32_t*>(blob + 39); }
    static inline uint64_t toDiff(uint64_t target) { return 0xFFFFFFFFFFFFFFFFULL / target; }
    static void toHex(const unsigned char* in, unsigned int len, char* out);

    bool operator==(const Job &other) const;
    bool operator!=(const Job &other) const;

private:
    bool m_nicehash;
    char m_coin[5];
    int m_algo;
    int m_poolId;
    int m_threadId;
    int m_variant;
    size_t m_size;
    uint64_t m_diff;
    uint64_t m_target;
    uint8_t m_blob[96]; // Max blob size is 84 (75 fixed + 9 variable), aligned to 96. https://github.com/xmrig/xmrig/issues/1 Thanks fireice-uk.
    xmrig::Id m_id;
};

#endif /* __JOB_H__ */
