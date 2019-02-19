/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
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

#ifndef XMRIG_JOB_H
#define XMRIG_JOB_H


#include <stddef.h>
#include <stdint.h>


#include "common/crypto/Algorithm.h"
#include "common/net/Id.h"


namespace xmrig {


class Job
{
public:
    // Max blob size is 84 (75 fixed + 9 variable), aligned to 96. https://github.com/xmrig/xmrig/issues/1 Thanks fireice-uk.
    // SECOR increase requirements for blob size: https://github.com/xmrig/xmrig/issues/913
    static constexpr const size_t kMaxBlobSize = 128;

    Job();
    Job(int poolId, bool nicehash, const Algorithm &algorithm, const Id &clientId);
    ~Job();

    bool isEqual(const Job &other) const;
    bool setBlob(const char *blob);
    bool setTarget(const char *target);
    void setAlgorithm(const char *algo);
    void setHeight(uint64_t height);

    inline bool isNicehash() const                    { return m_nicehash; }
    inline bool isValid() const                       { return m_size > 0 && m_diff > 0; }
    inline bool setId(const char *id)                 { return m_id.setId(id); }
    inline const uint32_t *nonce() const              { return reinterpret_cast<const uint32_t*>(m_blob + 39); }
    inline const uint8_t *blob() const                { return m_blob; }
    inline const Algorithm &algorithm() const         { return m_algorithm; }
    inline const Id &clientId() const                 { return m_clientId; }
    inline const Id &id() const                       { return m_id; }
    inline int poolId() const                         { return m_poolId; }
    inline int threadId() const                       { return m_threadId; }
    inline size_t size() const                        { return m_size; }
    inline uint32_t *nonce()                          { return reinterpret_cast<uint32_t*>(m_blob + 39); }
    inline uint32_t diff() const                      { return static_cast<uint32_t>(m_diff); }
    inline uint64_t target() const                    { return m_target; }
    inline uint64_t height() const                    { return m_height; }
    inline void reset()                               { m_size = 0; m_diff = 0; }
    inline void setClientId(const Id &id)             { m_clientId = id; }
    inline void setPoolId(int poolId)                 { m_poolId = poolId; }
    inline void setThreadId(int threadId)             { m_threadId = threadId; }
    inline void setVariant(const char *variant)       { m_algorithm.parseVariant(variant); }
    inline void setVariant(int variant)               { m_algorithm.parseVariant(variant); }

#   ifdef XMRIG_PROXY_PROJECT
    inline char *rawBlob()                 { return m_rawBlob; }
    inline const char *rawTarget() const   { return m_rawTarget; }
#   endif

    static bool fromHex(const char* in, unsigned int len, unsigned char* out);
    static inline uint32_t *nonce(uint8_t *blob)   { return reinterpret_cast<uint32_t*>(blob + 39); }
    static inline uint64_t toDiff(uint64_t target) { return 0xFFFFFFFFFFFFFFFFULL / target; }
    static void toHex(const unsigned char* in, unsigned int len, char* out);

#   ifdef APP_DEBUG
    static char *toHex(const unsigned char* in, unsigned int len);
#   endif

    inline bool operator==(const Job &other) const { return isEqual(other); }
    inline bool operator!=(const Job &other) const { return !isEqual(other); }

private:
    Variant variant() const;

    bool m_autoVariant;
    bool m_nicehash;
    int m_poolId;
    int m_threadId;
    size_t m_size;
    uint64_t m_diff;
    uint64_t m_target;
    uint8_t m_blob[kMaxBlobSize];
    uint64_t m_height;
    xmrig::Algorithm m_algorithm;
    xmrig::Id m_clientId;
    xmrig::Id m_id;

#   ifdef XMRIG_PROXY_PROJECT
    char m_rawBlob[kMaxBlobSize * 2 + 8];
    char m_rawTarget[24];
#   endif
};


} /* namespace xmrig */


#endif /* XMRIG_JOB_H */
