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

#ifndef __JOB_H__
#define __JOB_H__


#include <stdint.h>


class Job
{
public:
    Job(int poolId = -2);
    bool setBlob(const char *blob);
    bool setId(const char *id);
    bool setTarget(const char *target);

    inline bool isValid() const        { return m_size > 0 && m_diff > 0; }
    inline const char *id() const      { return m_id; }
    inline const uint8_t *blob() const { return m_blob; }
    inline int poolId() const          { return m_poolId; }
    inline uint32_t diff() const       { return m_diff; }
    inline uint32_t size() const       { return m_size; }
    inline uint64_t target() const     { return m_target; }
    inline void setPoolId(int poolId)  { m_poolId = poolId; }

    static bool fromHex(const char* in, unsigned int len, unsigned char* out);
    static void toHex(const unsigned char* in, unsigned int len, char* out);
    static inline uint64_t toDiff(uint64_t target) { return 0xFFFFFFFFFFFFFFFFULL / target; }

private:
    int m_poolId;
    char m_id[64]      __attribute__((aligned(16)));
    uint8_t m_blob[84] __attribute__((aligned(16))); // Max blob size is 84 (75 fixed + 9 variable), aligned to 96. https://github.com/xmrig/xmrig/issues/1 Thanks fireice-uk.
    uint32_t m_size;
    uint64_t m_diff;
    uint64_t m_target;
};

#endif /* __JOB_H__ */
