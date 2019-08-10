/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
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

#ifndef XMRIG_RECVBUF_H
#define XMRIG_RECVBUF_H


#include <string.h>


#include "base/kernel/interfaces/ILineListener.h"


namespace xmrig {


template<size_t N>
class RecvBuf
{
public:
    inline RecvBuf() :
        m_buf(),
        m_pos(0)
    {
    }

    inline char *base()                { return m_buf; }
    inline char *current()             { return m_buf + m_pos; }
    inline const char *base() const    { return m_buf; }
    inline const char *current() const { return m_buf + m_pos; }
    inline size_t available() const    { return N - m_pos; }
    inline size_t pos() const          { return m_pos; }
    inline void nread(size_t size)     { m_pos += size; }
    inline void reset()                { m_pos = 0; }

    constexpr inline size_t size() const { return N; }

    inline void getline(ILineListener *listener)
    {
        char *end;
        char *start = m_buf;
        size_t remaining = m_pos;

        while ((end = static_cast<char*>(memchr(start, '\n', remaining))) != nullptr) {
            *end = '\0';

            end++;
            const size_t len = static_cast<size_t>(end - start);

            listener->onLine(start, len - 1);

            remaining -= len;
            start = end;
        }

        if (remaining == 0) {
            m_pos = 0;
            return;
        }

        if (start == m_buf) {
            return;
        }

        memcpy(m_buf, start, remaining);
        m_pos = remaining;
    }

private:
    char m_buf[N];
    size_t m_pos;
};


} /* namespace xmrig */


#endif /* XMRIG_RECVBUF_H */
