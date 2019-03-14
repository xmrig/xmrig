/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
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

#ifndef XMRIG_ID_H
#define XMRIG_ID_H


#include <string.h>


namespace xmrig {


class Id
{
public:
    inline Id() :
        m_data()
    {
    }


    inline Id(const char *id, size_t sizeFix = 0)
    {
        setId(id, sizeFix);
    }


    inline bool operator==(const Id &other) const
    {
        return memcmp(m_data, other.m_data, sizeof(m_data)) == 0;
    }


    inline bool operator!=(const Id &other) const
    {
        return memcmp(m_data, other.m_data, sizeof(m_data)) != 0;
    }


    Id &operator=(const Id &other)
    {
        memcpy(m_data, other.m_data, sizeof(m_data));

        return *this;
    }


    inline bool setId(const char *id, size_t sizeFix = 0)
    {
        memset(m_data, 0, sizeof(m_data));
        if (!id) {
            return false;
        }

        const size_t size = strlen(id);
        if (size >= sizeof(m_data)) {
            return false;
        }

        memcpy(m_data, id, size - sizeFix);
        return true;
    }


    inline const char *data() const { return m_data; }
    inline bool isValid() const     { return *m_data != '\0'; }


private:
    char m_data[64];
};


} /* namespace xmrig */


#endif /* XMRIG_ID_H */
