/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
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

#ifndef __STORAGE_H__
#define __STORAGE_H__


#include <assert.h>
#include <map>


namespace xmrig {


template <class TYPE>
class Storage
{
public:
    inline Storage() :
        m_counter(0)
    {
    }


    inline uintptr_t add(TYPE *ptr)
    {
        m_data[m_counter] = ptr;

        return m_counter++;
    }


    inline static void *ptr(uintptr_t id) { return reinterpret_cast<void *>(id); }


    inline TYPE *get(void *id) const { return get(reinterpret_cast<uintptr_t>(id)); }
    inline TYPE *get(uintptr_t id) const
    {
        assert(m_data.count(id) > 0);

        if (m_data.count(id) == 0) {
            return nullptr;
        }

        return m_data.at(id);
    }


    inline void remove(void *id) { remove(reinterpret_cast<uintptr_t>(id)); }
    inline void remove(uintptr_t id)
    {
        TYPE *obj = get(id);
        if (obj == nullptr) {
            return;
        }

        auto it = m_data.find(id);
        if (it != m_data.end()) {
            m_data.erase(it);
        }

        delete obj;
    }


private:
    std::map<uintptr_t, TYPE *> m_data;
    uint64_t m_counter;
};


} /* namespace xmrig */


#endif /* __STORAGE_H__ */
