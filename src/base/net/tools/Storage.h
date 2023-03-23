/* XMRig
 * Copyright 2018-2023 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2023 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_STORAGE_H
#define XMRIG_STORAGE_H


#include <cassert>
#include <map>


namespace xmrig {


template <class TYPE>
class Storage
{
public:
    inline Storage() = default;


    inline uintptr_t add(TYPE *ptr)
    {
        m_data[m_counter] = ptr;

        return m_counter++;
    }


    inline TYPE *ptr(uintptr_t id)          { return reinterpret_cast<TYPE *>(id); }


    inline TYPE *get(const void *id) const  { return get(reinterpret_cast<uintptr_t>(id)); }
    inline TYPE *get(uintptr_t id) const
    {
        assert(m_data.count(id) > 0);
        if (m_data.count(id) == 0) {
            return nullptr;
        }

        return m_data.at(id);
    }

    inline bool isEmpty() const             { return m_data.empty(); }
    inline size_t size() const              { return m_data.size(); }


    inline void remove(const void *id)      { delete release(reinterpret_cast<uintptr_t>(id)); }
    inline void remove(uintptr_t id)        { delete release(id); }


    inline TYPE *release(const void *id)    { return release(reinterpret_cast<uintptr_t>(id)); }
    inline TYPE *release(uintptr_t id)
    {
        auto obj = get(id);
        if (obj != nullptr) {
            m_data.erase(id);
        }

        return obj;
    }


private:
    std::map<uintptr_t, TYPE *> m_data;
    uintptr_t m_counter  = 0;
};


} /* namespace xmrig */


#endif /* XMRIG_STORAGE_H */
