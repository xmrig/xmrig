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

#ifndef __C_STR_H__
#define __C_STR_H__


#include <string.h>
#include <stdlib.h>

#include <stdio.h>


namespace xmrig {


/**
 * @brief Simple C string wrapper.
 *
 * 1. I know about std:string.
 * 2. For some reason I prefer don't use std:string in miner, eg because of file size of MSYS2 builds.
 */
class c_str
{
public:
    inline c_str() : m_data(nullptr)                   {}
    inline c_str(c_str &&other)                        { m_data = other.m_data; other.m_data = nullptr; }
    inline c_str(const c_str &other) : m_data(nullptr) { set(other.data()); }
    inline c_str(const char *str) : m_data(nullptr)    { set(str); }
    inline ~c_str()                                    { free(m_data); }


    inline void set(const char *str)
    {
        free(m_data);

        m_data = str != nullptr ? strdup(str) : nullptr;
    }


    inline void set(char *str)
    {
        free(m_data);

        m_data = str;
    }


    inline bool isEqual(const char *str) const
    {
        return (m_data != nullptr && str != nullptr && strcmp(m_data, str) == 0) || (m_data == nullptr && m_data == nullptr);
    }


    inline bool contains(const char *str) const
    {
        return strstr(m_data, str) != nullptr;
    }


    inline bool isNull() const           { return m_data == nullptr; }
    inline const char *data() const      { return m_data; }
    inline size_t size() const           { return m_data == nullptr ? 0 : strlen(m_data); }


    inline bool operator!=(const c_str &str) const { return !isEqual(str.data()); }
    inline bool operator!=(const char *str) const  { return !isEqual(str); }
    inline bool operator==(const c_str &str) const { return isEqual(str.data()); }
    inline bool operator==(const char *str) const  { return isEqual(str); }
    inline c_str &operator=(char *str)             { set(str); return *this; }
    inline c_str &operator=(const c_str &str)      { set(str.data()); return *this; }
    inline c_str &operator=(const char *str)       { set(str); return *this; }


private:
    char *m_data;
};


} /* namespace xmrig */

#endif /* __C_STR_H__ */
