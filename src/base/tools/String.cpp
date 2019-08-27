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


#include "base/tools/String.h"
#include "rapidjson/document.h"


xmrig::String::String(const char *str) :
    m_data(nullptr),
    m_size(str == nullptr ? 0 : strlen(str))
{
    if (m_size == 0) {
        return;
    }

    m_data = new char[m_size + 1];
    memcpy(m_data, str, m_size + 1);
}


xmrig::String::String(const char *str, size_t size) :
    m_data(nullptr),
    m_size(size)
{
    if (str == nullptr) {
        m_size = 0;

        return;
    }

    m_data = new char[m_size + 1];
    memcpy(m_data, str, m_size);
    m_data[m_size] = '\0';
}


xmrig::String::String(const String &other) :
    m_data(nullptr),
    m_size(other.m_size)
{
    if (other.m_data == nullptr) {
        return;
    }

    m_data = new char[m_size + 1];
    memcpy(m_data, other.m_data, m_size + 1);
}

#ifdef _MSC_VER
#define strncasecmp _strnicmp
#define strcasecmp _stricmp
#endif

bool xmrig::String::isEqual(const char *str, bool caseInsensitive) const
{
    if(caseInsensitive)
        return (m_data != nullptr && str != nullptr && strcasecmp(m_data, str) == 0) || (m_data == nullptr && str == nullptr);
    else
        return (m_data != nullptr && str != nullptr && strcmp(m_data, str) == 0) || (m_data == nullptr && str == nullptr);
}


bool xmrig::String::isEqual(const String &other, bool caseInsensitive) const
{
    if (m_size != other.m_size) {
        return false;
    }

    if(caseInsensitive)
        return (m_data != nullptr && other.m_data != nullptr && strncasecmp(m_data, other.m_data, m_size) == 0) || (m_data == nullptr && other.m_data == nullptr);
    else
        return (m_data != nullptr && other.m_data != nullptr && memcmp(m_data, other.m_data, m_size) == 0) || (m_data == nullptr && other.m_data == nullptr);
}


rapidjson::Value xmrig::String::toJSON() const
{
    using namespace rapidjson;

    return isNull() ? Value(kNullType) : Value(StringRef(m_data));
}


rapidjson::Value xmrig::String::toJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;

    return isNull() ? Value(kNullType) : Value(m_data, doc.GetAllocator());
}


std::vector<xmrig::String> xmrig::String::split(char sep) const
{
    std::vector<xmrig::String> out;
    if (m_size == 0) {
        return out;
    }

    size_t start = 0;
    size_t pos   = 0;

    for (pos = 0; pos < m_size; ++pos) {
        if (m_data[pos] == sep) {
            if ((pos - start) > 0) {
                out.push_back(String(m_data + start, pos - start));
            }

            start = pos + 1;
        }
    }

    if ((pos - start) > 0) {
        out.push_back(String(m_data + start, pos - start));
    }

    return out;
}


xmrig::String xmrig::String::join(const std::vector<xmrig::String> &vec, char sep)
{
    if (vec.empty()) {
        return String();
    }

    size_t size = vec.size();
    for (const String &str : vec) {
        size += str.size();
    }

    size_t offset = 0;
    char *buf     = new char[size];

    for (const String &str : vec) {
        memcpy(buf + offset, str.data(), str.size());

        offset += str.size() + 1;

        if (offset < size) {
            buf[offset - 1] = sep;
        }
    }

    buf[size - 1] = '\0';

    return String(buf);
}


void xmrig::String::copy(const char *str)
{
    delete [] m_data;

    if (str == nullptr) {
        m_size = 0;
        m_data = nullptr;

        return;
    }

    m_size = strlen(str);
    m_data = new char[m_size + 1];

    memcpy(m_data, str, m_size + 1);
}


void xmrig::String::copy(const String &other)
{
    if (m_size > 0) {
        if (m_size == other.m_size) {
            memcpy(m_data, other.m_data, m_size + 1);

            return;
        }

        delete [] m_data;
    }

    delete [] m_data;

    if (other.m_data == nullptr) {
        m_size = 0;
        m_data = nullptr;

        return;
    }

    m_size = other.m_size;
    m_data = new char[m_size + 1];

    memcpy(m_data, other.m_data, m_size + 1);
}


void xmrig::String::move(char *str)
{
    delete [] m_data;

    m_size = str == nullptr ? 0 : strlen(str);
    m_data = str;
}


void xmrig::String::move(String &&other)
{
    delete [] m_data;

    m_data = other.m_data;
    m_size = other.m_size;

    other.m_data = nullptr;
    other.m_size = 0;
}
