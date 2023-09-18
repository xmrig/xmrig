/* XMRig
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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
#include "3rdparty/rapidjson/document.h"


#include <cctype>


xmrig::String::String(const char *str, size_t size) :
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


xmrig::String::String(const char *str) :
    m_size(str == nullptr ? 0 : strlen(str))
{
    if (str == nullptr) {
        return;
    }

    m_data = new char[m_size + 1];
    memcpy(m_data, str, m_size + 1);
}


xmrig::String::String(const rapidjson::Value &value)
{
    if (!value.IsString()) {
        return;
    }

    if ((m_size = value.GetStringLength()) == 0) {
        return;
    }

    m_data = new char[m_size + 1];
    memcpy(m_data, value.GetString(), m_size);
    m_data[m_size] = '\0';
}


xmrig::String::String(const String &other) :
    m_size(other.m_size)
{
    if (other.m_data == nullptr) {
        return;
    }

    m_data = new char[m_size + 1];
    memcpy(m_data, other.m_data, m_size + 1);
}


bool xmrig::String::isEqual(const char *str) const
{
    return (m_data != nullptr && str != nullptr && strcmp(m_data, str) == 0) || (m_data == nullptr && str == nullptr);
}


bool xmrig::String::isEqual(const String &other) const
{
    if (m_size != other.m_size) {
        return false;
    }

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
    std::vector<String> out;
    if (m_size == 0) {
        out.emplace_back(*this);

        return out;
    }

    size_t start = 0;
    size_t pos   = 0;

    for (pos = 0; pos < m_size; ++pos) {
        if (m_data[pos] == sep) {
            if (pos > start) {
                out.emplace_back(m_data + start, pos - start);
            }

            start = pos + 1;
        }
    }

    if (pos > start) {
        out.emplace_back(m_data + start, pos - start);
    }

    return out;
}


xmrig::String &xmrig::String::toLower()
{
    if (isNull() || isEmpty()) {
        return *this;
    }

    for (size_t i = 0; i < size(); ++i) {
        m_data[i] = static_cast<char>(tolower(m_data[i]));
    }

    return *this;
}


xmrig::String &xmrig::String::toUpper()
{
    if (isNull() || isEmpty()) {
        return *this;
    }

    for (size_t i = 0; i < size(); ++i) {
        m_data[i] = static_cast<char>(toupper(m_data[i]));
    }

    return *this;
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
    if (m_size > 0 && m_size == other.m_size) {
        memcpy(m_data, other.m_data, m_size + 1);

        return;
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
