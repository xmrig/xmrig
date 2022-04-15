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

#ifndef XMRIG_ARGUMENTS_H
#define XMRIG_ARGUMENTS_H


#include <algorithm>
#include <functional>


#include "base/tools/String.h"


namespace xmrig {


class Arguments
{
public:
    using Callback = std::function<void(const String &key, const String &value, size_t index)>;

    Arguments(int argc, char **argv);

    template<typename... Args>
    bool contains(Args... args) const {
        static_assert(sizeof...(args) >= 2, "Expected at least 2 arguments");

        for (const char *key : { args... }) {
            if (contains(key)) {
                return true;
            }
        }

        return false;
    }

    template<typename... Args>
    const String &value(Args... args) const {
        static_assert(sizeof...(args) >= 2, "Expected at least 2 arguments");

        for (const char *key : { args... }) {
            const auto &v = value(key);
            if (v) {
                return v;
            }
        }

        return m_empty;
    }

    const String &value(const char *key) const;
    const String &value(size_t i) const;
    size_t pos(const char *key) const;
    std::vector<String> values(const char *key) const;
    void forEach(const Callback &callback) const;

    inline bool contains(const char *key) const         { return size() > 1 && std::find(m_data.begin() + 1, m_data.end(), key) != m_data.end(); }
    inline char **argv() const                          { return m_argv; }
    inline const std::vector<String> &data() const      { return m_data; }
    inline const String &at(size_t i) const             { return i < size() ? m_data[i] : m_empty; }
    inline int argc() const                             { return m_argc; }
    inline size_t count(const char *key) const          { return size() > 1 ? std::count(m_data.begin() + 1, m_data.end(), key) : 0U; }
    inline size_t size() const                          { return m_data.size(); }

    inline const String &operator[](size_t i) const     { return at(i); }

private:
    static const String m_empty;

    inline void add(const char *arg, size_t size)       { m_data.emplace_back(arg, size); }


    void add(const char *arg);

    char **m_argv;
    const int m_argc;
    std::vector<String> m_data;
};


} /* namespace xmrig */


#endif /* XMRIG_ARGUMENTS_H */
