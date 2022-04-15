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

#include <uv.h>


#include "base/tools/Arguments.h"


namespace xmrig {


const String Arguments::m_empty;


} // namespace xmrig


xmrig::Arguments::Arguments(int argc, char **argv) :
    m_argv(argv),
    m_argc(argc)
{
    uv_setup_args(argc, argv);
    m_data.reserve(argc);

    for (size_t i = 0; i < static_cast<size_t>(argc); ++i) {
        add(argv[i]);
    }
}


const xmrig::String &xmrig::Arguments::value(const char *key) const
{
    if (size() < 3) {
        return m_empty;
    }

    for (size_t i = size() - 2; i > 0; i--) {
        if (at(i) == key) {
            const auto &v = value(i + 1);

            if (!v.isNull()) {
                return v;
            }
        }
    }

    return m_empty;
}


const xmrig::String &xmrig::Arguments::value(size_t i) const
{
    const auto &v = at(i);

    return v.size() < 1 || *v.data() == '-' ? m_empty : v;
}


size_t xmrig::Arguments::pos(const char *key) const
{
    if (size() < 2) {
        return 0;
    }

    for (size_t i = size() - 1; i > 0; i--) {
        if (at(i) == key) {
            return i;
        }
    }

    return 0;
}


std::vector<xmrig::String> xmrig::Arguments::values(const char *key) const
{
    const size_t n = count(key);
    if (!n) {
        return {};
    }

    std::vector<String> out;
    out.reserve(n);

    for (size_t i = 1; i < size(); ++i) {
        if (at(i) == key) {
            const auto &v = value(i + 1);

            if (!v.isNull()) {
                ++i;
                out.emplace_back(v);
            }
        }
    }

    return out;
}


void xmrig::Arguments::forEach(const Callback &callback) const
{
    for (size_t i = 1; i < size(); ++i) {
        const auto &v = value(i + 1);
        callback(at(i), v, i);

        if (!v.isNull()) {
            ++i;
        }
    }
}


void xmrig::Arguments::add(const char *arg)
{
    const size_t size = arg == nullptr ? 0U : strlen(arg);

    if (size < 1) {
        return;
    }

    if (*arg != '-') {
        return add(arg, size);
    }

    if (size > 1 && arg[1] != '-') {
        static char tmp[3] = { 0x2d, 0x00, 0x00 };

        for (size_t i = 1; i < size; i++) {
            tmp[1] = arg[i];

            add(tmp, 2);
        }

        return;
    }

    if (size > 3) {
        const char *p = nullptr;

        if (size > 5 && (p = strchr(arg, '='))) {
            const auto ks = static_cast<size_t>(p - arg);

            add(arg, ks);

            if (size - ks > 1) {
                add(arg + ks + 1, size - ks - 1);
            }

            return;
        }

        add(arg, size);
    }
}
