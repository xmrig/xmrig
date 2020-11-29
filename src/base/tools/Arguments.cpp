/* XMRig
 * Copyright (c) 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include <algorithm>
#include <uv.h>


#include "base/tools/Arguments.h"


xmrig::Arguments::Arguments(int argc, char **argv) :
    m_argv(argv),
    m_argc(argc)
{
    uv_setup_args(argc, argv);

    for (size_t i = 0; i < static_cast<size_t>(argc); ++i) {
        add(argv[i]);
    }
}


bool xmrig::Arguments::hasArg(const char *name) const
{
    if (m_argc == 1) {
        return false;
    }

    return std::find(m_data.begin() + 1, m_data.end(), name) != m_data.end();
}


const char *xmrig::Arguments::value(const char *key1, const char *key2) const
{
    const size_t size = m_data.size();
    if (size < 3) {
        return nullptr;
    }

    for (size_t i = 1; i < size - 1; ++i) {
        if (m_data[i] == key1 || (key2 && m_data[i] == key2)) {
            return m_data[i + 1];
        }
    }

    return nullptr;
}


void xmrig::Arguments::add(const char *arg)
{
    if (arg == nullptr) {
        return;
    }

    const size_t size = strlen(arg);
    if (size > 4 && arg[0] == '-' && arg[1] == '-') {
        const char *p = strchr(arg, '=');

        if (p) {
            const auto keySize = static_cast<size_t>(p - arg);

            m_data.emplace_back(arg, keySize);
            m_data.emplace_back(arg + keySize + 1);

            return;
        }
    }

    m_data.emplace_back(arg);
}
