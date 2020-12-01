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

#ifndef XMRIG_ARGUMENTS_H
#define XMRIG_ARGUMENTS_H


#include <vector>


#include "base/tools/String.h"


namespace xmrig {


class Arguments
{
public:
    Arguments(int argc, char **argv);

    bool hasArg(const char *name) const;
    const char *value(const char *key1, const char *key2 = nullptr) const;

    inline char **argv() const                     { return m_argv; }
    inline const std::vector<String> &data() const { return m_data; }
    inline int argc() const                        { return m_argc; }

private:
    void add(const char *arg);

    char **m_argv;
    int m_argc;
    std::vector<String> m_data;
};


} /* namespace xmrig */


#endif /* XMRIG_ARGUMENTS_H */
