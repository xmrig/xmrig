/* XMRig
 * Copyright (c) 2016-2021 SChernykh   <https://github.com/SChernykh>
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

#ifndef XMRIG_ENTRY_H
#define XMRIG_ENTRY_H


namespace xmrig {


class [[deprecated]] Process;


class Entry
{
public:
    enum Id {
        Default,
        Usage,
        Version,
        Topo,
        Platforms
    };

    static Id get(const Process &process);
    static int exec(const Process &process, Id id);
};


} /* namespace xmrig */


#endif /* XMRIG_ENTRY_H */
