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

#ifndef XMRIG_LIB_H
#define XMRIG_LIB_H


#include "base/tools/Object.h"


namespace xmrig {


class Lib
{
public:
    XMRIG_DISABLE_COPY_MOVE(Lib)

    Lib();
    inline ~Lib()   { close(); }

    bool isOpen() const;
    bool open(const char *filename);
    bool sym(const char *name, void **ptr);
    const char *lastError() const;
    void close();

    template<typename T>
    inline bool sym(const char *name, T t) { return sym(name, reinterpret_cast<void **>(t)); }

private:
    XMRIG_DECL_PRIVATE()
};


} // namespace xmrig


#endif // XMRIG_LIB_H
