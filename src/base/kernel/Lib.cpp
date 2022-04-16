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

#include "base/kernel/Lib.h"


#include <cassert>
#include <uv.h>


namespace xmrig {


class Lib::Private
{
public:
    bool open   = false;
    uv_lib_t lib{};
};


} // namespace xmrig


xmrig::Lib::Lib() :
    d(std::make_shared<Private>())
{
}


bool xmrig::Lib::isOpen() const
{
    return d->open;
}


bool xmrig::Lib::open(const char *filename)
{
    assert(!isOpen());

    return (d->open = uv_dlopen(filename, &d->lib) == 0);
}


bool xmrig::Lib::sym(const char *name, void **ptr)
{
    return isOpen() && uv_dlsym(&d->lib, name, ptr);
}


const char *xmrig::Lib::lastError() const
{
    return uv_dlerror(&d->lib);
}


void xmrig::Lib::close()
{
    if (isOpen()) {
        uv_dlclose(&d->lib);
        d->open = false;
    }
}
