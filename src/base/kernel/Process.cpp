/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
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


#include <uv.h>
#include <time.h>


#include "base/kernel/Process.h"


static size_t location(xmrig::Process::Location location, char *buf, size_t max)
{
    using namespace xmrig;

    size_t size = max;
    if (location == Process::ExeLocation) {
        return uv_exepath(buf, &size) < 0 ? 0 : size;
    }

    if (location == Process::CwdLocation) {
        return uv_cwd(buf, &size) < 0 ? 0 : size;
    }

    return 0;
}


xmrig::Process::Process(int argc, char **argv) :
    m_arguments(argc, argv)
{
    srand(static_cast<unsigned int>(static_cast<uintptr_t>(time(nullptr)) ^ reinterpret_cast<uintptr_t>(this)));
}


xmrig::Process::~Process()
{
}


xmrig::String xmrig::Process::location(Location location, const char *fileName) const
{
    constexpr const size_t max = 520;

    char *buf   = new char[max]();
    size_t size = ::location(location, buf, max);

    if (size == 0) {
        delete [] buf;

        return String();
    }

    if (fileName == nullptr) {
        return buf;
    }

    if (location == ExeLocation) {
        char *p = strrchr(buf, kDirSeparator);

        if (p == nullptr) {
            delete [] buf;

            return String();
        }

        size = static_cast<size_t>(p - buf);
    }

    if ((size + strlen(fileName) + 2) >= max) {
        delete [] buf;

        return String();
    }

    buf[size] = kDirSeparator;
    strcpy(buf + size + 1, fileName);

    return buf;
}
