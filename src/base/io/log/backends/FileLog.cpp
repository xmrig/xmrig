/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2019      Spudz76     <https://github.com/Spudz76>
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


#include "base/io/log/backends/FileLog.h"


#include <cassert>
#include <cstring>
#include <uv.h>


xmrig::FileLog::FileLog(const char *fileName)
{
    uv_fs_t req;
    m_file = uv_fs_open(uv_default_loop(), &req, fileName, O_CREAT | O_APPEND | O_WRONLY, 0644, nullptr);
    uv_fs_req_cleanup(&req);
}


void xmrig::FileLog::print(int, const char *line, size_t, size_t size, bool colors)
{
    if (m_file < 0 || colors) {
        return;
    }

    assert(strlen(line) == size);

    uv_buf_t buf = uv_buf_init(new char[size], size);
    memcpy(buf.base, line, size);

    auto req = new uv_fs_t;
    req->data = buf.base;

    uv_fs_write(uv_default_loop(), req, m_file, &buf, 1, -1, FileLog::onWrite);
}


void xmrig::FileLog::onWrite(uv_fs_t *req)
{
    delete [] static_cast<char *>(req->data);

    uv_fs_req_cleanup(req);
    delete req;
}
