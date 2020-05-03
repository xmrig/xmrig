/* XMRig
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include "base/io/log/FileLogWriter.h"
#include "base/io/Env.h"


#include <cassert>
#include <uv.h>


namespace xmrig {


static void fsWriteCallback(uv_fs_t *req)
{
    delete [] static_cast<char *>(req->data);

    uv_fs_req_cleanup(req);
    delete req;
}


static const char *kNewLine = "\n";


} // namespace xmrig


bool xmrig::FileLogWriter::open(const char *fileName)
{
    assert(fileName != nullptr);
    if (!fileName) {
        return false;
    }

    uv_fs_t req;
    m_file = uv_fs_open(uv_default_loop(), &req, Env::expand(fileName), O_CREAT | O_APPEND | O_WRONLY, 0644, nullptr);
    uv_fs_req_cleanup(&req);

    return isOpen();
}


bool xmrig::FileLogWriter::write(const char *data, size_t size)
{
    if (!isOpen()) {
        return false;
    }

    uv_buf_t buf = uv_buf_init(new char[size], size);
    memcpy(buf.base, data, size);

    auto req = new uv_fs_t;
    req->data = buf.base;

    uv_fs_write(uv_default_loop(), req, m_file, &buf, 1, -1, fsWriteCallback);

    return true;
}


bool xmrig::FileLogWriter::writeLine(const char *data, size_t size)
{
    uv_buf_t buf[2] = {
        uv_buf_init(new char[size], size),
        uv_buf_init(const_cast<char *>(kNewLine), 1)
    };

    memcpy(buf[0].base, data, size);

    auto req = new uv_fs_t;
    req->data = buf[0].base;

    uv_fs_write(uv_default_loop(), req, m_file, buf, 2, -1, fsWriteCallback);

    return true;
}
