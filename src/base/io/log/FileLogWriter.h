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

#ifndef XMRIG_FILELOGWRITER_H
#define XMRIG_FILELOGWRITER_H


#include <cstddef>
#include <cstdint>
#include <vector>
#include <uv.h>


namespace xmrig {


class FileLogWriter
{
public:
    FileLogWriter();
    FileLogWriter(const char* fileName);

    ~FileLogWriter();

    inline bool isOpen() const  { return m_file >= 0; }
    inline int64_t pos() const  { return m_pos; }

    bool open(const char *fileName);
    bool write(const char *data, size_t size);
    bool writeLine(const char *data, size_t size);

private:
#   ifdef XMRIG_OS_WIN
    const char m_endl[3]  = {'\r', '\n', 0};
#   else
    const char m_endl[2]  = {'\n', 0};
#   endif

    int m_file      = -1;
    int64_t m_pos   = 0;

    uv_mutex_t m_buffersLock;
    std::vector<uv_buf_t> m_buffers;

    uv_async_t m_flushAsync;

    void init();

    static void on_flush(uv_async_t* async) { reinterpret_cast<FileLogWriter*>(async->data)->flush(); }
    void flush();
};


} /* namespace xmrig */


#endif /* XMRIG_FILELOGWRITER_H */
