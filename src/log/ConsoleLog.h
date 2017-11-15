/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
 *
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

#ifndef __CONSOLELOG_H__
#define __CONSOLELOG_H__


#include <uv.h>


#include "interfaces/ILogBackend.h"


class ConsoleLog : public ILogBackend
{
public:
    ConsoleLog(bool colors);

    void message(int level, const char *fmt, va_list args) override;
    void text(const char *fmt, va_list args) override;

private:
    bool isWritable() const;
    void print(va_list args);

    bool m_colors;
    char m_buf[512];
    char m_fmt[256];
    uv_buf_t m_uvBuf;
    uv_stream_t *m_stream;
    uv_tty_t m_tty;
};

#endif /* __CONSOLELOG_H__ */
