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

#ifndef XMRIG_CONSOLELOG_H
#define XMRIG_CONSOLELOG_H


typedef struct uv_stream_s uv_stream_t;
typedef struct uv_tty_s uv_tty_t;


#include "base/kernel/interfaces/ILogBackend.h"


namespace xmrig {


class ConsoleLog : public ILogBackend
{
public:
    ConsoleLog();
    ~ConsoleLog() override;

protected:
    void print(int level, const char *line, size_t offset, size_t size, bool colors) override;

private:
    bool isWritable() const;

    uv_stream_t *m_stream;
    uv_tty_t *m_tty;
};


} /* namespace xmrig */


#endif /* XMRIG_CONSOLELOG_H */
