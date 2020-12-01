/* XMRig
 * Copyright (c) 2019      Spudz76     <https://github.com/Spudz76>
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

#ifndef XMRIG_CONSOLELOG_H
#define XMRIG_CONSOLELOG_H


using uv_stream_t = struct uv_stream_s;
using uv_tty_t    = struct uv_tty_s;


#include "base/kernel/interfaces/ILogBackend.h"
#include "base/tools/Object.h"


namespace xmrig {


class Title;


class ConsoleLog : public ILogBackend
{
public:
    XMRIG_DISABLE_COPY_MOVE(ConsoleLog)

    ConsoleLog(const Title &title);
    ~ConsoleLog() override;

protected:
    void print(uint64_t timestamp, int level, const char *line, size_t offset, size_t size, bool colors) override;

private:
    bool isSupported() const;

    uv_tty_t *m_tty = nullptr;

#   ifdef XMRIG_OS_WIN
    bool isWritable() const;

    uv_stream_t *m_stream = nullptr;
#   endif
};


} /* namespace xmrig */


#endif /* XMRIG_CONSOLELOG_H */
