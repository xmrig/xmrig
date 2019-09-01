/* XMRigCC
 * Copyright 2018-     BenDr0id    <https://github.com/BenDr0id>, <ben@graef.in>
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

#ifndef XMRIG_REMOTELOG_H
#define XMRIG_REMOTELOG_H

#include <list>
#include <mutex>
#include <string>
#include <uv.h>

#include "base/kernel/interfaces/ILogBackend.h"

namespace xmrig {


class RemoteLog : public ILogBackend
{
public:
    RemoteLog(size_t maxRows);
    ~RemoteLog() override;

    static std::string getRows();

protected:
    void print(int level, const char *line, size_t offset, size_t size, bool colors) override;

private:
    static RemoteLog* m_self;

    size_t m_maxRows;
    std::list<std::string> m_rows;
    std::mutex m_mutex;
};


} /* namespace xmrig */


#endif /* XMRIG_REMOTELOG_H */
