/* XMRig
 * Copyright 2018-     BenDr0id <ben@graef.in>
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

#ifndef __REMOTELOG_H__
#define __REMOTELOG_H__


#include <list>
#include <string>
#include <uv.h>

#include "interfaces/ILogBackend.h"


class RemoteLog : public ILogBackend
{
public:
    RemoteLog();
    ~RemoteLog();

    void message(int level, const char* fmt, va_list args) override;
    void text(const char* fmt, va_list args) override;

    static std::string getRows();


private:
    static RemoteLog* m_self;

    uv_mutex_t m_mutex;
    size_t m_maxRows;
    std::list<std::string> m_rows;
};

#endif /* __REMOTELOG_H__ */
