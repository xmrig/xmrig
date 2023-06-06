/* XMRig
 * Copyright (c) 2018-2023 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2023 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_CPU_H
#define XMRIG_CPU_H


#include "backend/cpu/interfaces/ICpuInfo.h"


namespace xmrig {


class Cpu
{
public:
    static ICpuInfo *info();
    static rapidjson::Value toJSON(rapidjson::Document &doc);
    static void release();

    inline static Assembly::Id assembly(Assembly::Id hint) { return hint == Assembly::AUTO ? Cpu::info()->assembly() : hint; }
};


} // namespace xmrig


#endif // XMRIG_CPU_H
