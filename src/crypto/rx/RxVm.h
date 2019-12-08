/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018-2019 tevador     <tevador@gmail.com>
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

#ifndef XMRIG_RX_VM_H
#define XMRIG_RX_VM_H


#include "base/tools/Object.h"
#include "backend/cpu/Cpu.h"


#include <cstdint>


class randomx_vm;


namespace xmrig
{


class RxDataset;


class RxVm
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(RxVm);

    RxVm(RxDataset *dataset, uint8_t *scratchpad, bool softAes, xmrig::Assembly assembly);
    ~RxVm();

    inline randomx_vm *get() const       { return m_vm; }

private:
    int m_flags      = 0;
    randomx_vm *m_vm = nullptr;
};


} /* namespace xmrig */


#endif /* XMRIG_RX_VM_H */
