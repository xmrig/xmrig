/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
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


#include <assert.h>


#include "common/cpu/Cpu.h"
#include "common/Platform.h"
#include "core/Controller.h"
#include "net/Network.h"


xmrig::Controller::Controller(Process *process) :
    Base(process),
    m_network(nullptr)
{
}


xmrig::Controller::~Controller()
{
    delete m_network;
}


bool xmrig::Controller::isReady() const
{
    return Base::isReady() && m_network;
}


int xmrig::Controller::init()
{
    Cpu::init();

    const int rc = Base::init();
    if (rc != 0) {
        return rc;
    }

    m_network = new Network(this);
    return 0;
}


void xmrig::Controller::start()
{
    Base::start();

    network()->connect();
}


void xmrig::Controller::stop()
{
    Base::stop();

    delete m_network;
    m_network = nullptr;
}


xmrig::Network *xmrig::Controller::network() const
{
    assert(m_network != nullptr);

    return m_network;
}
