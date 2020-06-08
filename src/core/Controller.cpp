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


#include "core/Controller.h"
#include "backend/cpu/Cpu.h"
#include "core/config/Config.h"
#include "core/Miner.h"
#include "crypto/common/VirtualMemory.h"
#include "net/Network.h"


#include <cassert>


xmrig::Controller::Controller(Process *process) :
    Base(process)
{
}


xmrig::Controller::~Controller()
{
    delete m_network;

    VirtualMemory::destroy();
}


int xmrig::Controller::init()
{
    Base::init();

    VirtualMemory::init(config()->cpu().memPoolSize(), config()->cpu().isHugePages());

    m_network = new Network(this);

    return 0;
}

#ifdef XMRIG_FEATURE_BENCHMARK
void xmrig::Controller::pre_start()
{
    m_miner = new Miner(this);
}
#endif


void xmrig::Controller::start()
{
    Base::start();
#ifndef XMRIG_FEATURE_BENCHMARK
    m_miner = new Miner(this);
#endif
    network()->connect();
}


void xmrig::Controller::stop()
{
    Base::stop();

    delete m_network;
    m_network = nullptr;

    m_miner->stop();

    delete m_miner;
    m_miner = nullptr;
}


xmrig::Miner *xmrig::Controller::miner() const
{
    assert(m_miner != nullptr);

    return m_miner;
}


xmrig::Network *xmrig::Controller::network() const
{
    assert(m_network != nullptr);

    return m_network;
}
