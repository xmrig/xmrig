/* xmlcore
 * Copyright 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2021 xmlcore       <https://github.com/xmlcore>, <support@xmlcore.com>
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


#ifdef xmlcore_FEATURE_API
#   include "base/api/Api.h"
#   include "hw/api/HwApi.h"
#endif


#include <cassert>


xmlcore::Controller::Controller(Process *process) :
    Base(process)
{
}


xmlcore::Controller::~Controller()
{
    VirtualMemory::destroy();
}


int xmlcore::Controller::init()
{
    Base::init();

    VirtualMemory::init(config()->cpu().memPoolSize(), config()->cpu().hugePageSize());

    m_network = std::make_shared<Network>(this);

#   ifdef xmlcore_FEATURE_API
    m_hwApi = std::make_shared<HwApi>();
    api()->addListener(m_hwApi.get());
#   endif

    return 0;
}


void xmlcore::Controller::start()
{
    Base::start();

    m_miner = std::make_shared<Miner>(this);

    network()->connect();
}


void xmlcore::Controller::stop()
{
    Base::stop();

    m_network.reset();

    m_miner->stop();
    m_miner.reset();
}


xmlcore::Miner *xmlcore::Controller::miner() const
{
    assert(m_miner);

    return m_miner.get();
}


xmlcore::Network *xmlcore::Controller::network() const
{
    assert(m_network);

    return m_network.get();
}


void xmlcore::Controller::execCommand(char command)
{
    miner()->execCommand(command);
    network()->execCommand(command);
}
