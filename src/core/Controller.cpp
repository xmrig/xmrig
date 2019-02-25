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


#include "common/config/ConfigLoader.h"
#include "common/cpu/Cpu.h"
#include "common/interfaces/IControllerListener.h"
#include "common/log/ConsoleLog.h"
#include "common/log/FileLog.h"
#include "common/log/Log.h"
#include "common/Platform.h"
#include "core/Config.h"
#include "core/Controller.h"
#include "net/Network.h"


#ifdef HAVE_SYSLOG_H
#   include "common/log/SysLog.h"
#endif


class xmrig::ControllerPrivate
{
public:
    inline ControllerPrivate(Process *process) :
        config(nullptr),
        network(nullptr),
        process(process)
    {}


    inline ~ControllerPrivate()
    {
        delete network;
        delete config;
    }


    Config *config;
    Network *network;
    Process *process;
    std::vector<IControllerListener *> listeners;
};


xmrig::Controller::Controller(Process *process)
    : d_ptr(new ControllerPrivate(process))
{
}


xmrig::Controller::~Controller()
{
    ConfigLoader::release();

    delete d_ptr;
}


bool xmrig::Controller::isReady() const
{
    return d_ptr->config && d_ptr->network;
}


xmrig::Config *xmrig::Controller::config() const
{
    assert(d_ptr->config != nullptr);

    return d_ptr->config;
}


int xmrig::Controller::init()
{
    Cpu::init();

    d_ptr->config = Config::load(d_ptr->process, this);
    if (!d_ptr->config) {
        return 1;
    }

    Log::init();
    Platform::init(config()->userAgent());
    Platform::setProcessPriority(d_ptr->config->priority());

    if (!config()->isBackground()) {
        Log::add(new ConsoleLog(this));
    }

    if (config()->logFile()) {
        Log::add(new FileLog(this, config()->logFile()));
    }

#   ifdef HAVE_SYSLOG_H
    if (config()->isSyslog()) {
        Log::add(new SysLog());
    }
#   endif

    d_ptr->network = new Network(this);
    return 0;
}


xmrig::Network *xmrig::Controller::network() const
{
    assert(d_ptr->network != nullptr);

    return d_ptr->network;
}


void xmrig::Controller::addListener(IControllerListener *listener)
{
    d_ptr->listeners.push_back(listener);
}


void xmrig::Controller::save()
{
    if (!config()) {
        return;
    }

    if (d_ptr->config->isShouldSave()) {
        d_ptr->config->save();
    }

    ConfigLoader::watch(d_ptr->config);
}


void xmrig::Controller::onNewConfig(IConfig *config)
{
    Config *previousConfig = d_ptr->config;
    d_ptr->config = static_cast<Config*>(config);

    for (xmrig::IControllerListener *listener : d_ptr->listeners) {
        listener->onConfigChanged(d_ptr->config, previousConfig);
    }

    delete previousConfig;
}
