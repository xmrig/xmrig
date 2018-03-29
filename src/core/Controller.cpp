/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2016-2018 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


//#include "core/Config.h"
//#include "core/ConfigLoader.h"
#include "core/Controller.h"
#include "log/ConsoleLog.h"
#include "log/FileLog.h"
#include "log/Log.h"
#include "Platform.h"
//#include "proxy/Proxy.h"
#include "interfaces/IControllerListener.h"


#ifdef HAVE_SYSLOG_H
#   include "log/SysLog.h"
#endif


class xmrig::ControllerPrivate
{
public:
    inline ControllerPrivate() :
        config(nullptr)
    {}


    inline ~ControllerPrivate()
    {
//        delete config;
    }


    xmrig::Config *config;
    std::vector<xmrig::IControllerListener *> listeners;
};


xmrig::Controller::Controller()
    : d_ptr(new ControllerPrivate())
{
}


xmrig::Controller::~Controller()
{
//    ConfigLoader::release();

    delete d_ptr;
}


xmrig::Config *xmrig::Controller::config() const
{
    return d_ptr->config;
}


int xmrig::Controller::init(int argc, char **argv)
{
//    d_ptr->config = xmrig::Config::load(argc, argv, this);
//    if (!d_ptr->config) {
//        return 1;
//    }

//    Log::init();
//    Platform::init(config()->userAgent());

//    if (!config()->background()) {
//        Log::add(new ConsoleLog(this));
//    }

//    if (config()->logFile()) {
//        Log::add(new FileLog(config()->logFile()));
//    }

//#   ifdef HAVE_SYSLOG_H
//    if (config()->syslog()) {
//        Log::add(new SysLog());
//    }
//#   endif

//    d_ptr->proxy = new Proxy(this);
    return 0;
}


void xmrig::Controller::addListener(IControllerListener *listener)
{
    d_ptr->listeners.push_back(listener);
}


void xmrig::Controller::onNewConfig(Config *config)
{
//    xmrig::Config *previousConfig = d_ptr->config;
//    d_ptr->config = config;

//    for (xmrig::IControllerListener *listener : d_ptr->listeners) {
//        listener->onConfigChanged(config, previousConfig);
//    }

//    delete previousConfig;
}
