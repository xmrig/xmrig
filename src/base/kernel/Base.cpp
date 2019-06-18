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
#include <memory>


#include "base/io/json/Json.h"
#include "base/io/json/JsonChain.h"
#include "base/io/log/backends/ConsoleLog.h"
#include "base/io/log/backends/FileLog.h"
#include "base/io/log/Log.h"
#include "base/io/Watcher.h"
#include "base/kernel/Base.h"
#include "base/kernel/interfaces/IBaseListener.h"
#include "base/kernel/Process.h"
#include "common/Platform.h"
#include "core/config/Config.h"
#include "core/config/ConfigTransform.h"


#ifdef HAVE_SYSLOG_H
#   include "base/io/log/backends/SysLog.h"
#endif


#ifdef XMRIG_FEATURE_API
#   include "api/Api.h"
#endif


#ifdef XMRIG_FEATURE_EMBEDDED_CONFIG
#   include "core/config/Config_default.h"
#endif


class xmrig::BasePrivate
{
public:
    inline BasePrivate(Process *process) :
        api(nullptr),
        config(nullptr),
        process(process),
        watcher(nullptr)
    {}


    inline ~BasePrivate()
    {
#       ifdef XMRIG_FEATURE_API
        delete api;
#       endif

        delete config;
        delete watcher;
    }


    inline bool read(const JsonChain &chain, std::unique_ptr<Config> &config)
    {
        config = std::unique_ptr<Config>(new Config());

        return config->read(chain, chain.fileName());
    }


    inline Config *load()
    {
        JsonChain chain;
        ConfigTransform transform;
        std::unique_ptr<Config> config;

        transform.load(chain, process, transform);

        if (read(chain, config)) {
            return config.release();
        }

        chain.addFile(process->location(Process::ExeLocation, "config.json"));

        if (read(chain, config)) {
            return config.release();
        }

#       ifdef XMRIG_FEATURE_EMBEDDED_CONFIG
        chain.addRaw(default_config);

        if (read(chain, config)) {
            return config.release();
        }
#       endif

        return nullptr;
    }


    inline void replace(Config *newConfig)
    {
        Config *previousConfig = config;
        config = newConfig;

        for (IBaseListener *listener : listeners) {
            listener->onConfigChanged(config, previousConfig);
        }

        delete previousConfig;
    }


    Api *api;
    Config *config;
    Process *process;
    std::vector<IBaseListener *> listeners;
    Watcher *watcher;
};


xmrig::Base::Base(Process *process)
    : d_ptr(new BasePrivate(process))
{
}


xmrig::Base::~Base()
{
    delete d_ptr;
}


bool xmrig::Base::isReady() const
{
    return d_ptr->config != nullptr;
}


int xmrig::Base::init()
{
    d_ptr->config = d_ptr->load();

    if (!d_ptr->config) {
        LOG_EMERG("No valid configuration found. Exiting.");

        return 1;
    }

#   ifdef XMRIG_FEATURE_API
    d_ptr->api = new Api(this);
#   endif

    Platform::init(config()->userAgent());

#   ifndef XMRIG_PROXY_PROJECT
    Platform::setProcessPriority(config()->priority());
#   endif

    if (!config()->isBackground()) {
        Log::add(new ConsoleLog());
    }

    if (config()->logFile()) {
        Log::add(new FileLog(config()->logFile()));
    }

#   ifdef HAVE_SYSLOG_H
    if (config()->isSyslog()) {
        Log::add(new SysLog());
    }
#   endif

    return 0;
}


void xmrig::Base::start()
{
#   ifdef XMRIG_FEATURE_API
    api()->start();
#   endif

    if (config()->isShouldSave()) {
        config()->save();
    }

    if (config()->isWatch()) {
        d_ptr->watcher = new Watcher(config()->fileName(), this);
    }
}


void xmrig::Base::stop()
{
#   ifdef XMRIG_FEATURE_API
    api()->stop();
#   endif

    delete d_ptr->watcher;
    d_ptr->watcher = nullptr;
}


xmrig::Api *xmrig::Base::api() const
{
    assert(d_ptr->api != nullptr);

    return d_ptr->api;
}


bool xmrig::Base::reload(const rapidjson::Value &json)
{
    JsonReader reader(json);
    if (reader.isEmpty()) {
        return false;
    }

    Config *config = new Config();
    if (!config->read(reader, d_ptr->config->fileName())) {
        delete config;

        return false;
    }

    const bool saved = config->save();

    if (config->isWatch() && d_ptr->watcher && saved) {
        delete config;

        return true;
    }

    d_ptr->replace(config);

    return true;
}


xmrig::Config *xmrig::Base::config() const
{
    assert(d_ptr->config != nullptr);

    return d_ptr->config;
}


void xmrig::Base::addListener(IBaseListener *listener)
{
    d_ptr->listeners.push_back(listener);
}


void xmrig::Base::onFileChanged(const String &fileName)
{
    LOG_WARN("\"%s\" was changed, reloading configuration", fileName.data());

    JsonChain chain;
    chain.addFile(fileName);

    Config *config = new Config();

    if (!config->read(chain, chain.fileName())) {
        LOG_ERR("reloading failed");

        delete config;
        return;
    }

    d_ptr->replace(config);
}
