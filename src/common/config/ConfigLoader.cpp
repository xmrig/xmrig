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
#include <limits.h>
#include <stdio.h>
#include <uv.h>


#ifndef XMRIG_NO_HTTPD
#   include <microhttpd.h>
#endif


#ifndef XMRIG_NO_TLS
#   include <openssl/opensslv.h>
#endif


#include "base/io/Json.h"
#include "base/kernel/interfaces/IConfigListener.h"
#include "base/kernel/Process.h"
#include "common/config/ConfigLoader.h"
#include "common/config/ConfigWatcher.h"
#include "common/interfaces/IConfig.h"
#include "common/Platform.h"
#include "core/ConfigCreator.h"
#include "core/ConfigLoader_platform.h"
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/fwd.h"


#ifdef XMRIG_FEATURE_EMBEDDED_CONFIG
#   include "core/ConfigLoader_default.h"
#endif


xmrig::ConfigWatcher *xmrig::ConfigLoader::m_watcher     = nullptr;
xmrig::IConfigCreator *xmrig::ConfigLoader::m_creator    = nullptr;
xmrig::IConfigListener *xmrig::ConfigLoader::m_listener  = nullptr;


#ifndef ARRAY_SIZE
#   define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#endif


bool xmrig::ConfigLoader::loadFromFile(xmrig::IConfig *config, const char *fileName)
{
    rapidjson::Document doc;
    if (!getJSON(fileName, doc)) {
        return false;
    }

    config->setFileName(fileName);

    return loadFromJSON(config, doc);
}


bool xmrig::ConfigLoader::loadFromJSON(xmrig::IConfig *config, const char *json)
{
    using namespace rapidjson;
    Document doc;
    doc.Parse<kParseCommentsFlag | kParseTrailingCommasFlag>(json);

    if (doc.HasParseError() || !doc.IsObject()) {
        return false;
    }

    return loadFromJSON(config, doc);
}


bool xmrig::ConfigLoader::loadFromJSON(xmrig::IConfig *config, const rapidjson::Document &doc)
{
    for (size_t i = 0; i < ARRAY_SIZE(config_options); i++) {
        parseJSON(config, &config_options[i], doc);
    }

    const rapidjson::Value &api = doc["api"];
    if (api.IsObject()) {
        for (size_t i = 0; i < ARRAY_SIZE(api_options); i++) {
            parseJSON(config, &api_options[i], api);
        }
    }

    config->parseJSON(doc);

    return config->finalize();
}


bool xmrig::ConfigLoader::reload(xmrig::IConfig *oldConfig, const char *json)
{
    xmrig::IConfig *config = m_creator->create();
    if (!loadFromJSON(config, json)) {
        delete config;

        return false;
    }

    config->setFileName(oldConfig->fileName());
    const bool saved = config->save();

    if (config->isWatch() && m_watcher && saved) {
        delete config;

        return true;
    }

    m_listener->onNewConfig(config);
    return true;
}


bool xmrig::ConfigLoader::watch(IConfig *config)
{
    if (!config->isWatch()) {
        return false;
    }

    assert(m_watcher == nullptr);

    m_watcher = new xmrig::ConfigWatcher(config->fileName(), m_creator, m_listener);
    return true;
}


xmrig::IConfig *xmrig::ConfigLoader::load(Process *process, IConfigCreator *creator, IConfigListener *listener)
{
    m_creator  = creator;
    m_listener = listener;

    xmrig::IConfig *config = m_creator->create();
    int key;
    int argc    = process->arguments().argc();
    char **argv = process->arguments().argv();

    while (1) {
        key = getopt_long(argc, argv, short_options, options, nullptr);
        if (key < 0) {
            break;
        }

        if (!parseArg(config, key, optarg)) {
            delete config;
            return nullptr;
        }
    }

    if (optind < argc) {
        fprintf(stderr, "%s: unsupported non-option argument '%s'\n", argv[0], argv[optind]);
        delete config;
        return nullptr;
    }

    if (!config->finalize()) {
        delete config;

        config = m_creator->create();
        loadFromFile(config, process->location(Process::ExeLocation, "config.json"));
    }

#   ifdef XMRIG_FEATURE_EMBEDDED_CONFIG
    if (!config->finalize()) {
        delete config;

        config = m_creator->create();
        loadFromJSON(config, default_config);
    }
#   endif

    if (!config->finalize()) {
        if (!config->algorithm().isValid()) {
            fprintf(stderr, "No valid algorithm specified. Exiting.\n");
        }
        else {
            fprintf(stderr, "No valid configuration found. Exiting.\n");
        }

        delete config;
        return nullptr;
    }

    return config;
}


void xmrig::ConfigLoader::release()
{
    delete m_watcher;
    delete m_creator;

    m_watcher = nullptr;
    m_creator = nullptr;
}


bool xmrig::ConfigLoader::getJSON(const char *fileName, rapidjson::Document &doc)
{
    if (Json::get(fileName, doc)) {
        return true;
    }

    if (doc.HasParseError()) {
        printf("%s<offset:%zu>: \"%s\"\n", fileName, doc.GetErrorOffset(), rapidjson::GetParseError_En(doc.GetParseError()));
    }
    else {
       fprintf(stderr, "unable to open \"%s\".\n", fileName);
    }

    return false;
}


bool xmrig::ConfigLoader::parseArg(xmrig::IConfig *config, int key, const char *arg)
{
    if (key == xmrig::IConfig::ConfigKey) {
        return loadFromFile(config, arg);
    }

    return config->parseString(key, arg);
}


void xmrig::ConfigLoader::parseJSON(xmrig::IConfig *config, const struct option *option, const rapidjson::Value &object)
{
    if (!option->name || !object.HasMember(option->name)) {
        return;
    }

    const rapidjson::Value &value = object[option->name];

    if (option->has_arg) {
        if (value.IsString()) {
            config->parseString(option->val, value.GetString());
        }
        else if (value.IsInt64()) {
            config->parseUint64(option->val, value.GetUint64());
        }
        else if (value.IsBool()) {
            config->parseBoolean(option->val, value.IsTrue());
        }
    }
    else if (value.IsBool()) {
        config->parseBoolean(option->val, value.IsTrue());
    }
}
