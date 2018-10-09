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


#include <limits.h>
#include <stdio.h>
#include <uv.h>


#ifndef XMRIG_NO_HTTPD
#   include <microhttpd.h>
#endif


#ifndef XMRIG_NO_TLS
#   include <openssl/opensslv.h>
#endif


#include "common/config/ConfigLoader.h"
#include "common/config/ConfigWatcher.h"
#include "common/interfaces/IConfig.h"
#include "common/interfaces/IWatcherListener.h"
#include "common/net/Pool.h"
#include "common/Platform.h"
#include "core/ConfigCreator.h"
#include "core/ConfigLoader_platform.h"
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/filereadstream.h"


xmrig::ConfigWatcher *xmrig::ConfigLoader::m_watcher     = nullptr;
xmrig::IConfigCreator *xmrig::ConfigLoader::m_creator    = nullptr;
xmrig::IWatcherListener *xmrig::ConfigLoader::m_listener = nullptr;


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
    rapidjson::Document doc;
    doc.Parse(json);

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

    const rapidjson::Value &pools = doc["pools"];
    if (pools.IsArray()) {
        for (const rapidjson::Value &value : pools.GetArray()) {
            if (!value.IsObject()) {
                continue;
            }

            for (size_t i = 0; i < ARRAY_SIZE(pool_options); i++) {
                parseJSON(config, &pool_options[i], value);
            }
        }
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


xmrig::IConfig *xmrig::ConfigLoader::load(int argc, char **argv, IConfigCreator *creator, IWatcherListener *listener)
{
    m_creator  = creator;
    m_listener = listener;

    xmrig::IConfig *config = m_creator->create();
    int key;

    while (1) {
        key = getopt_long(argc, argv, short_options, options, NULL);
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
        loadFromFile(config, Platform::defaultConfigName());
    }

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

    if (config->isWatch()) {
        m_watcher = new xmrig::ConfigWatcher(config->fileName(), creator, listener);
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
    uv_fs_t req;
    const int fd = uv_fs_open(uv_default_loop(), &req, fileName, O_RDONLY, 0644, nullptr);
    if (fd < 0) {
        fprintf(stderr, "unable to open %s: %s\n", fileName, uv_strerror(fd));
        return false;
    }

    uv_fs_req_cleanup(&req);

    FILE *fp = fdopen(fd, "rb");
    char buf[8192];
    rapidjson::FileReadStream is(fp, buf, sizeof(buf));

    doc.ParseStream(is);

    uv_fs_close(uv_default_loop(), &req, fd, nullptr);
    uv_fs_req_cleanup(&req);

    if (doc.HasParseError()) {
        printf("%s<%d>: %s\n", fileName, (int) doc.GetErrorOffset(), rapidjson::GetParseError_En(doc.GetParseError()));
        return false;
    }

    return doc.IsObject();
}


bool xmrig::ConfigLoader::parseArg(xmrig::IConfig *config, int key, const char *arg)
{
    switch (key) {
    case xmrig::IConfig::VersionKey: /* --version */
        showVersion();
        return false;

    case xmrig::IConfig::HelpKey: /* --help */
        showUsage();
        return false;

    case xmrig::IConfig::ConfigKey: /* --config */
        loadFromFile(config, arg);
        break;

    default:
        return config->parseString(key, arg);;
    }

    return true;
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


void xmrig::ConfigLoader::showUsage()
{
    printf(usage);
}


void xmrig::ConfigLoader::showVersion()
{
    printf(APP_NAME " " APP_VERSION "\n built on " __DATE__

#   if defined(__clang__)
    " with clang " __clang_version__);
#   elif defined(__GNUC__)
    " with GCC");
    printf(" %d.%d.%d", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#   elif defined(_MSC_VER)
    " with MSVC");
    printf(" %d", MSVC_VERSION);
#   else
    );
#   endif

    printf("\n features:"
#   if defined(__i386__) || defined(_M_IX86)
    " 32-bit"
#   elif defined(__x86_64__) || defined(_M_AMD64)
    " 64-bit"
#   endif

#   if defined(__AES__) || defined(_MSC_VER)
    " AES"
#   endif
    "\n");

    printf("\nlibuv/%s\n", uv_version_string());

#   ifndef XMRIG_NO_HTTPD
    printf("microhttpd/%s\n", MHD_get_version());
#   endif

#   if !defined(XMRIG_NO_TLS) && defined(OPENSSL_VERSION_TEXT)
    {
        constexpr const char *v = OPENSSL_VERSION_TEXT + 8;
        printf("OpenSSL/%.*s\n", static_cast<int>(strchr(v, ' ') - v), v);
    }
#   endif
}
