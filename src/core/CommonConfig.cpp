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


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <uv.h>


#include "core/CommonConfig.h"
#include "donate.h"
#include "log/Log.h"
#include "net/Url.h"
#include "rapidjson/document.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/prettywriter.h"
#include "xmrig.h"


static const char *algoNames[] = {
    "cryptonight",
    "cryptonight-lite",
    "cryptonight-heavy"
};


#if defined(_WIN32) && !defined(strcasecmp)
#   define strcasecmp _stricmp
#endif


xmrig::CommonConfig::CommonConfig() :
    m_algorithm(CRYPTONIGHT),
    m_adjusted(false),
    m_apiIPv6(true),
    m_apiRestricted(true),
    m_background(false),
    m_colors(true),
    m_syslog(false),
    m_watch(false), // TODO: enable config file watch by default when this feature propertly handled and tested.
    m_apiToken(nullptr),
    m_apiWorkerId(nullptr),
    m_fileName(nullptr),
    m_logFile(nullptr),
    m_userAgent(nullptr),
    m_apiPort(0),
    m_donateLevel(kDefaultDonateLevel),
    m_printTime(60),
    m_retries(5),
    m_retryPause(5)
{
    m_pools.push_back(new Url());

#   ifdef XMRIG_PROXY_PROJECT
    m_retries    = 2;
    m_retryPause = 1;
#   endif
}


xmrig::CommonConfig::~CommonConfig()
{
    for (Url *url : m_pools) {
        delete url;
    }

    m_pools.clear();

    free(m_fileName);
    free(m_apiToken);
    free(m_apiWorkerId);
    free(m_logFile);
    free(m_userAgent);
}


const char *xmrig::CommonConfig::algoName(Algo algorithm)
{
    return algoNames[algorithm];
}


bool xmrig::CommonConfig::adjust()
{
    if (m_adjusted) {
        return false;
    }

    m_adjusted = true;

    for (Url *url : m_pools) {
        url->adjust(algorithm());
    }

    return true;
}


bool xmrig::CommonConfig::isValid() const
{
    return m_pools[0]->isValid();
}


bool xmrig::CommonConfig::parseBoolean(int key, bool enable)
{
    switch (key) {
    case BackgroundKey: /* --background */
        m_background = enable;
        break;

    case SyslogKey: /* --syslog */
        m_syslog = enable;
        break;

    case KeepAliveKey: /* --keepalive */
        m_pools.back()->setKeepAlive(enable ? Url::kKeepAliveTimeout : 0);
        break;

#   ifndef XMRIG_PROXY_PROJECT
    case NicehashKey: /* --nicehash */
        m_pools.back()->setNicehash(enable);
        break;
#   endif

    case ColorKey: /* --no-color */
        m_colors = enable;
        break;

    case WatchKey: /* watch */
        m_watch = enable;
        break;

    case ApiIPv6Key: /* ipv6 */
        m_apiIPv6 = enable;

    case ApiRestrictedKey: /* restricted */
        m_apiRestricted = enable;

    default:
        break;
    }

    return true;
}


bool xmrig::CommonConfig::parseString(int key, const char *arg)
{
    switch (key) {
    case AlgorithmKey: /* --algo */
        setAlgo(arg);
        break;

    case UserpassKey: /* --userpass */
        if (!m_pools.back()->setUserpass(arg)) {
            return false;
        }

        break;

    case UrlKey: /* --url */
        if (m_pools.size() > 1 || m_pools[0]->isValid()) {
            Url *url = new Url(arg);
            if (url->isValid()) {
                m_pools.push_back(url);
            }
            else {
                delete url;
            }
        }
        else {
            m_pools[0]->parse(arg);
        }

        if (!m_pools.back()->isValid()) {
            return false;
        }

        break;

    case UserKey: /* --user */
        m_pools.back()->setUser(arg);
        break;

    case PasswordKey: /* --pass */
        m_pools.back()->setPassword(arg);
        break;

    case LogFileKey: /* --log-file */
        free(m_logFile);
        m_logFile = strdup(arg);
        break;

    case ApiAccessTokenKey: /* --api-access-token */
        free(m_apiToken);
        m_apiToken = strdup(arg);
        break;

    case ApiWorkerIdKey: /* --api-worker-id */
        free(m_apiWorkerId);
        m_apiWorkerId = strdup(arg);
        break;

    case UserAgentKey: /* --user-agent */
        free(m_userAgent);
        m_userAgent = strdup(arg);
        break;

    case RetriesKey:     /* --retries */
    case RetryPauseKey:  /* --retry-pause */
    case VariantKey:     /* --variant */
    case ApiPort:        /* --api-port */
    case PrintTimeKey:   /* --cpu-priority */
        return parseUint64(key, strtol(arg, nullptr, 10));

    case BackgroundKey: /* --background */
    case SyslogKey:     /* --syslog */
    case KeepAliveKey:  /* --keepalive */
    case NicehashKey:   /* --nicehash */
        return parseBoolean(key, true);

    case ColorKey:         /* --no-color */
    case WatchKey:         /* --no-watch */
    case ApiRestrictedKey: /* --api-no-restricted */
    case ApiIPv6Key:       /* --api-no-ipv6 */
        return parseBoolean(key, false);

#   ifdef XMRIG_PROXY_PROJECT
    case 1003: /* --donate-level */
        if (strncmp(arg, "minemonero.pro", 14) == 0) {
            m_donateLevel = 0;
        }
        else {
            parseUint64(key, strtol(arg, nullptr, 10));
        }
        break;
#   endif

    default:
        break;
    }

    return true;
}


bool xmrig::CommonConfig::parseUint64(int key, uint64_t arg)
{
    return parseInt(key, static_cast<int>(arg));
}


bool xmrig::CommonConfig::save()
{
    if (!m_fileName) {
        return false;
    }

    uv_fs_t req;
    const int fd = uv_fs_open(uv_default_loop(), &req, m_fileName, O_WRONLY | O_CREAT | O_TRUNC, 0644, nullptr);
    if (fd < 0) {
        return false;
    }

    uv_fs_req_cleanup(&req);

    rapidjson::Document doc;
    getJSON(doc);

    FILE *fp = fdopen(fd, "w");

    char buf[4096];
    rapidjson::FileWriteStream os(fp, buf, sizeof(buf));
    rapidjson::PrettyWriter<rapidjson::FileWriteStream> writer(os);
    doc.Accept(writer);

    fclose(fp);

    uv_fs_close(uv_default_loop(), &req, fd, nullptr);
    uv_fs_req_cleanup(&req);

    LOG_NOTICE("configuration saved to: \"%s\"", m_fileName);
    return true;
}


void xmrig::CommonConfig::setFileName(const char *fileName)
{
    free(m_fileName);
    m_fileName = fileName ? strdup(fileName) : nullptr;
}


bool xmrig::CommonConfig::parseInt(int key, int arg)
{
    switch (key) {
    case RetriesKey: /* --retries */
        if (arg > 0 && arg <= 1000) {
            m_retries = arg;
        }
        break;

    case RetryPauseKey: /* --retry-pause */
        if (arg > 0 && arg <= 3600) {
            m_retryPause = arg;
        }
        break;

    case KeepAliveKey: /* --keepalive */
        m_pools.back()->setKeepAlive(arg);
        break;

    case VariantKey: /* --variant */
        m_pools.back()->setVariant(arg);
        break;

    case DonateLevelKey: /* --donate-level */
        if (arg >= kMinDonateLevel && arg <= 99) {
            m_donateLevel = arg;
        }
        break;

    case ApiPort: /* --api-port */
        if (arg > 0 && arg <= 65536) {
            m_apiPort = arg;
        }
        break;

    case PrintTimeKey: /* --print-time */
        if (arg >= 0 && arg <= 3600) {
            m_printTime = arg;
        }
        break;

    default:
        break;
    }

    return true;
}


void xmrig::CommonConfig::setAlgo(const char *algo)
{
    if (strcasecmp(algo, "cryptonight-light") == 0) {
        fprintf(stderr, "Algorithm \"cryptonight-light\" is deprecated, use \"cryptonight-lite\" instead\n");

        m_algorithm = CRYPTONIGHT_LITE;
        return;
    }

    const size_t size = sizeof(algoNames) / sizeof((algoNames)[0]);

    for (size_t i = 0; i < size; i++) {
        if (algoNames[i] && strcasecmp(algo, algoNames[i]) == 0) {
            m_algorithm = static_cast<Algo>(i);
            break;
        }
    }
}
