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


#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <uv.h>


#ifndef XMRIG_NO_HTTPD
#   include <microhttpd.h>
#endif


#ifndef XMRIG_NO_TLS
#   include <openssl/opensslv.h>
#endif


#ifdef XMRIG_AMD_PROJECT
#   if defined(__APPLE__)
#       include <OpenCL/cl.h>
#   else
#       include "3rdparty/CL/cl.h"
#   endif
#endif


#ifdef XMRIG_NVIDIA_PROJECT
#   include "nvidia/cryptonight.h"
#endif


#include "common/config/CommonConfig.h"
#include "common/log/Log.h"
#include "donate.h"
#include "rapidjson/document.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/prettywriter.h"
#include "version.h"


xmrig::CommonConfig::CommonConfig() :
    m_algorithm(CRYPTONIGHT, VARIANT_AUTO),
    m_adjusted(false),
    m_apiIPv6(false),
    m_apiRestricted(true),
    m_autoSave(true),
    m_background(false),
    m_colors(true),
    m_dryRun(false),
    m_syslog(false),

#   ifdef XMRIG_PROXY_PROJECT
    m_watch(true),
#   else
    m_watch(false), // TODO: enable config file watch by default when this feature propertly handled and tested.
#   endif

    m_apiPort(0),
    m_donateLevel(kDefaultDonateLevel),
    m_printTime(60),
    m_retries(5),
    m_retryPause(5),
    m_state(NoneState)
{
    m_pools.push_back(Pool());

#   ifdef XMRIG_PROXY_PROJECT
    m_retries    = 2;
    m_retryPause = 1;
#   endif
}


void xmrig::CommonConfig::printAPI()
{
#   ifndef XMRIG_NO_API
    if (apiPort() == 0) {
        return;
    }

    Log::i()->text(isColors() ? GREEN_BOLD(" * ") WHITE_BOLD("%-13s") CYAN("%s:") CYAN_BOLD("%d")
                              : " * %-13s%s:%d",
                   "API BIND", isApiIPv6() ? "[::]" : "0.0.0.0", apiPort());
#   endif
}


void xmrig::CommonConfig::printPools()
{
    for (size_t i = 0; i < m_activePools.size(); ++i) {
        if (!isColors()) {
            Log::i()->text(" * POOL #%-7zu%s variant=%s, TLS=%d",
                           i + 1,
                           m_activePools[i].url(),
                           m_activePools[i].algorithm().variantName(),
                           static_cast<int>(m_activePools[i].isTLS())
                           );
        }
        else {
            Log::i()->text(GREEN_BOLD(" * ") WHITE_BOLD("POOL #%-7zu") "\x1B[1;%dm%s\x1B[0m variant " WHITE_BOLD("%s"),
                           i + 1,
                           m_activePools[i].isTLS() ? 32 : 36,
                           m_activePools[i].url(),
                           m_activePools[i].algorithm().variantName()
                           );
        }
    }

#   ifdef APP_DEBUG
    LOG_NOTICE("POOLS --------------------------------------------------------------------");
    for (const Pool &pool : m_activePools) {
        pool.print();
    }
    LOG_NOTICE("--------------------------------------------------------------------------");
#   endif
}


void xmrig::CommonConfig::printVersions()
{
    char buf[256] = { 0 };

#   if defined(__clang__)
    snprintf(buf, sizeof buf, "clang/%d.%d.%d", __clang_major__, __clang_minor__, __clang_patchlevel__);
#   elif defined(__GNUC__)
    snprintf(buf, sizeof buf, "gcc/%d.%d.%d", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#   elif defined(_MSC_VER)
    snprintf(buf, sizeof buf, "MSVC/%d", MSVC_VERSION);
#   endif

    Log::i()->text(isColors() ? GREEN_BOLD(" * ") WHITE_BOLD("%-13s") CYAN_BOLD("%s/%s") WHITE_BOLD(" %s")
                              : " * %-13s%s/%s %s",
                   "ABOUT", APP_NAME, APP_VERSION, buf);

#   if defined(XMRIG_AMD_PROJECT)
#   if CL_VERSION_2_0
    const char *ocl = "2.0";
#   elif CL_VERSION_1_2
    const char *ocl = "1.2";
#   elif CL_VERSION_1_1
    const char *ocl = "1.1";
#   elif CL_VERSION_1_0
    const char *ocl = "1.0";
#   else
    const char *ocl = "0.0";
#   endif
    int length = snprintf(buf, sizeof buf, "OpenCL/%s ", ocl);
#   elif defined(XMRIG_NVIDIA_PROJECT)
    const int cudaVersion = cuda_get_runtime_version();
    int length = snprintf(buf, sizeof buf, "CUDA/%d.%d ", cudaVersion / 1000, cudaVersion % 100);
#   else
    memset(buf, 0, 16);
    int length = 0;
#   endif

#   if !defined(XMRIG_NO_TLS) && defined(OPENSSL_VERSION_TEXT)
    {
        constexpr const char *v = OPENSSL_VERSION_TEXT + 8;
        length += snprintf(buf + length, (sizeof buf) - length, "OpenSSL/%.*s ", static_cast<int>(strchr(v, ' ') - v), v);
    }
#   endif

#   ifndef XMRIG_NO_HTTPD
    length += snprintf(buf + length, (sizeof buf) - length, "microhttpd/%s ", MHD_get_version());
#   endif

    Log::i()->text(isColors() ? GREEN_BOLD(" * ") WHITE_BOLD("%-13slibuv/%s %s")
                              : " * %-13slibuv/%s %s",
                   "LIBS", uv_version_string(), buf);
}


bool xmrig::CommonConfig::save()
{
    if (m_fileName.isNull()) {
        return false;
    }

    uv_fs_t req;
    const int fd = uv_fs_open(uv_default_loop(), &req, m_fileName.data(), O_WRONLY | O_CREAT | O_TRUNC, 0644, nullptr);
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

    fflush(fp);

    uv_fs_close(uv_default_loop(), &req, fd, nullptr);
    uv_fs_req_cleanup(&req);

    LOG_NOTICE("configuration saved to: \"%s\"", m_fileName.data());
    return true;
}


bool xmrig::CommonConfig::finalize()
{
    if (m_state == ReadyState) {
        return true;
    }

    if (m_state == ErrorState) {
        return false;
    }

    if (!m_algorithm.isValid()) {
        return false;
    }

    for (Pool &pool : m_pools) {
        pool.adjust(m_algorithm);

        if (pool.isValid() && pool.algorithm().isValid()) {
#           ifdef XMRIG_NO_TLS
            if (pool.isTLS()) {
                continue;
            }
#           endif

            m_activePools.push_back(std::move(pool));
        }
    }

    m_pools.clear();

    if (m_activePools.empty()) {
        m_state = ErrorState;
        return false;
    }

    m_state = ReadyState;
    return true;
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
        m_pools.back().setKeepAlive(enable ? Pool::kKeepAliveTimeout : 0);
        break;

    case TlsKey: /* --tls */
        m_pools.back().setTLS(enable);
        break;

#   ifndef XMRIG_PROXY_PROJECT
    case NicehashKey: /* --nicehash */
        m_pools.back().setNicehash(enable);
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
        break;

    case ApiRestrictedKey: /* restricted */
        m_apiRestricted = enable;
        break;

    case DryRunKey: /* --dry-run */
        m_dryRun = enable;
        break;

    case AutoSaveKey:
        m_autoSave = enable;
        break;

    default:
        break;
    }

    return true;
}


bool xmrig::CommonConfig::parseString(int key, const char *arg)
{
    switch (key) {
    case AlgorithmKey: /* --algo */
        m_algorithm.parseAlgorithm(arg);
        break;

    case UserpassKey: /* --userpass */
        if (!m_pools.back().setUserpass(arg)) {
            return false;
        }

        break;

    case UrlKey: /* --url */
        if (m_pools.size() > 1 || m_pools[0].isValid()) {
            Pool pool(arg);

            if (pool.isValid()) {
                m_pools.push_back(std::move(pool));
            }
        }
        else {
            m_pools[0].parse(arg);
        }

        if (!m_pools.back().isValid()) {
            return false;
        }

        break;

    case UserKey: /* --user */
        m_pools.back().setUser(arg);
        break;

    case PasswordKey: /* --pass */
        m_pools.back().setPassword(arg);
        break;

    case RigIdKey: /* --rig-id */
        m_pools.back().setRigId(arg);
        break;

    case FingerprintKey: /* --tls-fingerprint */
        m_pools.back().setFingerprint(arg);
        break;

    case VariantKey: /* --variant */
        m_pools.back().algorithm().parseVariant(arg);
        break;

    case LogFileKey: /* --log-file */
        m_logFile = arg;
        break;

    case ApiAccessTokenKey: /* --api-access-token */
        m_apiToken = arg;
        break;

    case ApiWorkerIdKey: /* --api-worker-id */
        m_apiWorkerId = arg;
        break;

    case ApiIdKey: /* --api-id */
        m_apiId = arg;
        break;

    case UserAgentKey: /* --user-agent */
        m_userAgent = arg;
        break;

    case RetriesKey:     /* --retries */
    case RetryPauseKey:  /* --retry-pause */
    case ApiPort:        /* --api-port */
    case PrintTimeKey:   /* --print-time */
        return parseUint64(key, strtol(arg, nullptr, 10));

    case BackgroundKey: /* --background */
    case SyslogKey:     /* --syslog */
    case KeepAliveKey:  /* --keepalive */
    case NicehashKey:   /* --nicehash */
    case TlsKey:        /* --tls */
    case ApiIPv6Key:    /* --api-ipv6 */
    case DryRunKey:     /* --dry-run */
        return parseBoolean(key, true);

    case ColorKey:         /* --no-color */
    case WatchKey:         /* --no-watch */
    case ApiRestrictedKey: /* --api-no-restricted */
        return parseBoolean(key, false);

    case DonateLevelKey: /* --donate-level */
#       ifdef XMRIG_PROXY_PROJECT
        if (strncmp(arg, "minemonero.pro", 14) == 0) {
            m_donateLevel = 0;
            return true;
        }
#       endif
        return parseUint64(key, strtol(arg, nullptr, 10));

    default:
        break;
    }

    return true;
}


bool xmrig::CommonConfig::parseUint64(int key, uint64_t arg)
{
    return parseInt(key, static_cast<int>(arg));
}


void xmrig::CommonConfig::setFileName(const char *fileName)
{
    m_fileName = fileName;
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
        m_pools.back().setKeepAlive(arg);
        break;

    case VariantKey: /* --variant */
        m_pools.back().algorithm().parseVariant(arg);
        break;

    case DonateLevelKey: /* --donate-level */
        if (arg >= kMinimumDonateLevel && arg <= 99) {
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
