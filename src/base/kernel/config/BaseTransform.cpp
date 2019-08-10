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


#include <stdio.h>


#ifdef _MSC_VER
#   include "getopt/getopt.h"
#else
#   include <getopt.h>
#endif


#include "base/kernel/config/BaseTransform.h"
#include "base/kernel/Process.h"
#include "base/io/log/Log.h"
#include "base/kernel/interfaces/IConfig.h"
#include "base/io/json/JsonChain.h"
#include "core/config/Config_platform.h"


namespace xmrig
{

static const char *kAlgo  = "algo";
static const char *kApi   = "api";
static const char *kHttp  = "http";
static const char *kPools = "pools";

}


xmrig::BaseTransform::BaseTransform()
{
}


void xmrig::BaseTransform::load(JsonChain &chain, Process *process, IConfigTransform &transform)
{
    using namespace rapidjson;

    int key;
    int argc    = process->arguments().argc();
    char **argv = process->arguments().argv();

    Document doc(kObjectType);

    while (1) {
        key = getopt_long(argc, argv, short_options, options, nullptr);
        if (key < 0) {
            break;
        }

        if (key == IConfig::ConfigKey) {
            chain.add(std::move(doc));
            chain.addFile(optarg);

            doc = Document(kObjectType);
        }
        else {
            transform.transform(doc, key, optarg);
        }
    }

    if (optind < argc) {
        LOG_WARN("%s: unsupported non-option argument '%s'", argv[0], argv[optind]);
    }

    transform.finalize(doc);
    chain.add(std::move(doc));
}


void xmrig::BaseTransform::finalize(rapidjson::Document &doc)
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    if (m_algorithm.isValid() && doc.HasMember(kPools)) {
        auto &pools = doc[kPools];
        for (Value &pool : pools.GetArray()) {
            if (!pool.HasMember(kAlgo)) {
                pool.AddMember(StringRef(kAlgo), m_algorithm.toJSON(), allocator);
            }
        }
    }
}


void xmrig::BaseTransform::transform(rapidjson::Document &doc, int key, const char *arg)
{
    switch (key) {
    case IConfig::AlgorithmKey: /* --algo */
        if (!doc.HasMember(kPools)) {
            m_algorithm = arg;
        }
        else {
            return add(doc, kPools, kAlgo, arg);
        }
        break;

    case IConfig::UserpassKey: /* --userpass */
        {
            const char *p = strrchr(arg, ':');
            if (!p) {
                return;
            }

            char *user = new char[p - arg + 1]();
            strncpy(user, arg, static_cast<size_t>(p - arg));

            add<const char *>(doc, kPools, "user", user);
            add(doc, kPools, "pass", p + 1);
            delete [] user;
        }
        break;

    case IConfig::UrlKey: /* --url */
        return add(doc, kPools, "url", arg, true);

    case IConfig::UserKey: /* --user */
        return add(doc, kPools, "user", arg);

    case IConfig::PasswordKey: /* --pass */
        return add(doc, kPools, "pass", arg);

    case IConfig::RigIdKey: /* --rig-id */
        return add(doc, kPools, "rig-id", arg);

    case IConfig::FingerprintKey: /* --tls-fingerprint */
        return add(doc, kPools, "tls-fingerprint", arg);

    case IConfig::LogFileKey: /* --log-file */
        return set(doc, "log-file", arg);

    case IConfig::HttpAccessTokenKey: /* --http-access-token */
        return set(doc, kHttp, "access-token", arg);

    case IConfig::HttpHostKey: /* --http-host */
        return set(doc, kHttp, "host", arg);

    case IConfig::ApiWorkerIdKey: /* --api-worker-id */
        return set(doc, kApi, "worker-id", arg);

    case IConfig::ApiIdKey: /* --api-id */
        return set(doc, kApi, "id", arg);

    case IConfig::UserAgentKey: /* --user-agent */
        return set(doc, "user-agent", arg);

    case IConfig::RetriesKey:     /* --retries */
    case IConfig::RetryPauseKey:  /* --retry-pause */
    case IConfig::PrintTimeKey:   /* --print-time */
    case IConfig::HttpPort:       /* --http-port */
    case IConfig::DonateLevelKey: /* --donate-level */
    case IConfig::DaemonPollKey:  /* --daemon-poll-interval */
        return transformUint64(doc, key, static_cast<uint64_t>(strtol(arg, nullptr, 10)));

    case IConfig::BackgroundKey:  /* --background */
    case IConfig::SyslogKey:      /* --syslog */
    case IConfig::KeepAliveKey:   /* --keepalive */
    case IConfig::NicehashKey:    /* --nicehash */
    case IConfig::TlsKey:         /* --tls */
    case IConfig::DryRunKey:      /* --dry-run */
    case IConfig::HttpEnabledKey: /* --http-enabled */
    case IConfig::DaemonKey:      /* --daemon */
        return transformBoolean(doc, key, true);

    case IConfig::ColorKey:          /* --no-color */
    case IConfig::HttpRestrictedKey: /* --http-no-restricted */
        return transformBoolean(doc, key, false);

    default:
        break;
    }
}


void xmrig::BaseTransform::transformBoolean(rapidjson::Document &doc, int key, bool enable)
{
    switch (key) {
    case IConfig::BackgroundKey: /* --background */
        return set(doc, "background", enable);

    case IConfig::SyslogKey: /* --syslog */
        return set(doc, "syslog", enable);

    case IConfig::KeepAliveKey: /* --keepalive */
        return add(doc, kPools, "keepalive", enable);

    case IConfig::TlsKey: /* --tls */
        return add(doc, kPools, "tls", enable);

    case IConfig::DaemonKey: /* --daemon */
        return add(doc, kPools, "daemon", enable);

#   ifndef XMRIG_PROXY_PROJECT
    case IConfig::NicehashKey: /* --nicehash */
        return add<bool>(doc, kPools, "nicehash", enable);
#   endif

    case IConfig::ColorKey: /* --no-color */
        return set(doc, "colors", enable);

    case IConfig::HttpRestrictedKey: /* --http-no-restricted */
        return set(doc, kHttp, "restricted", enable);

    case IConfig::HttpEnabledKey: /* --http-enabled */
        return set(doc, kHttp, "enabled", enable);

    case IConfig::DryRunKey: /* --dry-run */
        return set(doc, "dry-run", enable);

    default:
        break;
    }
}


void xmrig::BaseTransform::transformUint64(rapidjson::Document &doc, int key, uint64_t arg)
{
    switch (key) {
    case IConfig::RetriesKey: /* --retries */
        return set(doc, "retries", arg);

    case IConfig::RetryPauseKey: /* --retry-pause */
        return set(doc, "retry-pause", arg);

    case IConfig::DonateLevelKey: /* --donate-level */
        return set(doc, "donate-level", arg);

    case IConfig::ProxyDonateKey: /* --donate-over-proxy */
        return set(doc, "donate-over-proxy", arg);

    case IConfig::HttpPort: /* --http-port */
        return set(doc, kHttp, "port", arg);

    case IConfig::PrintTimeKey: /* --print-time */
        return set(doc, "print-time", arg);

    case IConfig::DaemonPollKey:  /* --daemon-poll-interval */
        return add(doc, kPools, "daemon-poll-interval", arg);

    default:
        break;
    }
}
