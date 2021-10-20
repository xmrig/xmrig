/* XMRig
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#include <cstdio>


#ifdef _MSC_VER
#   include "getopt/getopt.h"
#else
#   include <getopt.h>
#endif


#include "base/kernel/config/BaseTransform.h"
#include "base/io/json/JsonChain.h"
#include "base/io/log/Log.h"
#include "base/kernel/config/BaseConfig.h"
#include "base/kernel/interfaces/IConfig.h"
#include "base/kernel/Process.h"
#include "base/net/dns/DnsConfig.h"
#include "base/net/stratum/Pool.h"
#include "base/net/stratum/Pools.h"
#include "core/config/Config_platform.h"


#ifdef XMRIG_FEATURE_TLS
#   include "base/net/tls/TlsConfig.h"
#endif


void xmrig::BaseTransform::load(JsonChain &chain, Process *process, IConfigTransform &transform)
{
    using namespace rapidjson;

    int key     = 0;
    int argc    = process->arguments().argc();
    char **argv = process->arguments().argv();

    Document doc(kObjectType);

    while (true) {
        key = getopt_long(argc, argv, short_options, options, nullptr); // NOLINT(concurrency-mt-unsafe)
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

    if (m_algorithm.isValid() && doc.HasMember(Pools::kPools)) {
        auto &pools = doc[Pools::kPools];
        for (Value &pool : pools.GetArray()) {
            if (!pool.HasMember(Pool::kAlgo)) {
                pool.AddMember(StringRef(Pool::kAlgo), m_algorithm.toJSON(), allocator);
            }
        }
    }

    if (m_coin.isValid() && doc.HasMember(Pools::kPools)) {
        auto &pools = doc[Pools::kPools];
        for (Value &pool : pools.GetArray()) {
            if (!pool.HasMember(Pool::kCoin)) {
                pool.AddMember(StringRef(Pool::kCoin), m_coin.toJSON(), allocator);
            }
        }
    }

    if (m_http) {
        set(doc, BaseConfig::kHttp, Http::kEnabled, true);
    }
}


void xmrig::BaseTransform::transform(rapidjson::Document &doc, int key, const char *arg)
{
    switch (key) {
    case IConfig::AlgorithmKey: /* --algo */
        if (!doc.HasMember(Pools::kPools)) {
            m_algorithm = arg;
        }
        else {
            return add(doc, Pools::kPools, Pool::kAlgo, arg);
        }
        break;

    case IConfig::CoinKey: /* --coin */
        if (!doc.HasMember(Pools::kPools)) {
            m_coin = arg;
        }
        else {
            return add(doc, Pools::kPools, Pool::kCoin, arg);
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

            add<const char *>(doc, Pools::kPools, Pool::kUser, user);
            add(doc, Pools::kPools, Pool::kPass, p + 1);
            delete [] user;
        }
        break;

    case IConfig::UrlKey:    /* --url */
    case IConfig::StressKey: /* --stress */
    {
        if (!doc.HasMember(Pools::kPools)) {
            doc.AddMember(rapidjson::StringRef(Pools::kPools), rapidjson::kArrayType, doc.GetAllocator());
        }

        rapidjson::Value &array = doc[Pools::kPools];
        if (array.Size() == 0 || Pool(array[array.Size() - 1]).isValid()) {
            array.PushBack(rapidjson::kObjectType, doc.GetAllocator());
        }

#       ifdef XMRIG_FEATURE_BENCHMARK
        if (key != IConfig::UrlKey) {
            set(doc, array[array.Size() - 1], Pool::kUrl,
#           ifdef XMRIG_FEATURE_TLS
                "stratum+ssl://randomx.xmrig.com:443"
#           else
                "randomx.xmrig.com:3333"
#           endif
            );
        } else
#       endif
        {
            set(doc, array[array.Size() - 1], Pool::kUrl, arg);
        }
        break;
    }

    case IConfig::UserKey: /* --user */
        return add(doc, Pools::kPools, Pool::kUser, arg);

    case IConfig::PasswordKey: /* --pass */
        return add(doc, Pools::kPools, Pool::kPass, arg);

    case IConfig::SpendSecretKey: /* --spend-secret-key */
        return add(doc, Pools::kPools, Pool::kSpendSecretKey, arg);

    case IConfig::RigIdKey: /* --rig-id */
        return add(doc, Pools::kPools, Pool::kRigId, arg);

    case IConfig::FingerprintKey: /* --tls-fingerprint */
        return add(doc, Pools::kPools, Pool::kFingerprint, arg);

    case IConfig::SelfSelectKey: /* --self-select */
        return add(doc, Pools::kPools, Pool::kSelfSelect, arg);

    case IConfig::ProxyKey: /* --proxy */
        return add(doc, Pools::kPools, Pool::kSOCKS5, arg);

    case IConfig::LogFileKey: /* --log-file */
        return set(doc, BaseConfig::kLogFile, arg);

    case IConfig::HttpAccessTokenKey: /* --http-access-token */
        m_http = true;
        return set(doc, BaseConfig::kHttp, Http::kToken, arg);

    case IConfig::HttpHostKey: /* --http-host */
        m_http = true;
        return set(doc, BaseConfig::kHttp, Http::kHost, arg);

    case IConfig::ApiWorkerIdKey: /* --api-worker-id */
        return set(doc, BaseConfig::kApi, BaseConfig::kApiWorkerId, arg);

    case IConfig::ApiIdKey: /* --api-id */
        return set(doc, BaseConfig::kApi, BaseConfig::kApiId, arg);

    case IConfig::UserAgentKey: /* --user-agent */
        return set(doc, BaseConfig::kUserAgent, arg);

    case IConfig::TitleKey: /* --title */
        return set(doc, BaseConfig::kTitle, arg);

#   ifdef XMRIG_FEATURE_TLS
    case IConfig::TlsCertKey: /* --tls-cert */
        return set(doc, BaseConfig::kTls, TlsConfig::kCert, arg);

    case IConfig::TlsCertKeyKey: /* --tls-cert-key */
        return set(doc, BaseConfig::kTls, TlsConfig::kCertKey, arg);

    case IConfig::TlsDHparamKey: /* --tls-dhparam */
        return set(doc, BaseConfig::kTls, TlsConfig::kDhparam, arg);

    case IConfig::TlsCiphersKey: /* --tls-ciphers */
        return set(doc, BaseConfig::kTls, TlsConfig::kCiphers, arg);

    case IConfig::TlsCipherSuitesKey: /* --tls-ciphersuites */
        return set(doc, BaseConfig::kTls, TlsConfig::kCipherSuites, arg);

    case IConfig::TlsProtocolsKey: /* --tls-protocols */
        return set(doc, BaseConfig::kTls, TlsConfig::kProtocols, arg);

    case IConfig::TlsGenKey: /* --tls-gen */
        return set(doc, BaseConfig::kTls, TlsConfig::kGen, arg);
#   endif

    case IConfig::RetriesKey:       /* --retries */
#   ifdef XMRIG_FEATURE_MO_BENCHMARK
    case IConfig::BenchAlgoTimeKey: /* --bench-algo-time */
    case IConfig::AlgoMinTimeKey:   /* --algo-min-time */
#   endif
    case IConfig::RetryPauseKey:    /* --retry-pause */
    case IConfig::PrintTimeKey:     /* --print-time */
    case IConfig::HttpPort:         /* --http-port */
    case IConfig::DonateLevelKey:   /* --donate-level */
    case IConfig::DaemonPollKey:    /* --daemon-poll-interval */
    case IConfig::DnsTtlKey:        /* --dns-ttl */
    case IConfig::DaemonZMQPortKey: /* --daemon-zmq-port */
        return transformUint64(doc, key, static_cast<uint64_t>(strtol(arg, nullptr, 10)));

    case IConfig::BackgroundKey:  /* --background */
    case IConfig::SyslogKey:      /* --syslog */
    case IConfig::KeepAliveKey:   /* --keepalive */
    case IConfig::NicehashKey:    /* --nicehash */
#   ifdef XMRIG_FEATURE_TLS
    case IConfig::TlsKey:         /* --tls */
#   endif
    case IConfig::DryRunKey:      /* --dry-run */
#   ifdef XMRIG_FEATURE_HTTP
    case IConfig::HttpEnabledKey: /* --http-enabled */
    case IConfig::DaemonKey:      /* --daemon */
#   endif
#   ifdef XMRIG_FEATURE_MO_BENCHMARK
    case IConfig::RebenchAlgoKey: /* --rebench-algo */
#   endif
    case IConfig::PauseOnBatteryKey: /* --pause-on-battery */
    case IConfig::SubmitToOriginKey: /* --submit-to-origin */
    case IConfig::VerboseKey:     /* --verbose */
    case IConfig::DnsIPv6Key:     /* --dns-ipv6 */
        return transformBoolean(doc, key, true);

    case IConfig::ColorKey:          /* --no-color */
    case IConfig::HttpRestrictedKey: /* --http-no-restricted */
    case IConfig::NoTitleKey:        /* --no-title */
        return transformBoolean(doc, key, false);

    default:
        break;
    }
}


void xmrig::BaseTransform::transformBoolean(rapidjson::Document &doc, int key, bool enable)
{
    switch (key) {
    case IConfig::BackgroundKey: /* --background */
        return set(doc, BaseConfig::kBackground, enable);

    case IConfig::SyslogKey: /* --syslog */
        return set(doc, BaseConfig::kSyslog, enable);

    case IConfig::KeepAliveKey: /* --keepalive */
        return add(doc, Pools::kPools, Pool::kKeepalive, enable);

    case IConfig::TlsKey: /* --tls */
        return add(doc, Pools::kPools, Pool::kTls, enable);

    case IConfig::SubmitToOriginKey: /* --submit-to-origin */
        return add(doc, Pools::kPools, Pool::kSubmitToOrigin, enable);
#   ifdef XMRIG_FEATURE_HTTP
    case IConfig::DaemonKey: /* --daemon */
        return add(doc, Pools::kPools, Pool::kDaemon, enable);
#   endif

#   ifndef XMRIG_PROXY_PROJECT
    case IConfig::NicehashKey: /* --nicehash */
        return add<bool>(doc, Pools::kPools, Pool::kNicehash, enable);
#   endif

    case IConfig::ColorKey: /* --no-color */
        return set(doc, BaseConfig::kColors, enable);

    case IConfig::HttpRestrictedKey: /* --http-no-restricted */
        m_http = true;
        return set(doc, BaseConfig::kHttp, Http::kRestricted, enable);

    case IConfig::HttpEnabledKey: /* --http-enabled */
        m_http = true;
        break;

    case IConfig::DryRunKey: /* --dry-run */
        return set(doc, BaseConfig::kDryRun, enable);

    case IConfig::VerboseKey: /* --verbose */
        return set(doc, BaseConfig::kVerbose, enable);

    case IConfig::NoTitleKey: /* --no-title */
        return set(doc, BaseConfig::kTitle, enable);

#   ifdef XMRIG_FEATURE_MO_BENCHMARK
    case IConfig::RebenchAlgoKey: /* --rebench-algo */
        return set(doc, BaseConfig::kRebenchAlgo, enable);
#   endif

    case IConfig::DnsIPv6Key: /* --dns-ipv6 */
        return set(doc, DnsConfig::kField, DnsConfig::kIPv6, enable);

    default:
        break;
    }
}


void xmrig::BaseTransform::transformUint64(rapidjson::Document &doc, int key, uint64_t arg)
{
    switch (key) {
    case IConfig::RetriesKey: /* --retries */
        return set(doc, Pools::kRetries, arg);

    case IConfig::RetryPauseKey: /* --retry-pause */
        return set(doc, Pools::kRetryPause, arg);

    case IConfig::DonateLevelKey: /* --donate-level */
        return set(doc, Pools::kDonateLevel, arg);

    case IConfig::ProxyDonateKey: /* --donate-over-proxy */
        return set(doc, Pools::kDonateOverProxy, arg);

    case IConfig::HttpPort: /* --http-port */
        m_http = true;
        return set(doc, BaseConfig::kHttp, Http::kPort, arg);

    case IConfig::PrintTimeKey: /* --print-time */
        return set(doc, BaseConfig::kPrintTime, arg);

    case IConfig::DnsTtlKey: /* --dns-ttl */
        return set(doc, DnsConfig::kField, DnsConfig::kTTL, arg);

#   ifdef XMRIG_FEATURE_HTTP
    case IConfig::DaemonPollKey:  /* --daemon-poll-interval */
        return add(doc, Pools::kPools, Pool::kDaemonPollInterval, arg);

    case IConfig::DaemonZMQPortKey:  /* --daemon-zmq-port */
        return add(doc, Pools::kPools, Pool::kDaemonZMQPort, arg);
#   endif

#   ifdef XMRIG_FEATURE_MO_BENCHMARK
    case IConfig::BenchAlgoTimeKey: /* --bench-algo-time */
        return set(doc, BaseConfig::kBenchAlgoTime, arg);

    case IConfig::AlgoMinTimeKey:   /* --algo-min-time */
        return set(doc, BaseConfig::kAlgoMinTime, arg);
#   endif

    default:
        break;
    }
}
