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

#ifndef XMRIG_BASECONFIG_H
#define XMRIG_BASECONFIG_H


#include "base/kernel/config/Title.h"
#include "base/kernel/interfaces/IConfig.h"
#include "base/net/http/Http.h"
#include "base/net/stratum/Pools.h"


#ifdef XMRIG_FEATURE_TLS
#   include "base/net/tls/TlsConfig.h"
#endif


namespace xmrig {


class IJsonReader;


class BaseConfig : public IConfig
{
public:
#   ifdef XMRIG_FEATURE_MO_BENCHMARK
    static const char *kAlgoMinTime;
    static const char *kAlgoPerf;
#   endif
    static const char *kApi;
    static const char *kApiId;
    static const char *kApiWorkerId;
    static const char *kAutosave;
    static const char *kBackground;
#   ifdef XMRIG_FEATURE_MO_BENCHMARK
    static const char *kBenchAlgoTime;
#   endif
    static const char *kColors;
    static const char *kDryRun;
    static const char *kHttp;
    static const char *kLogFile;
    static const char *kPrintTime;
#   ifdef XMRIG_FEATURE_MO_BENCHMARK
    static const char *kRebenchAlgo;
#   endif
    static const char *kSyslog;
    static const char *kTitle;
    static const char *kUserAgent;
    static const char *kVerbose;
    static const char *kWatch;

#   ifdef XMRIG_FEATURE_TLS
    static const char *kTls;
#   endif

    BaseConfig() = default;

    inline bool isAutoSave() const                          { return m_autoSave; }
    inline bool isBackground() const                        { return m_background; }
    inline bool isDryRun() const                            { return m_dryRun; }
    inline bool isSyslog() const                            { return m_syslog; }
    inline const char *logFile() const                      { return m_logFile.data(); }
    inline const char *userAgent() const                    { return m_userAgent.data(); }
    inline const Http &http() const                         { return m_http; }
    inline const Pools &pools() const                       { return m_pools; }
    inline const String &apiId() const                      { return m_apiId; }
    inline const String &apiWorkerId() const                { return m_apiWorkerId; }
    inline const Title &title() const                       { return m_title; }
    inline uint32_t printTime() const                       { return m_printTime; }

#   ifdef XMRIG_FEATURE_MO_BENCHMARK
    inline bool isRebenchAlgo() const                       { return m_rebenchAlgo; }
    inline int  benchAlgoTime() const                       { return m_benchAlgoTime; }
    inline int  algoMinTime() const                         { return m_algoMinTime; }
#   endif

#   ifdef XMRIG_FEATURE_TLS
    inline const TlsConfig &tls() const                     { return m_tls; }
#   endif

    inline bool isWatch() const override                    { return m_watch && !m_fileName.isNull(); }
    inline const String &fileName() const override          { return m_fileName; }
    inline void setFileName(const char *fileName) override  { m_fileName = fileName; }

    bool read(const IJsonReader &reader, const char *fileName) override;
    bool save() override;

    static void printVersions();

protected:
    bool m_autoSave         = true;
    bool m_background       = false;
    bool m_dryRun           = false;
    bool m_syslog           = false;
    bool m_upgrade          = false;
    bool m_watch            = true;
    Http m_http;
    Pools m_pools;
    String m_apiId;
    String m_apiWorkerId;
    String m_fileName;
    String m_logFile;
    String m_userAgent;
    Title m_title;
    uint32_t m_printTime    = 60;

#   ifdef XMRIG_FEATURE_MO_BENCHMARK
    bool m_rebenchAlgo   = false;
    int  m_benchAlgoTime = 10;
    int  m_algoMinTime   = 0;
#   endif

#   ifdef XMRIG_FEATURE_TLS
    TlsConfig m_tls;
#   endif

private:
    static void setVerbose(const rapidjson::Value &value);
};


} // namespace xmrig


#endif /* XMRIG_BASECONFIG_H */
