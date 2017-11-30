/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
 * Copyright 2017-     BenDr0id    <ben@graef.in>
 *
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

#ifndef __OPTIONS_H__
#define __OPTIONS_H__


#include <stdint.h>
#include <vector>


#include "rapidjson/fwd.h"


class Url;
struct option;


class Options
{
public:
    enum Algo {
        ALGO_CRYPTONIGHT,      /* CryptoNight (Monero) */
        ALGO_CRYPTONIGHT_LITE, /* CryptoNight-Lite (AEON) */
    };

    enum AlgoVariant {
        AV0_AUTO,
        AV1_AESNI,
        AV2_AESNI_DOUBLE,
        AV3_SOFT_AES,
        AV4_SOFT_AES_DOUBLE,
        AV_MAX
    };

    static inline Options* i() { return m_self; }
    static Options *parse(int argc, char **argv);

    inline bool background() const                  { return m_background; }
    inline bool colors() const                      { return m_colors; }
    inline bool doubleHash() const                  { return m_doubleHash; }
    inline bool hugePages() const                   { return m_hugePages; }
    inline bool syslog() const                      { return m_syslog; }
    inline bool daemonized() const                  { return m_daemonized; }
    inline const char *configFile() const           { return m_configFile; }
    inline const char *apiToken() const             { return m_apiToken; }
    inline const char *apiWorkerId() const          { return m_apiWorkerId; }
    inline const char *logFile() const              { return m_logFile; }
    inline const char *userAgent() const            { return m_userAgent; }
    inline const char *ccHost() const               { return m_ccHost; }
    inline const char *ccToken() const              { return m_ccToken; }
    inline const char *ccWorkerId() const           { return m_ccWorkerId; }
    inline const char *ccAdminUser() const          { return m_ccAdminUser; }
    inline const char *ccAdminPass() const          { return m_ccAdminPass; }
    inline const char *ccClientConfigFolder() const { return m_ccClientConfigFolder; }
    inline const char *ccCustomDashboard() const    { return m_ccCustomDashboard == nullptr ? "index.html" : m_ccCustomDashboard; }
    inline const std::vector<Url*> &pools() const   { return m_pools; }
    inline int algo() const                         { return m_algo; }
    inline int algoVariant() const                  { return m_algoVariant; }
    inline int apiPort() const                      { return m_apiPort; }
    inline int donateLevel() const                  { return m_donateLevel; }
    inline int printTime() const                    { return m_printTime; }
    inline int priority() const                     { return m_priority; }
    inline int retries() const                      { return m_retries; }
    inline int retryPause() const                   { return m_retryPause; }
    inline int threads() const                      { return m_threads; }
    inline int ccUpdateInterval() const             { return m_ccUpdateInterval; }
    inline int ccPort() const                       { return m_ccPort; }
    inline int64_t affinity() const                 { return m_affinity; }

    inline static void release()                  { delete m_self; }

    const char *algoName() const;

private:
    constexpr static uint16_t kDefaultCCPort        = 3344;

    Options(int argc, char **argv);
    ~Options();

    inline bool isReady() const { return m_ready; }

    static Options *m_self;

    bool getJSON(const char *fileName, rapidjson::Document &doc);
    bool parseArg(int key, const char *arg);
    bool parseArg(int key, uint64_t arg);
    bool parseBoolean(int key, bool enable);
    bool parseCCUrl(const char *arg);
    Url *parseUrl(const char *arg) const;
    void parseConfig(const char *fileName);
    void parseJSON(const struct option *option, const rapidjson::Value &object);
    void showUsage(int status) const;
    void showVersion(void);

    bool setAlgo(const char *algo);

    int getAlgoVariant() const;
#   ifndef XMRIG_NO_AEON
    int getAlgoVariantLite() const;
#   endif


    bool m_background;
    bool m_colors;
    bool m_doubleHash;
    bool m_hugePages;
    bool m_ready;
    bool m_safe;
    bool m_syslog;
    bool m_daemonized;
    const char* m_configFile;
    char *m_apiToken;
    char *m_apiWorkerId;
    char *m_logFile;
    char *m_userAgent;
    char *m_ccHost;
    char *m_ccToken;
    char *m_ccWorkerId;
    char *m_ccAdminUser;
    char *m_ccAdminPass;
    char *m_ccClientConfigFolder;
    char *m_ccCustomDashboard;
    int m_algo;
    int m_algoVariant;
    int m_apiPort;
    int m_donateLevel;
    int m_maxCpuUsage;
    int m_printTime;
    int m_priority;
    int m_retries;
    int m_retryPause;
    int m_threads;
    int m_ccUpdateInterval;
    int m_ccPort;
    int64_t m_affinity;
    std::vector<Url*> m_pools;
};

#endif /* __OPTIONS_H__ */
