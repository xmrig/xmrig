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

#ifndef MAX_NUM_HASH_BLOCKS
#define MAX_NUM_HASH_BLOCKS 5
#endif

#include <cstdint>
#include <vector>

#include "rapidjson/fwd.h"
#include "PowVariant.h"

class Url;
struct option;


class Options
{
public:
    enum Algo {
        ALGO_CRYPTONIGHT,      /* CryptoNight (2MB ScratchPad) */
        ALGO_CRYPTONIGHT_LITE, /* CryptoNight-Lite (1MB ScratchPad) */
        ALGO_CRYPTONIGHT_HEAVY /* CryptoNight-Heavy (4MB ScratchPad) */
    };

    enum AlgoVariant {
        AV0_AUTO,
        AV1_AESNI,
        AV2_AESNI_DOUBLE,
        AV3_SOFT_AES,
        AV4_SOFT_AES_DOUBLE,
        AV_MAX
    };

    enum AesNi {
        AESNI_AUTO,
        AESNI_ON,
        AESNI_OFF
    };

    static inline Options* i() { return m_self; }
    static Options *parse(int argc, char **argv);

    inline bool background() const                  { return m_background; }
    inline bool colors() const                      { return m_colors; }
    inline bool hugePages() const                   { return m_hugePages; }
    inline bool syslog() const                      { return m_syslog; }
    inline bool daemonized() const                  { return m_daemonized; }
    inline bool ccUseTls() const                    { return m_ccUseTls; }
    inline bool ccUseRemoteLogging() const          { return m_ccUseRemoteLogging; }
    inline bool ccUploadConfigOnStartup() const     { return m_ccUploadConfigOnStartup; }
    inline const char *fileName() const             { return m_fileName; }
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
    inline const char *ccKeyFile() const            { return m_ccKeyFile == nullptr ? "server.key" : m_ccKeyFile; }
    inline const char *ccCertFile() const           { return m_ccCertFile == nullptr ? "server.pem" : m_ccCertFile; }
    inline const std::vector<Url*> &pools() const   { return m_pools; }
    inline Algo algo() const                        { return m_algo; }
    inline PowVariant powVariant() const            { return m_powVariant; }
    inline bool aesni() const                       { return m_aesni == AESNI_ON; }
    inline size_t hashFactor() const                { return m_hashFactor; }
    inline int apiPort() const                      { return m_apiPort; }
    inline int donateLevel() const                  { return m_donateLevel; }
    inline int printTime() const                    { return m_printTime; }
    inline int priority() const                     { return m_priority; }
    inline int retries() const                      { return m_retries; }
    inline int retryPause() const                   { return m_retryPause; }
    inline size_t threads() const                   { return m_threads; }
    inline int ccUpdateInterval() const             { return m_ccUpdateInterval; }
    inline int ccPort() const                       { return m_ccPort; }
    inline size_t ccClientLogLinesHistory() const   { return m_ccClientLogLinesHistory; }
    inline int64_t affinity() const                 { return m_affinity; }
    inline int64_t multiHashThreadMask() const      { return m_multiHashThreadMask; }
    inline void setColors(bool colors)              { m_colors = colors; }

    inline static void release()                    { delete m_self; }

    const char *algoName() const;
    const char *algoShortName() const;

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
    void showDeprecateWarning(const char* deprecated, const char* newParam) const;
    void showVersion(void);

    bool setAlgo(const char *algo);
    bool parsePowVariant(const char *powVariant);

    void optimizeAlgorithmConfiguration();

    bool m_background;
    bool m_colors;
    bool m_hugePages;
    bool m_ready;
    bool m_safe;
    bool m_syslog;
    bool m_daemonized;
    bool m_ccUseTls;
    bool m_ccUseRemoteLogging;
    bool m_ccUploadConfigOnStartup;
    const char* m_fileName;
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
    char *m_ccKeyFile;
    char *m_ccCertFile;
    Algo m_algo;
    AlgoVariant m_algoVariant;
    AesNi m_aesni;
    PowVariant m_powVariant;
    size_t m_hashFactor;
    int m_apiPort;
    int m_donateLevel;
    size_t m_maxCpuUsage;
    int m_printTime;
    int m_priority;
    int m_retries;
    int m_retryPause;
    size_t m_threads;
    int m_ccUpdateInterval;
    int m_ccPort;
    size_t m_ccClientLogLinesHistory;
    int64_t m_affinity;
    int64_t m_multiHashThreadMask;
    std::vector<Url*> m_pools;
};

#endif /* __OPTIONS_H__ */
