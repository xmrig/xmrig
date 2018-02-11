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
        ALGO_CRYPTONIGHT
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

    inline bool background() const                { return m_background; }
    inline bool colors() const                    { return m_colors; }
    inline bool doubleHash() const                { return m_doubleHash; }
    inline bool hugePages() const                 { return m_hugePages; }
    inline bool syslog() const                    { return m_syslog; }
    inline const char *apiToken() const           { return m_apiToken; }
    inline const char *apiWorkerId() const        { return m_apiWorkerId; }
    inline const char *logFile() const            { return m_logFile; }
    inline const char *userAgent() const          { return m_userAgent; }
    inline const std::vector<Url*> &pools() const { return m_pools; }
    inline int algo() const                       { return m_algo; }
    inline int algoVariant() const                { return m_algoVariant; }
    inline int apiPort() const                    { return m_apiPort; }
    inline int donateLevel() const                { return m_donateLevel; }
    inline int printTime() const                  { return m_printTime; }
    inline int priority() const                   { return m_priority; }
    inline int retries() const                    { return m_retries; }
    inline int retryPause() const                 { return m_retryPause; }
    inline int threads() const                    { return m_threads; }
    inline int64_t affinity() const               { return m_affinity; }

    inline static void release()                  { delete m_self; }

    const char *algoName() const;

private:
    Options(int argc, char **argv);
    ~Options();

    inline bool isReady() const { return m_ready; }

    static Options *m_self;

    bool parseArg(int key, const char *arg);
    bool parseArg(int key, uint64_t arg);
    bool parseBoolean(int key, bool enable);
    Url *parseUrl(const char *arg) const;

    bool setAlgo(const char *algo);

    int getAlgoVariant() const;

    bool m_background;
    bool m_colors;
    bool m_doubleHash;
    bool m_hugePages;
    bool m_ready;
    bool m_safe;
    bool m_syslog;
    char *m_apiToken;
    char *m_apiWorkerId;
    char *m_logFile;
    char *m_userAgent;
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
    int64_t m_affinity;
    std::vector<Url*> m_pools;
};

#endif /* __OPTIONS_H__ */
