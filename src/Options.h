/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
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


class Url;


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

    inline bool isReady() const         { return m_ready; }
    inline const char *pass() const     { return m_pass; }
    inline const char *user() const     { return m_user; }
    inline const Url *backupUrl() const { return m_backupUrl; }
    inline const Url *url() const       { return m_url; }

private:
    Options(int argc, char **argv);
    ~Options();

    static Options *m_self;

    bool parseArg(int key, char *arg);
    Url *parseUrl(const char *arg) const;
    void showUsage(int status) const;
    void showVersion(void);

    bool setAlgo(const char *algo);
    bool setUserpass(const char *userpass);

    bool m_background;
    bool m_colors;
    bool m_doubleHash;
    bool m_keepalive;
    bool m_nicehash;
    bool m_ready;
    bool m_safe;
    char *m_pass;
    char *m_user;
    int m_algo;
    int m_algoVariant;
    int m_donateLevel;
    int m_maxCpuUsage;
    int m_retries;
    int m_retryPause;
    int m_threads;
    int64_t m_affinity;
    Url *m_backupUrl;
    Url *m_url;
};

#endif /* __OPTIONS_H__ */
