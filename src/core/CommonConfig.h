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

#ifndef __COMMONCONFIG_H__
#define __COMMONCONFIG_H__


#include <vector>


#include "interfaces/IConfig.h"
#include "xmrig.h"


class Url;


namespace xmrig {


class CommonConfig : public IConfig
{
public:
    CommonConfig();
    ~CommonConfig();

    const char *algoName() const;

    inline Algo algorithm() const                  { return m_algorithm; }
    inline bool isApiIPv6() const                  { return m_apiIPv6; }
    inline bool isApiRestricted() const            { return m_apiRestricted; }
    inline bool isBackground() const               { return m_background; }
    inline bool isColors() const                   { return m_colors; }
    inline bool isSyslog() const                   { return m_syslog; }
    inline const char *apiToken() const            { return m_apiToken; }
    inline const char *apiWorkerId() const         { return m_apiWorkerId; }
    inline const char *logFile() const             { return m_logFile; }
    inline const char *userAgent() const           { return m_userAgent; }
    inline const std::vector<Url*> &pools() const  { return m_pools; }
    inline int apiPort() const                     { return m_apiPort; }
    inline int donateLevel() const                 { return m_donateLevel; }
    inline int printTime() const                   { return m_printTime; }
    inline int retries() const                     { return m_retries; }
    inline int retryPause() const                  { return m_retryPause; }
    inline void setColors(bool colors)             { m_colors = colors; }

    inline bool isWatch() const override           { return m_watch && m_fileName; }
    inline const char *fileName() const override   { return m_fileName; }

protected:
    bool adjust() override;
    bool isValid() const override;
    bool parseBoolean(int key, bool enable) override;
    bool parseString(int key, const char *arg) override;
    bool parseUint64(int key, uint64_t arg) override;
    bool save() override;
    void setFileName(const char *fileName) override;

    Algo m_algorithm;
    bool m_adjusted;
    bool m_apiIPv6;
    bool m_apiRestricted;
    bool m_background;
    bool m_colors;
    bool m_syslog;
    bool m_watch;
    char *m_apiToken;
    char *m_apiWorkerId;
    char *m_fileName;
    char *m_logFile;
    char *m_userAgent;
    int m_apiPort;
    int m_donateLevel;
    int m_printTime;
    int m_retries;
    int m_retryPause;
    std::vector<Url*> m_pools;

private:
    bool parseInt(int key, int arg);
    void setAlgo(const char *algo);
};


} /* namespace xmrig */

#endif /* __COMMONCONFIG_H__ */
