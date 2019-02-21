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

#ifndef XMRIG_COMMONCONFIG_H
#define XMRIG_COMMONCONFIG_H


#include "base/net/Pools.h"
#include "base/tools/String.h"
#include "common/interfaces/IConfig.h"
#include "common/xmrig.h"


namespace xmrig {


class CommonConfig : public IConfig
{
public:
    CommonConfig();

    inline bool isApiIPv6() const                  { return m_apiIPv6; }
    inline bool isApiRestricted() const            { return m_apiRestricted; }
    inline bool isAutoSave() const                 { return m_autoSave; }
    inline bool isBackground() const               { return m_background; }
    inline bool isDryRun() const                   { return m_dryRun; }
    inline bool isSyslog() const                   { return m_syslog; }
    inline const char *apiId() const               { return m_apiId.data(); }
    inline const char *apiToken() const            { return m_apiToken.data(); }
    inline const char *apiWorkerId() const         { return m_apiWorkerId.data(); }
    inline const char *logFile() const             { return m_logFile.data(); }
    inline const char *userAgent() const           { return m_userAgent.data(); }
    inline const Pools &pools() const              { return m_pools; }
    inline int apiPort() const                     { return m_apiPort; }
    inline int donateLevel() const                 { return m_donateLevel; }
    inline int printTime() const                   { return m_printTime; }

    inline bool isWatch() const override               { return m_watch && !m_fileName.isNull(); }
    inline const Algorithm &algorithm() const override { return m_algorithm; }
    inline const String &fileName() const override     { return m_fileName; }

    bool save() override;

    bool isColors() const;
    void printAPI();
    void printPools();
    void printVersions();

protected:
    enum State {
        NoneState,
        ReadyState,
        ErrorState
    };

    bool finalize() override;
    bool parseBoolean(int key, bool enable) override;
    bool parseString(int key, const char *arg) override;
    bool parseUint64(int key, uint64_t arg) override;
    void parseJSON(const rapidjson::Document &doc) override;
    void setFileName(const char *fileName) override;

    Algorithm m_algorithm;
    bool m_adjusted;
    bool m_apiIPv6;
    bool m_apiRestricted;
    bool m_autoSave;
    bool m_background;
    bool m_dryRun;
    bool m_syslog;
    bool m_watch;
    int m_apiPort;
    int m_donateLevel;
    int m_printTime;
    Pools m_pools;
    State m_state;
    String m_apiId;
    String m_apiToken;
    String m_apiWorkerId;
    String m_fileName;
    String m_logFile;
    String m_userAgent;

private:
    bool parseInt(int key, int arg);
};


} /* namespace xmrig */

#endif /* XMRIG_COMMONCONFIG_H */
