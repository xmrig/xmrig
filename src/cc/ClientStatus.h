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

#ifndef __CLIENT_STATUS_H__
#define __CLIENT_STATUS_H__

#include <string>
#include <ctime>
#include "rapidjson/document.h"

class ClientStatus
{

public:
    enum Status {
        RUNNING,
        PAUSED,
        CONFIG_UPDATED
    };

public:
    ClientStatus();

    inline const char *toString (Status status)
    {
        return status_str[static_cast<int>(status)];
    }

    inline Status toStatus (const char *status)
    {
        const int n = sizeof(status_str) / sizeof(status_str[0]);
        for (int i = 0; i < n; ++i)
        {
            if (strcmp(status_str[i], status) == 0)
                return (Status) i;
        }
        return Status::RUNNING;
    }

    std::string getClientId() const;
    void setClientId(const std::string& clientId);

    std::string getCurrentPool() const;
    void setCurrentPool(const std::string& currentPool);

    std::string getCurrentAlgoName() const;
    void setCurrentAlgoName(const std::string &algoName);

    Status getCurrentStatus() const;
    void setCurrentStatus(Status currentStatus);

    double getHashrateShort() const;
    void setHashrateShort(double hashrateShort);

    double getHashrateMedium() const;
    void setHashrateMedium(double hashrateMedium);

    double getHashrateLong() const;
    void setHashrateLong(double hashrateLong);

    uint64_t getSharesGood() const;
    void setSharesGood(uint64_t sharesGood);

    uint64_t getSharesTotal() const;
    void setSharesTotal(uint64_t sharesTotal);

    uint64_t getHashesTotal() const;
    void setHashesTotal(uint64_t hashesTotal);

    void setHashrateHighest(double hashrateHighest);
    double getHashrateHighest() const;

    void setAvgTime(uint32_t avgTime);
    uint32_t getAvgTime() const;

    std::time_t getLastStatusUpdate() const;

    std::string toJsonString();
    rapidjson::Value toJson(rapidjson::MemoryPoolAllocator<rapidjson::CrtAllocator>& allocator);
    bool parseFromJson(const rapidjson::Document& document);


private:
    const char* status_str[3] = {
            "RUNNING",
            "PAUSED",
            "CONFIG_UPDATED"
    };

    Status m_currentStatus;

    std::string m_clientId;
    std::string m_currentPool;
    std::string m_currentAlgoName;

    double m_hashrateShort;
    double m_hashrateMedium;
    double m_hashrateLong;
    double m_hashrateHighest;

    uint64_t m_sharesGood;
    uint64_t m_sharesTotal;
    uint64_t m_hashesTotal;

    uint32_t m_avgTime;

    std::time_t m_lastStatusUpdate;
};

#endif /* __CLIENT_STATUS_H__ */
