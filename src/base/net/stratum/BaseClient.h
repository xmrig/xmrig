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

#ifndef XMRIG_BASECLIENT_H
#define XMRIG_BASECLIENT_H


#include <map>


#include "base/kernel/interfaces/IClient.h"
#include "base/net/stratum/Job.h"
#include "base/net/stratum/Pool.h"


namespace xmrig {


class IClientListener;
class SubmitResult;


class BaseClient : public IClient
{
public:
    BaseClient(int id, IClientListener *listener);

    inline bool isEnabled() const override                     { return m_enabled; }
    inline const Job &job() const override                     { return m_job; }
    inline const Pool &pool() const override                   { return m_pool; }
    inline const String &ip() const override                   { return m_ip; }
    inline int id() const override                             { return m_id; }
    inline void setAlgo(const Algorithm &algo) override        { m_pool.setAlgo(algo); }
    inline void setEnabled(bool enabled) override              { m_enabled = enabled; }
    inline void setPool(const Pool &pool) override             { if (pool.isValid()) { m_pool = pool; } }
    inline void setQuiet(bool quiet) override                  { m_quiet = quiet; }
    inline void setRetries(int retries) override               { m_retries = retries; }
    inline void setRetryPause(uint64_t ms) override            { m_retryPause = ms; }

protected:
    enum SocketState {
        UnconnectedState,
        HostLookupState,
        ConnectingState,
        ConnectedState,
        ClosingState
    };

    inline bool isQuiet() const { return m_quiet || m_failures >= m_retries; }

    bool handleSubmitResponse(int64_t id, const char *error = nullptr);

    bool m_quiet;
    IClientListener *m_listener;
    int m_id;
    int m_retries;
    int64_t m_failures;
    Job m_job;
    Pool m_pool;
    SocketState m_state;
    std::map<int64_t, SubmitResult> m_results;
    String m_ip;
    uint64_t m_retryPause;

    static int64_t m_sequence;

private:
    bool m_enabled;
};


} /* namespace xmrig */


#endif /* XMRIG_BASECLIENT_H */
