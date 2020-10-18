/* XMRig
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_NULLCLIENT_H
#define XMRIG_NULLCLIENT_H


#include "base/net/stratum/Client.h"


namespace xmrig {


class NullClient : public IClient
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(NullClient)

    NullClient(IClientListener* listener);
    ~NullClient() override = default;

    inline bool disconnect() override                                               { return true; }
    inline bool hasExtension(Extension extension) const noexcept override           { return false; }
    inline bool isEnabled() const override                                          { return true; }
    inline bool isTLS() const override                                              { return false; }
    inline const char *mode() const override                                        { return "benchmark"; }
    inline const char *tag() const override                                         { return "null"; }
    inline const char *tlsFingerprint() const override                              { return nullptr; }
    inline const char *tlsVersion() const override                                  { return nullptr; }
    inline const Job &job() const override                                          { return m_job; }
    inline const Pool &pool() const override                                        { return m_pool; }
    inline const String &ip() const override                                        { return m_ip; }
    inline int id() const override                                                  { return 0; }
    inline int64_t send(const rapidjson::Value& obj, Callback callback) override    { return 0; }
    inline int64_t send(const rapidjson::Value& obj) override                       { return 0; }
    inline int64_t sequence() const override                                        { return 0; }
    inline int64_t submit(const JobResult& result) override                         { return 0; }
    inline void connect(const Pool& pool) override                                  { setPool(pool); }
    inline void deleteLater() override                                              {}
    inline void setAlgo(const Algorithm& algo) override                             {}
    inline void setEnabled(bool enabled) override                                   {}
    inline void setProxy(const ProxyUrl& proxy) override                            {}
    inline void setQuiet(bool quiet) override                                       {}
    inline void setRetries(int retries) override                                    {}
    inline void setRetryPause(uint64_t ms) override                                 {}
    inline void tick(uint64_t now) override                                         {}

    void connect() override;
    void setPool(const Pool& pool) override;

private:
    IClientListener* m_listener;
    Job m_job;
    Pool m_pool;
    String m_ip;
};


} /* namespace xmrig */


#endif /* XMRIG_NULLCLIENT_H */
