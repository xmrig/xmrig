/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2019      jtgrassie   <https://github.com/jtgrassie>
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

#ifndef XMRIG_SELFSELECTCLIENT_H
#define XMRIG_SELFSELECTCLIENT_H


#include "base/kernel/interfaces/IClient.h"
#include "base/kernel/interfaces/IClientListener.h"
#include "base/net/http/HttpListener.h"
#include "base/net/stratum/Job.h"
#include "base/tools/Object.h"


#include <memory>


namespace xmrig {


class SelfSelectClient : public IClient, public IClientListener, public IHttpListener
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(SelfSelectClient)

    SelfSelectClient(int id, const char *agent, IClientListener *listener);
    ~SelfSelectClient() override;

protected:
    // IClient
    inline bool disconnect() override                                               { return m_client->disconnect(); }
    inline bool hasExtension(Extension extension) const noexcept override           { return m_client->hasExtension(extension); }
    inline bool isEnabled() const override                                          { return m_client->isEnabled(); }
    inline bool isTLS() const override                                              { return m_client->isTLS(); }
    inline const char *mode() const override                                        { return m_client->mode(); }
    inline const char *tag() const override                                         { return m_client->tag(); }
    inline const char *tlsFingerprint() const override                              { return m_client->tlsFingerprint(); }
    inline const char *tlsVersion() const override                                  { return m_client->tlsVersion(); }
    inline const Job &job() const override                                          { return m_job; }
    inline const Pool &pool() const override                                        { return m_client->pool(); }
    inline const String &ip() const override                                        { return m_client->ip(); }
    inline int id() const override                                                  { return m_client->id(); }
    inline int64_t send(const rapidjson::Value &obj, Callback callback) override    { return m_client->send(obj, callback); }
    inline int64_t send(const rapidjson::Value &obj) override                       { return m_client->send(obj); }
    inline int64_t sequence() const override                                        { return m_client->sequence(); }
    inline int64_t submit(const JobResult &result) override                         { return m_client->submit(result); }
    inline void connect() override                                                  { m_client->connect(); }
    inline void connect(const Pool &pool) override                                  { m_client->connect(pool); }
    inline void deleteLater() override                                              { m_client->deleteLater(); }
    inline void setAlgo(const Algorithm &algo) override                             { m_client->setAlgo(algo); }
    inline void setEnabled(bool enabled) override                                   { m_client->setEnabled(enabled); }
    inline void setPool(const Pool &pool) override                                  { m_client->setPool(pool); }
    inline void setProxy(const ProxyUrl &proxy) override                            { m_client->setProxy(proxy); }
    inline void setQuiet(bool quiet) override                                       { m_client->setQuiet(quiet); m_quiet = quiet;  }
    inline void setRetries(int retries) override                                    { m_client->setRetries(retries); m_retries = retries; }
    inline void setRetryPause(uint64_t ms) override                                 { m_client->setRetryPause(ms); m_retryPause = ms; }

    void tick(uint64_t now) override;

    // IClientListener
    inline void onClose(IClient *, int failures) override                                           { m_listener->onClose(this, failures); setState(IdleState); m_active = false; }
    inline void onLoginSuccess(IClient *) override                                                  { m_listener->onLoginSuccess(this); setState(IdleState); m_active = true; }
    inline void onResultAccepted(IClient *, const SubmitResult &result, const char *error) override { m_listener->onResultAccepted(this, result, error); }
    inline void onVerifyAlgorithm(const IClient *, const Algorithm &algorithm, bool *ok) override   { m_listener->onVerifyAlgorithm(this, algorithm, ok); }

    void onJobReceived(IClient *, const Job &job, const rapidjson::Value &params) override;
    void onLogin(IClient *, rapidjson::Document &doc, rapidjson::Value &params) override;

    // IHttpListener
    void onHttpData(const HttpData &data) override;

private:
    enum State {
        IdleState,
        WaitState,
        RetryState
    };

    inline bool isQuiet() const { return m_quiet || m_failures >= m_retries; }

    bool parseResponse(int64_t id, rapidjson::Value &result, const rapidjson::Value &error);
    void getBlockTemplate();
    void retry();
    void setState(State state);
    void submitBlockTemplate(rapidjson::Value &result);

    bool m_active           = false;
    bool m_quiet            = false;
    IClient *m_client;
    IClientListener *m_listener;
    int m_retries           = 5;
    int64_t m_failures      = 0;
    int64_t m_sequence      = 1;
    Job m_job;
    State m_state           = IdleState;
    std::shared_ptr<IHttpListener> m_httpListener;
    uint64_t m_retryPause   = 5000;
    uint64_t m_timestamp    = 0;
};


} /* namespace xmrig */


#endif /* XMRIG_SELFSELECTCLIENT_H */
