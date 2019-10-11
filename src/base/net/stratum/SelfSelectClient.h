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

#ifndef XMRIG_SELFSELECTCLIENT_H
#define XMRIG_SELFSELECTCLIENT_H


#include "base/kernel/interfaces/IClientListener.h"
#include "base/tools/Object.h"
#include "base/kernel/interfaces/IClient.h"


namespace xmrig {


class SelfSelectClient : public IClient, public IClientListener
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(SelfSelectClient)

    SelfSelectClient(int id, const char *agent, IClientListener *listener);
    ~SelfSelectClient() override;

protected:
    // IClient
    bool disconnect() override                                          { return m_client->disconnect(); }
    bool hasExtension(Extension extension) const noexcept override      { return m_client->hasExtension(extension); }
    bool isEnabled() const override                                     { return m_client->isEnabled(); }
    bool isTLS() const override                                         { return m_client->isTLS(); }
    const char *mode() const override                                   { return m_client->mode(); }
    const char *tlsFingerprint() const override                         { return m_client->tlsFingerprint(); }
    const char *tlsVersion() const override                             { return m_client->tlsVersion(); }
    const Job &job() const override                                     { return m_client->job(); }
    const Pool &pool() const override                                   { return m_client->pool(); }
    const String &ip() const override                                   { return m_client->ip(); }
    int id() const override                                             { return m_client->id(); }
    int64_t submit(const JobResult &result) override                    { return m_client->submit(result); }
    void connect() override                                             { m_client->connect(); }
    void connect(const Pool &pool) override                             { m_client->connect(pool); }
    void deleteLater() override                                         { m_client->deleteLater(); }
    void setAlgo(const Algorithm &algo) override                        { m_client->setAlgo(algo); }
    void setEnabled(bool enabled) override                              { m_client->setEnabled(enabled); }
    void setPool(const Pool &pool) override                             { m_client->setPool(pool); }
    void setQuiet(bool quiet) override                                  { m_client->setQuiet(quiet); }
    void setRetries(int retries) override                               { m_client->setRetries(retries); }
    void setRetryPause(uint64_t ms) override                            { m_client->setRetryPause(ms); }
    void tick(uint64_t now) override                                    { m_client->tick(now); }

    // IClientListener
    void onClose(IClient *, int failures) override                                           { m_listener->onClose(this, failures); }
    void onJobReceived(IClient *, const Job &job, const rapidjson::Value &params) override   { m_listener->onJobReceived(this, job, params); }
    void onLogin(IClient *, rapidjson::Document &doc, rapidjson::Value &params) override     { m_listener->onLogin(this, doc, params); }
    void onLoginSuccess(IClient *) override                                                  { m_listener->onLoginSuccess(this); }
    void onResultAccepted(IClient *, const SubmitResult &result, const char *error) override { m_listener->onResultAccepted(this, result, error); }
    void onVerifyAlgorithm(const IClient *, const Algorithm &algorithm, bool *ok) override   { m_listener->onVerifyAlgorithm(this, algorithm, ok); }

private:
    IClient *m_client;
    IClientListener *m_listener;
};


} /* namespace xmrig */


#endif /* XMRIG_SELFSELECTCLIENT_H */
