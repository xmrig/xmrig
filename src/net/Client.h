/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
 * Copyright 2018-     BenDr0id    <ben@graef.in>
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

#ifndef __CLIENT_H__
#define __CLIENT_H__


#include <map>
#include <uv.h>
#include <memory>


#include "net/Job.h"
#include "net/SubmitResult.h"
#include "net/Url.h"
#include "rapidjson/fwd.h"

extern "C"
{
#include "net.h"

#ifndef XMRIG_NO_SSL_TLS
#include "tls.h"
#endif
}

class IClientListener;
class JobResult;


class Client
{
public:
    constexpr static int kResponseTimeout  = 20 * 1000;
    constexpr static int kKeepAliveTimeout = 60 * 1000;

    Client(int id, const char *agent, IClientListener *listener);
    ~Client();

    int64_t submit(const JobResult &result);
    void connect();
    void connect(const Url *url);
    void disconnect();
    void setUrl(const Url *url);
    void tick(uint64_t now);

    inline const char *host() const          { return m_url.host(); }
    inline const Job &job() const            { return m_job; }
    inline int id() const                    { return m_id; }
    inline uint16_t port() const             { return m_url.port(); }
    inline void setQuiet(bool quiet)         { m_quiet = quiet; }
    inline void setRetryPause(int ms)        { m_retryPause = ms; }

private:
    bool isCriticalError(const char *message);
    bool parseJob(const rapidjson::Value &params, int *code);
    bool parseLogin(const rapidjson::Value &result, int *code);
    int64_t send(size_t size);
    void close();
    void login();
    void parse(char *line, size_t len);
    void parseNotification(const char *method, const rapidjson::Value &params, const rapidjson::Value &error);
    void parseResponse(int64_t id, const rapidjson::Value &result, const rapidjson::Value &error);
    void ping();
    void reconnect();
    void startTimeout();

    static void onRead(net_t *net, size_t read, char *buf);
    static void onConnect(net_t *net);
    static void onError(net_t *net, int err, char *errStr);

    static inline Client *getClient(void *data) { return static_cast<Client*>(data); }

    bool m_quiet;
    char m_buf[2048];
    char m_rpcId[64];
    char m_sendBuf[768];
    const char *m_agent;
    IClientListener *m_listener;
    int m_id;
    int m_retryPause;
    int64_t m_failures;
    Job m_job;
    size_t m_recvBufPos;
    static int64_t m_sequence;
    std::map<int64_t, SubmitResult> m_results;
    uint64_t m_expire;
    Url m_url;
    uv_buf_t m_recvBuf;

    net_t* m_net;

#   ifndef XMRIG_PROXY_PROJECT
    uv_timer_t m_keepAliveTimer;
#   endif

};


#endif /* __CLIENT_H__ */
