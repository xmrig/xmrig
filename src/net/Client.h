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

#ifndef __CLIENT_H__
#define __CLIENT_H__


#include <jansson.h>
#include <uv.h>


#include "net/Job.h"


class Url;
class IClientListener;


class Client
{
public:
    enum SocketState {
        UnconnectedState,
        HostLookupState,
        ConnectingState,
        ConnectedState,
        ClosingState
    };

    constexpr static int kResponseTimeout  = 15 * 1000;
    constexpr static int kKeepAliveTimeout = 60 * 1000;

    Client(int id, IClientListener *listener);
    ~Client();

    void connect();
    void connect(const Url *url);
    void disconnect();
    void login(const char *user, const char *pass, const char *agent);
    void send(char *data);
    void setUrl(const Url *url);

    inline bool isReady() const              { return m_state == ConnectedState && m_failures == 0; }
    inline const char *host() const          { return m_host; }
    inline const Job &job() const            { return m_job; }
    inline int id() const                    { return m_id; }
    inline SocketState state() const         { return m_state; }
    inline uint16_t port() const             { return m_port; }
    inline void setKeepAlive(bool keepAlive) { m_keepAlive = keepAlive; }
    inline void setRetryPause(int ms)        { m_retryPause = ms; }

private:
    constexpr static size_t kRecvBufSize = 4096;

    bool parseJob(const json_t *params, int *code);
    bool parseLogin(const json_t *result, int *code);
    int resolve(const char *host);
    void close();
    void connect(struct sockaddr *addr);
    void parse(char *line, size_t len);
    void parseNotification(const char *method, const json_t *params, const json_t *error);
    void parseResponse(int64_t id, const json_t *result, const json_t *error);
    void ping();
    void reconnect();
    void setState(SocketState state);
    void startTimeout();

    static void onAllocBuffer(uv_handle_t *handle, size_t suggested_size, uv_buf_t *buf);
    static void onClose(uv_handle_t *handle);
    static void onConnect(uv_connect_t *req, int status);
    static void onRead(uv_stream_t *stream, ssize_t nread, const uv_buf_t *buf);
    static void onResolved(uv_getaddrinfo_t *req, int status, struct addrinfo *res);

    static Client *getClient(void *data);

    bool m_keepAlive;
    char *m_host;
    char m_rpcId[64];
    IClientListener *m_listener;
    int m_id;
    int m_retryPause;
    int64_t m_failures;
    int64_t m_sequence;
    Job m_job;
    size_t m_recvBufPos;
    SocketState m_state;
    struct addrinfo m_hints;
    uint16_t m_port;
    uv_buf_t m_recvBuf;
    uv_getaddrinfo_t m_resolver;
    uv_stream_t *m_stream;
    uv_tcp_t *m_socket;
    uv_timer_t m_keepAliveTimer;
    uv_timer_t m_responseTimer;
    uv_timer_t m_retriesTimer;
};


#endif /* __CLIENT_H__ */
