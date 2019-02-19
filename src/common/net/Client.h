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

#ifndef XMRIG_CLIENT_H
#define XMRIG_CLIENT_H


#include <map>
#include <uv.h>
#include <vector>


#include "base/net/Pool.h"
#include "common/crypto/Algorithm.h"
#include "common/net/Id.h"
#include "common/net/Job.h"
#include "common/net/Storage.h"
#include "common/net/SubmitResult.h"
#include "rapidjson/fwd.h"


typedef struct bio_st BIO;


namespace xmrig {


class IClientListener;
class JobResult;


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

    constexpr static int kResponseTimeout = 20 * 1000;

#   ifndef XMRIG_NO_TLS
    constexpr static int kInputBufferSize = 1024 * 16;
#   else
    constexpr static int kInputBufferSize = 1024 * 2;
#   endif

    Client(int id, const char *agent, IClientListener *listener);
    ~Client();

    bool disconnect();
    const char *tlsFingerprint() const;
    const char *tlsVersion() const;
    int64_t submit(const JobResult &result);
    void connect();
    void connect(const Pool &pool);
    void deleteLater();
    void setPool(const Pool &pool);
    void tick(uint64_t now);

    inline bool isReady() const                       { return m_state == ConnectedState && m_failures == 0; }
    inline const char *host() const                   { return m_pool.host(); }
    inline const char *ip() const                     { return m_ip; }
    inline const Job &job() const                     { return m_job; }
    inline int id() const                             { return m_id; }
    inline SocketState state() const                  { return m_state; }
    inline uint16_t port() const                      { return m_pool.port(); }
    inline void setAlgo(const Algorithm &algo)        { m_pool.setAlgo(algo); }
    inline void setQuiet(bool quiet)                  { m_quiet = quiet; }
    inline void setRetries(int retries)               { m_retries = retries; }
    inline void setRetryPause(int ms)                 { m_retryPause = ms; }

private:
    class Tls;


    enum Extensions {
        NicehashExt  = 1,
        AlgoExt      = 2
    };

    bool close();
    bool isCriticalError(const char *message);
    bool isTLS() const;
    bool parseJob(const rapidjson::Value &params, int *code);
    bool parseLogin(const rapidjson::Value &result, int *code);
    bool send(BIO *bio);
    bool verifyAlgorithm(const Algorithm &algorithm) const;
    int resolve(const char *host);
    int64_t send(const rapidjson::Document &doc);
    int64_t send(size_t size);
    void connect(const std::vector<addrinfo*> &ipv4, const std::vector<addrinfo*> &ipv6);
    void connect(sockaddr *addr);
    void handshake();
    void login();
    void onClose();
    void parse(char *line, size_t len);
    void parseExtensions(const rapidjson::Value &value);
    void parseNotification(const char *method, const rapidjson::Value &params, const rapidjson::Value &error);
    void parseResponse(int64_t id, const rapidjson::Value &result, const rapidjson::Value &error);
    void ping();
    void read();
    void reconnect();
    void setState(SocketState state);
    void startTimeout();

    inline bool isQuiet() const { return m_quiet || m_failures >= m_retries; }

    static void onAllocBuffer(uv_handle_t *handle, size_t suggested_size, uv_buf_t *buf);
    static void onClose(uv_handle_t *handle);
    static void onConnect(uv_connect_t *req, int status);
    static void onRead(uv_stream_t *stream, ssize_t nread, const uv_buf_t *buf);
    static void onResolved(uv_getaddrinfo_t *req, int status, struct addrinfo *res);

    static inline Client *getClient(void *data) { return m_storage.get(data); }

    addrinfo m_hints;
    bool m_ipv6;
    bool m_nicehash;
    bool m_quiet;
    char m_buf[kInputBufferSize];
    char m_ip[46];
    char m_sendBuf[2048];
    const char *m_agent;
    IClientListener *m_listener;
    int m_extensions;
    int m_id;
    int m_retries;
    int m_retryPause;
    int64_t m_failures;
    Job m_job;
    Pool m_pool;
    size_t m_recvBufPos;
    SocketState m_state;
    std::map<int64_t, SubmitResult> m_results;
    Tls *m_tls;
    uint64_t m_expire;
    uint64_t m_jobs;
    uint64_t m_keepAlive;
    uintptr_t m_key;
    uv_buf_t m_recvBuf;
    uv_getaddrinfo_t m_resolver;
    uv_stream_t *m_stream;
    uv_tcp_t *m_socket;
    Id m_rpcId;

    static int64_t m_sequence;
    static Storage<Client> m_storage;
};


} /* namespace xmrig */


#endif /* XMRIG_CLIENT_H */
