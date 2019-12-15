/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2019      jtgrassie   <https://github.com/jtgrassie>
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


#include <bitset>
#include <map>
#include <uv.h>
#include <vector>


#include "base/kernel/interfaces/IDnsListener.h"
#include "base/kernel/interfaces/ILineListener.h"
#include "base/net/stratum/BaseClient.h"
#include "base/net/stratum/Job.h"
#include "base/net/stratum/Pool.h"
#include "base/net/stratum/SubmitResult.h"
#include "base/net/tools/RecvBuf.h"
#include "base/net/tools/Storage.h"
#include "base/tools/Object.h"
#include "crypto/common/Algorithm.h"


using BIO = struct bio_st;


namespace xmrig {


class IClientListener;
class JobResult;


class Client : public BaseClient, public IDnsListener, public ILineListener
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(Client)

    constexpr static uint64_t kConnectTimeout  = 20 * 1000;
    constexpr static uint64_t kResponseTimeout = 20 * 1000;

#   ifdef XMRIG_FEATURE_TLS
    constexpr static size_t kInputBufferSize = 1024 * 16;
#   else
    constexpr static size_t kInputBufferSize = 1024 * 2;
#   endif

    Client(int id, const char *agent, IClientListener *listener);
    ~Client() override;

protected:
    bool disconnect() override;
    bool isTLS() const override;
    const char *tlsFingerprint() const override;
    const char *tlsVersion() const override;
    int64_t send(const rapidjson::Value &obj, Callback callback) override;
    int64_t send(const rapidjson::Value &obj) override;
    int64_t submit(const JobResult &result) override;
    void connect() override;
    void connect(const Pool &pool) override;
    void deleteLater() override;
    void tick(uint64_t now) override;

    void onResolved(const Dns &dns, int status) override;

    inline bool hasExtension(Extension extension) const noexcept override { return m_extensions.test(extension); }
    inline const char *mode() const override                              { return "pool"; }
    inline void onLine(char *line, size_t size) override                  { parse(line, size); }

private:
    class Tls;

    bool close();
    bool isCriticalError(const char *message);
    bool parseJob(const rapidjson::Value &params, int *code);
    bool parseLogin(const rapidjson::Value &result, int *code);
    bool send(BIO *bio);
    bool verifyAlgorithm(const Algorithm &algorithm, const char *algo) const;
    int resolve(const String &host);
    int64_t send(size_t size);
    void connect(sockaddr *addr);
    void handshake();
    void login();
    void onClose();
    void parse(char *line, size_t len);
    void parseExtensions(const rapidjson::Value &result);
    void parseNotification(const char *method, const rapidjson::Value &params, const rapidjson::Value &error);
    void parseResponse(int64_t id, const rapidjson::Value &result, const rapidjson::Value &error);
    void ping();
    void read(ssize_t nread);
    void reconnect();
    void setState(SocketState state);
    void startTimeout();

    inline const char *url() const                                { return m_pool.url(); }
    inline SocketState state() const                              { return m_state; }
    inline void setExtension(Extension ext, bool enable) noexcept { m_extensions.set(ext, enable); }
    template<Extension ext> inline bool has() const noexcept      { return m_extensions.test(ext); }

    static void onAllocBuffer(uv_handle_t *handle, size_t suggested_size, uv_buf_t *buf);
    static void onClose(uv_handle_t *handle);
    static void onConnect(uv_connect_t *req, int status);
    static void onRead(uv_stream_t *stream, ssize_t nread, const uv_buf_t *buf);

    static inline Client *getClient(void *data) { return m_storage.get(data); }

    char m_sendBuf[4096] = { 0 };
    const char *m_agent;
    Dns *m_dns;
    RecvBuf<kInputBufferSize> m_recvBuf;
    std::bitset<EXT_MAX> m_extensions;
    String m_rpcId;
    Tls *m_tls                  = nullptr;
    uint64_t m_expire           = 0;
    uint64_t m_jobs             = 0;
    uint64_t m_keepAlive        = 0;
    uintptr_t m_key             = 0;
    uv_stream_t *m_stream       = nullptr;
    uv_tcp_t *m_socket          = nullptr;

    static Storage<Client> m_storage;
};


template<> inline bool Client::has<Client::EXT_NICEHASH>() const noexcept  { return m_extensions.test(EXT_NICEHASH) || m_pool.isNicehash(); }
template<> inline bool Client::has<Client::EXT_KEEPALIVE>() const noexcept { return m_extensions.test(EXT_KEEPALIVE) || m_pool.keepAlive() > 0; }


} /* namespace xmrig */


#endif /* XMRIG_CLIENT_H */
