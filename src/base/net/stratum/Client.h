/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2019      jtgrassie   <https://github.com/jtgrassie>
 * Copyright 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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
#include "base/net/tools/LineReader.h"
#include "base/net/tools/Storage.h"
#include "base/tools/Object.h"


using BIO = struct bio_st;


namespace xmrig {


class DnsRequest;
class IClientListener;
class JobResult;


class Client : public BaseClient, public IDnsListener, public ILineListener
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(Client)

    constexpr static uint64_t kConnectTimeout   = 20 * 1000;
    constexpr static uint64_t kResponseTimeout  = 20 * 1000;
    constexpr static size_t kMaxSendBufferSize  = 1024 * 16;

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

    void onResolved(const DnsRecords &records, int status, const char *error) override;

    inline bool hasExtension(Extension extension) const noexcept override   { return m_extensions.test(extension); }
    inline const char *mode() const override                                { return "pool"; }
    inline void onLine(char *line, size_t size) override                    { parse(line, size); }

    inline const char *agent() const                                        { return m_agent; }
    inline const char *url() const                                          { return m_pool.url(); }
    inline const String &rpcId() const                                      { return m_rpcId; }
    inline void setRpcId(const char *id)                                    { m_rpcId = id; }

    virtual bool parseLogin(const rapidjson::Value &result, int *code);
    virtual void login();
    virtual void parseNotification(const char* method, const rapidjson::Value& params, const rapidjson::Value& error);

    bool close();
    virtual void onClose();

private:
    class Socks5;
    class Tls;

    bool isCriticalError(const char *message);
    bool parseJob(const rapidjson::Value &params, int *code);
    bool send(BIO *bio);
    bool verifyAlgorithm(const Algorithm &algorithm, const char *algo) const;
    bool write(const uv_buf_t &buf);
    int resolve(const String &host);
    int64_t send(size_t size);
    void connect(const sockaddr *addr);
    void handshake();
    void parse(char *line, size_t len);
    void parseExtensions(const rapidjson::Value &result);
    void parseResponse(int64_t id, const rapidjson::Value &result, const rapidjson::Value &error);
    void ping();
    void read(ssize_t nread, const uv_buf_t *buf);
    void reconnect();
    void setState(SocketState state);
    void startTimeout();

    inline SocketState state() const                                { return m_state; }
    inline uv_stream_t *stream() const                              { return reinterpret_cast<uv_stream_t *>(m_socket); }
    inline void setExtension(Extension ext, bool enable) noexcept   { m_extensions.set(ext, enable); }
    template<Extension ext> inline bool has() const noexcept        { return m_extensions.test(ext); }

    static void onClose(uv_handle_t *handle);
    static void onConnect(uv_connect_t *req, int status);
    static void onRead(uv_stream_t *stream, ssize_t nread, const uv_buf_t *buf);

    static inline Client *getClient(void *data) { return m_storage.get(data); }

    const char *m_agent;
    LineReader m_reader;
    Socks5 *m_socks5            = nullptr;
    std::bitset<EXT_MAX> m_extensions;
    std::shared_ptr<DnsRequest> m_dns;
    std::vector<char> m_sendBuf;
    std::vector<char> m_tempBuf;
    String m_rpcId;
    Tls *m_tls                  = nullptr;
    uint64_t m_expire           = 0;
    uint64_t m_jobs             = 0;
    uint64_t m_keepAlive        = 0;
    uintptr_t m_key             = 0;
    uv_tcp_t *m_socket          = nullptr;

    static Storage<Client> m_storage;
};


template<> inline bool Client::has<Client::EXT_NICEHASH>() const noexcept  { return m_extensions.test(EXT_NICEHASH) || m_pool.isNicehash(); }
template<> inline bool Client::has<Client::EXT_KEEPALIVE>() const noexcept { return m_extensions.test(EXT_KEEPALIVE) || m_pool.keepAlive() > 0; }


} /* namespace xmrig */


#endif /* XMRIG_CLIENT_H */
