/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2019      Howard Chu  <https://github.com/hyc>
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

#ifndef XMRIG_DAEMONCLIENT_H
#define XMRIG_DAEMONCLIENT_H


#include <uv.h>


#include "base/kernel/interfaces/IDnsListener.h"
#include "base/kernel/interfaces/IHttpListener.h"
#include "base/kernel/interfaces/ITimerListener.h"
#include "base/net/stratum/BaseClient.h"
#include "base/tools/Object.h"
#include "base/tools/cryptonote/BlockTemplate.h"
#include "base/net/tools/Storage.h"


#include <memory>


namespace xmrig {


class DnsRequest;


class DaemonClient : public BaseClient, public IDnsListener, public ITimerListener, public IHttpListener
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(DaemonClient)

    DaemonClient(int id, IClientListener *listener);
    ~DaemonClient() override;

protected:
    bool disconnect() override;
    bool isTLS() const override;
    int64_t submit(const JobResult &result) override;
    void connect() override;
    void connect(const Pool &pool) override;

    void onHttpData(const HttpData &data) override;
    void onTimer(const Timer *timer) override;
    void onResolved(const DnsRecords& records, int status, const char* error) override;

    inline bool hasExtension(Extension) const noexcept override         { return false; }
    inline const char *mode() const override                            { return "daemon"; }
    inline const char *tlsFingerprint() const override                  { return m_tlsFingerprint; }
    inline const char *tlsVersion() const override                      { return m_tlsVersion; }
    inline int64_t send(const rapidjson::Value &, Callback) override    { return -1; }
    inline int64_t send(const rapidjson::Value &) override              { return -1; }
    void deleteLater() override;
    inline void tick(uint64_t) override                                 {}

private:
    bool isOutdated(uint64_t height, const char *hash) const;
    bool parseJob(const rapidjson::Value &params, int *code);
    bool parseResponse(int64_t id, const rapidjson::Value &result, const rapidjson::Value &error);
    int64_t getBlockTemplate();
    int64_t rpcSend(const rapidjson::Document &doc);
    void retry();
    void send(const char *path);
    void setState(SocketState state);

    enum {
        API_CRYPTONOTE_DEFAULT,
        API_MONERO,
        API_DERO,
    } m_apiVersion = API_MONERO;

    std::shared_ptr<IHttpListener> m_httpListener;
    String m_currentJobId;
    String m_blocktemplateStr;
    String m_blockhashingblob;
    String m_prevHash;
    String m_tlsFingerprint;
    String m_tlsVersion;
    Timer *m_timer;
    uint64_t m_blocktemplateRequestHeight = 0;
    String m_blocktemplateRequestHash;

    BlockTemplate m_blocktemplate;

private:
    static inline DaemonClient* getClient(void* data) { return m_storage.get(data); }

    uintptr_t m_key = 0;
    static Storage<DaemonClient> m_storage;

    static void onZMQConnect(uv_connect_t* req, int status);
    static void onZMQRead(uv_stream_t* stream, ssize_t nread, const uv_buf_t* buf);
    static void onZMQClose(uv_handle_t* handle);
    static void onZMQShutdown(uv_handle_t* handle);

    void ZMQConnected();
    bool ZMQWrite(const char* data, size_t size);
    void ZMQRead(ssize_t nread, const uv_buf_t* buf);
    void ZMQParse();
    bool ZMQClose(bool shutdown = false);

    std::shared_ptr<DnsRequest> m_dns;
    uv_tcp_t* m_ZMQSocket = nullptr;

    enum {
        ZMQ_NOT_CONNECTED,
        ZMQ_GREETING_1,
        ZMQ_GREETING_2,
        ZMQ_HANDSHAKE,
        ZMQ_CONNECTED,
        ZMQ_DISCONNECTING,
    } m_ZMQConnectionState = ZMQ_NOT_CONNECTED;

    std::vector<char> m_ZMQSendBuf;
    std::vector<char> m_ZMQRecvBuf;
};


} /* namespace xmrig */


#endif /* XMRIG_DAEMONCLIENT_H */
