/* XMRig
 * Copyright (c) 2019      Howard Chu  <https://github.com/hyc>
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include "base/kernel/interfaces/IDnsListener.h"
#include "base/kernel/interfaces/IHttpListener.h"
#include "base/kernel/interfaces/ITimerListener.h"
#include "base/net/stratum/BaseClient.h"
#include "base/net/tools/Storage.h"
#include "base/tools/cryptonote/BlockTemplate.h"
#include "base/tools/cryptonote/WalletAddress.h"


#include <memory>


using uv_buf_t      = struct uv_buf_t;
using uv_connect_t  = struct uv_connect_s;
using uv_handle_t   = struct uv_handle_s;
using uv_stream_t   = struct uv_stream_s;
using uv_tcp_t      = struct uv_tcp_s;


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
    void setPool(const Pool &pool) override;

    void onHttpData(const HttpData &data) override;
    void onTimer(const Timer *timer) override;
    void onResolved(const DnsRecords &records, int status, const char* error) override;

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

    BlockTemplate m_blocktemplate;
    Coin m_coin;
    std::shared_ptr<IHttpListener> m_httpListener;
    String m_blockhashingblob;
    String m_blocktemplateRequestHash;
    String m_blocktemplateStr;
    String m_currentJobId;
    String m_prevHash;
    String m_tlsFingerprint;
    String m_tlsVersion;
    Timer *m_timer;
    uint64_t m_blocktemplateRequestHeight = 0;
    WalletAddress m_walletAddress;

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
