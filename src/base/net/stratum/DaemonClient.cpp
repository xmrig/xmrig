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


#include <uv.h>


#include "base/net/stratum/DaemonClient.h"
#include "3rdparty/rapidjson/document.h"
#include "3rdparty/rapidjson/error/en.h"
#include "base/io/json/Json.h"
#include "base/io/json/JsonRequest.h"
#include "base/io/log/Log.h"
#include "base/kernel/interfaces/IClientListener.h"
#include "base/net/dns/Dns.h"
#include "base/net/dns/DnsRecords.h"
#include "base/net/http/Fetch.h"
#include "base/net/http/HttpData.h"
#include "base/net/http/HttpListener.h"
#include "base/net/stratum/SubmitResult.h"
#include "base/net/tools/NetBuffer.h"
#include "base/tools/bswap_64.h"
#include "base/tools/Cvt.h"
#include "base/tools/Timer.h"
#include "base/tools/cryptonote/Signatures.h"
#include "net/JobResult.h"


#ifdef XMRIG_FEATURE_TLS
#include <openssl/ssl.h>
#endif


#include <algorithm>
#include <cassert>
#include <random>


namespace xmrig {


Storage<DaemonClient> DaemonClient::m_storage;


static const char* kBlocktemplateBlob       = "blocktemplate_blob";
static const char* kBlockhashingBlob        = "blockhashing_blob";
static const char* kLastError               = "lasterror";
static const char *kGetHeight               = "/getheight";
static const char *kGetInfo                 = "/getinfo";
static const char *kHash                    = "hash";
static const char *kHeight                  = "height";
static const char *kJsonRPC                 = "/json_rpc";

static constexpr size_t kBlobReserveSize    = 8;

static const char kZMQGreeting[64] = { static_cast<char>(-1), 0, 0, 0, 0, 0, 0, 0, 0, 127, 3, 0, 'N', 'U', 'L', 'L' };
static constexpr size_t kZMQGreetingSize1 = 11;

static const char kZMQHandshake[] = "\4\x19\5READY\xbSocket-Type\0\0\0\3SUB";
static const char kZMQSubscribe[] = "\0\x18\1json-minimal-chain_main";

static const char kWSSLogin[] = "\
GET /ws/%s HTTP/1.1\r\n\
Host: %s\r\n\
Upgrade: websocket\r\n\
Connection: Upgrade\r\n\
Sec-WebSocket-Key: %s\r\n\
Sec-WebSocket-Version: 13\r\n\r\n";

} // namespace xmrig


xmrig::DaemonClient::DaemonClient(int id, IClientListener *listener) :
    BaseClient(id, listener)
{
    m_httpListener  = std::make_shared<HttpListener>(this);
    m_timer         = new Timer(this);
    m_key           = m_storage.add(this);
}


xmrig::DaemonClient::~DaemonClient()
{
    delete m_timer;
    delete m_ZMQSocket;
#   ifdef XMRIG_FEATURE_TLS
    delete m_wss.m_socket;
#   endif
}


void xmrig::DaemonClient::deleteLater()
{
#   ifdef XMRIG_FEATURE_TLS
    if (m_pool.isWSS()) {
        WSSClose(true);
    }
    else
#   endif
    if (m_pool.zmq_port() >= 0) {
        ZMQClose(true);
    }
    else {
        delete this;
    }
}


bool xmrig::DaemonClient::disconnect()
{
    if (m_state != UnconnectedState) {
        setState(UnconnectedState);
    }

    return true;
}


bool xmrig::DaemonClient::isTLS() const
{
#   ifdef XMRIG_FEATURE_TLS
    return m_pool.isTLS();
#   else
    return false;
#   endif
}


bool xmrig::DaemonClient::isWSS() const
{
    return m_pool.isWSS();
}


int64_t xmrig::DaemonClient::submit(const JobResult &result)
{
    if (result.jobId != m_currentJobId) {
        return -1;
    }

    char *data = (m_apiVersion == API_DERO) ? m_blockhashingblob.data() : m_blocktemplateStr.data();

    const size_t sig_offset = m_job.nonceOffset() + m_job.nonceSize();

#   ifdef XMRIG_PROXY_PROJECT

    memcpy(data + m_job.nonceOffset() * 2, result.nonce, 8);

    if (m_blocktemplate.hasMinerSignature() && result.sig) {
        memcpy(data + sig_offset * 2, result.sig, 64 * 2);
        memcpy(data + m_blocktemplate.offset(BlockTemplate::TX_PUBKEY_OFFSET) * 2, result.sig_data, 32 * 2);
        memcpy(data + m_blocktemplate.offset(BlockTemplate::EPH_PUBLIC_KEY_OFFSET) * 2, result.sig_data + 32 * 2, 32 * 2);
    }

    if (result.extra_nonce >= 0) {
        Cvt::toHex(data + m_blocktemplate.offset(BlockTemplate::TX_EXTRA_NONCE_OFFSET) * 2, 8, reinterpret_cast<const uint8_t*>(&result.extra_nonce), 4);
    }

#   else

    Cvt::toHex(data + m_job.nonceOffset() * 2, 8, reinterpret_cast<const uint8_t*>(&result.nonce), 4);

#   ifdef XMRIG_FEATURE_TLS
    if (m_pool.isWSS() && (m_apiVersion == API_DERO) && (m_pool.algorithm().id() == Algorithm::ASTROBWT_DERO_2)) {
        char buf[256];
        const int n = snprintf(buf, sizeof(buf), "{\"jobid\":\"%s\",\"mbl_blob\":\"%s\"}", m_job.id().data(), data);
        if (0 <= n && n < static_cast<int>(sizeof(buf))) {
            return WSSWrite(buf, n) ? 1 : -1;
        }
        return -1;
    }
#   endif

    if (m_blocktemplate.hasMinerSignature()) {
        Cvt::toHex(data + sig_offset * 2, 128, result.minerSignature(), 64);
    }

#   endif

    using namespace rapidjson;
    Document doc(kObjectType);

    Value params(kArrayType);
    if (m_apiVersion == API_DERO) {
        params.PushBack(m_blocktemplateStr.toJSON(), doc.GetAllocator());
        params.PushBack(m_blockhashingblob.toJSON(), doc.GetAllocator());
    }
    else {
        params.PushBack(m_blocktemplateStr.toJSON(), doc.GetAllocator());
    }

    JsonRequest::create(doc, m_sequence, "submitblock", params);

#   ifdef XMRIG_PROXY_PROJECT
    m_results[m_sequence] = SubmitResult(m_sequence, result.diff, result.actualDiff(), result.id, 0);
#   else
    m_results[m_sequence] = SubmitResult(m_sequence, result.diff, result.actualDiff(), 0, result.backend);
#   endif

    return rpcSend(doc);
}


void xmrig::DaemonClient::connect()
{
    auto connectError = [this](const char *message) {
        if (!isQuiet()) {
            LOG_ERR("%s " RED("connect error: ") RED_BOLD("\"%s\""), tag(), message);
        }

        retry();
    };

    setState(ConnectingState);

    if (!m_coin.isValid() && !m_pool.algorithm().isValid()) {
        return connectError("Invalid algorithm.");
    }

    if (!m_pool.algorithm().isValid()) {
        m_pool.setAlgo(m_coin.algorithm());
    }

    const xmrig::Algorithm algo = m_pool.algorithm();
    if ((algo == Algorithm::ASTROBWT_DERO) || (algo == Algorithm::ASTROBWT_DERO_2) || (m_coin == Coin::DERO) || (m_coin == Coin::DERO_HE)) {
        m_apiVersion = API_DERO;
    }

    if ((m_apiVersion == API_MONERO) && !m_walletAddress.isValid()) {
        return connectError("Invalid wallet address.");
    }

    if ((m_pool.zmq_port() >= 0) || m_pool.isWSS()) {
        m_dns = Dns::resolve(m_pool.host(), this);
    }
    else {
        getBlockTemplate();
    }
}


void xmrig::DaemonClient::connect(const Pool &pool)
{
    setPool(pool);
    connect();
}


void xmrig::DaemonClient::setPool(const Pool &pool)
{
    BaseClient::setPool(pool);

    m_walletAddress.decode(m_user);

    m_coin = pool.coin().isValid() ?  pool.coin() : m_walletAddress.coin();

    if (!m_coin.isValid() && pool.algorithm() == Algorithm::RX_WOW) {
        m_coin = Coin::WOWNERO;
    }
}


void xmrig::DaemonClient::onHttpData(const HttpData &data)
{
    if (data.status != 200) {
        return retry();
    }

    m_ip = data.ip().c_str();

#   ifdef XMRIG_FEATURE_TLS
    m_tlsVersion     = data.tlsVersion();
    m_tlsFingerprint = data.tlsFingerprint();
#   endif

    rapidjson::Document doc;
    if (doc.Parse(data.body.c_str()).HasParseError()) {
        if (!isQuiet()) {
            LOG_ERR("%s " RED("JSON decode failed: ") RED_BOLD("\"%s\""), tag(), rapidjson::GetParseError_En(doc.GetParseError()));
        }

        return retry();
    }

    if (data.method == HTTP_GET) {
        if (data.url == kGetHeight) {
            if (!doc.HasMember(kHash)) {
                m_apiVersion = API_CRYPTONOTE_DEFAULT;

                return send(kGetInfo);
            }

            const uint64_t height = Json::getUint64(doc, kHeight);
            const String hash = Json::getString(doc, kHash);

            if (isOutdated(height, hash)) {
                // Multiple /getheight responses can come at once resulting in multiple getBlockTemplate() calls
                if ((height != m_blocktemplateRequestHeight) || (hash != m_blocktemplateRequestHash)) {
                    m_blocktemplateRequestHeight = height;
                    m_blocktemplateRequestHash = hash;
                    getBlockTemplate();
                }
            }
        }
        else if (data.url == kGetInfo) {
            const uint64_t height = Json::getUint64(doc, kHeight);
            const String hash = Json::getString(doc, "top_block_hash");

            if (isOutdated(height, hash)) {
                // Multiple /getinfo responses can come at once resulting in multiple getBlockTemplate() calls
                if ((height != m_blocktemplateRequestHeight) || (hash != m_blocktemplateRequestHash)) {
                    m_blocktemplateRequestHeight = height;
                    m_blocktemplateRequestHash = hash;
                    getBlockTemplate();
                }
            }
        }

        return;
    }

    if (!parseResponse(Json::getInt64(doc, "id", -1), Json::getObject(doc, "result"), Json::getObject(doc, "error"))) {
        retry();
    }
}


void xmrig::DaemonClient::onTimer(const Timer *)
{
    if (m_state == ConnectingState) {
        connect();
    }
    else if (m_state == ConnectedState) {
        if (m_pool.isWSS()) {
            return;
        }
        if (m_apiVersion == API_DERO) {
            rpcSend(JsonRequest::create(m_sequence, "get_info"));
        }
        else {
            send((m_apiVersion == API_MONERO) ? kGetHeight : kGetInfo);
        }
    }
}


void xmrig::DaemonClient::onResolved(const DnsRecords &records, int status, const char* error)
{
    m_dns.reset();

    if (status < 0 && records.isEmpty()) {
        if (!isQuiet()) {
            LOG_ERR("%s " RED("DNS error: ") RED_BOLD("\"%s\""), tag(), error);
        }

        retry();
        return;
    }


    const auto &record = records.get();
    m_ip = record.ip();

    auto req = new uv_connect_t;
    req->data = m_storage.ptr(m_key);

    uv_tcp_t* s = new uv_tcp_t;
    s->data = m_storage.ptr(m_key);

    uv_tcp_init(uv_default_loop(), s);
    uv_tcp_nodelay(s, 1);

#   ifndef WIN32
    uv_tcp_keepalive(s, 1, 60);
#   endif

#   ifdef XMRIG_FEATURE_TLS
    if (m_pool.isWSS()) {
        delete m_wss.m_socket;
        m_wss.m_socket = s;
        uv_tcp_connect(req, s, record.addr(m_pool.port()), onWSSConnect);
    }
    else
#   endif
    if (m_pool.zmq_port() > 0) {
        delete m_ZMQSocket;
        m_ZMQSocket = s;
        uv_tcp_connect(req, s, record.addr(m_pool.zmq_port()), onZMQConnect);
    }
}


bool xmrig::DaemonClient::isOutdated(uint64_t height, const char *hash) const
{
    return m_job.height() != height || m_prevHash != hash;
}


bool xmrig::DaemonClient::parseJob(const rapidjson::Value &params, int *code)
{
    auto jobError = [this, code](const char *message) {
        if (!isQuiet()) {
            LOG_ERR("%s " RED("job error: ") RED_BOLD("\"%s\""), tag(), message);
        }

        *code = 1;

        return false;
    };

    Job job(false, m_pool.algorithm(), String());

    String blocktemplate = Json::getString(params, kBlocktemplateBlob);

    if (blocktemplate.isNull()) {
        return jobError("Empty block template received from daemon."); // FIXME
    }

    if (!m_blocktemplate.parse(blocktemplate, m_coin)) {
        return jobError("Invalid block template received from daemon.");
    }

#   ifdef XMRIG_PROXY_PROJECT
    const size_t k = m_blocktemplate.offset(BlockTemplate::MINER_TX_PREFIX_OFFSET);
    job.setMinerTx(
        m_blocktemplate.blob() + k,
        m_blocktemplate.blob() + m_blocktemplate.offset(BlockTemplate::MINER_TX_PREFIX_END_OFFSET),
        m_blocktemplate.offset(BlockTemplate::EPH_PUBLIC_KEY_OFFSET) - k,
        m_blocktemplate.offset(BlockTemplate::TX_PUBKEY_OFFSET) - k,
        m_blocktemplate.offset(BlockTemplate::TX_EXTRA_NONCE_OFFSET) - k,
        m_blocktemplate.txExtraNonce().size(),
        m_blocktemplate.minerTxMerkleTreeBranch()
    );
#   endif

    m_blockhashingblob = Json::getString(params, kBlockhashingBlob);

    if (m_blocktemplate.hasMinerSignature()) {
        if (m_pool.spendSecretKey().isEmpty()) {
            return jobError("Secret spend key is not set.");
        }

        if (m_pool.spendSecretKey().size() != 64) {
            return jobError("Secret spend key has invalid length. It must be 64 hex characters.");
        }

        uint8_t secret_spendkey[32];
        if (!Cvt::fromHex(secret_spendkey, 32, m_pool.spendSecretKey(), 64)) {
            return jobError("Secret spend key is not a valid hex data.");
        }

        uint8_t public_spendkey[32];
        if (!secret_key_to_public_key(secret_spendkey, public_spendkey)) {
            return jobError("Secret spend key is invalid.");
        }

#       ifdef XMRIG_PROXY_PROJECT
        job.setSpendSecretKey(secret_spendkey);
#       else
        uint8_t secret_viewkey[32];
        derive_view_secret_key(secret_spendkey, secret_viewkey);

        uint8_t public_viewkey[32];
        if (!secret_key_to_public_key(secret_viewkey, public_viewkey)) {
            return jobError("Secret view key is invalid.");
        }

        uint8_t derivation[32];
        if (!generate_key_derivation(m_blocktemplate.blob(BlockTemplate::TX_PUBKEY_OFFSET), secret_viewkey, derivation)) {
            return jobError("Failed to generate key derivation for miner signature.");
        }

        if (!m_walletAddress.decode(m_pool.user())) {
            return jobError("Invalid wallet address.");
        }

        if (memcmp(m_walletAddress.spendKey(), public_spendkey, sizeof(public_spendkey)) != 0) {
            return jobError("Wallet address and spend key don't match.");
        }

        if (memcmp(m_walletAddress.viewKey(), public_viewkey, sizeof(public_viewkey)) != 0) {
            return jobError("Wallet address and view key don't match.");
        }

        uint8_t eph_secret_key[32];
        derive_secret_key(derivation, 0, secret_spendkey, eph_secret_key);

        job.setEphemeralKeys(m_blocktemplate.blob(BlockTemplate::EPH_PUBLIC_KEY_OFFSET), eph_secret_key);
#       endif
    }

    if (m_apiVersion == API_DERO) {
        const uint64_t offset = Json::getUint64(params, "reserved_offset");
        Cvt::toHex(m_blockhashingblob.data() + offset * 2, kBlobReserveSize * 2, Cvt::randomBytes(kBlobReserveSize).data(), kBlobReserveSize);
    }

    if (m_coin.isValid()) {
        job.setAlgorithm(m_coin.algorithm(m_blocktemplate.majorVersion()));
    }

    if (!job.setBlob(m_blockhashingblob)) {
        *code = 3;
        return false;
    }

    job.setSeedHash(Json::getString(params, "seed_hash"));
    job.setHeight(Json::getUint64(params, kHeight));
    job.setDiff(Json::getUint64(params, "difficulty"));

    m_currentJobId = Cvt::toHex(Cvt::randomBytes(4));
    job.setId(m_currentJobId);

    m_job              = std::move(job);
    m_blocktemplateStr = std::move(blocktemplate);
    m_prevHash         = Json::getString(params, "prev_hash");

    if (m_apiVersion == API_DERO) {
        // Truncate to 32 bytes to have the same data as in get_info RPC
        if (m_prevHash.size() > 64) {
            m_prevHash.data()[64] = '\0';
        }
    }

    if (m_state == ConnectingState) {
        setState(ConnectedState);
    }

    m_listener->onJobReceived(this, m_job, params);
    return true;
}


bool xmrig::DaemonClient::parseResponse(int64_t id, const rapidjson::Value &result, const rapidjson::Value &error)
{
    if (id == -1) {
        return false;
    }

    if (error.IsObject()) {
        const char *message = error["message"].GetString();

        if (!handleSubmitResponse(id, message) && !isQuiet()) {
            LOG_ERR("[%s:%d] error: " RED_BOLD("\"%s\"") RED_S ", code: %d", m_pool.host().data(), m_pool.port(), message, error["code"].GetInt());
        }

        return false;
    }

    if (!result.IsObject()) {
        return false;
    }

    if (result.HasMember("top_block_hash")) {
        if (m_prevHash != Json::getString(result, "top_block_hash")) {
            getBlockTemplate();
        }
        return true;
    }

    int code = -1;
    if (result.HasMember(kBlocktemplateBlob) && parseJob(result, &code)) {
        return true;
    }

    const char* error_msg = nullptr;

    if ((m_apiVersion == API_DERO) && result.HasMember("status")) {
        error_msg = result["status"].GetString();
        if (!error_msg || (strlen(error_msg) == 0) || (strcmp(error_msg, "OK") == 0)) {
            error_msg = nullptr;
        }
    }

    if (handleSubmitResponse(id, error_msg)) {
        if (error_msg || (m_pool.zmq_port() < 0)) {
            getBlockTemplate();
        }
        return true;
    }


    return false;
}


int64_t xmrig::DaemonClient::getBlockTemplate()
{
    using namespace rapidjson;
    Document doc(kObjectType);
    auto &allocator = doc.GetAllocator();

    Value params(kObjectType);
    params.AddMember("wallet_address", m_user.toJSON(), allocator);
    if (m_apiVersion == API_DERO) {
        params.AddMember("reserve_size", static_cast<uint64_t>(kBlobReserveSize), allocator);
    }
    else {
        params.AddMember("extra_nonce", Cvt::toHex(Cvt::randomBytes(kBlobReserveSize)).toJSON(doc), allocator);
    }

    JsonRequest::create(doc, m_sequence, "getblocktemplate", params);

    return rpcSend(doc);
}


int64_t xmrig::DaemonClient::rpcSend(const rapidjson::Document &doc)
{
    FetchRequest req(HTTP_POST, m_pool.host(), m_pool.port(), kJsonRPC, doc, m_pool.isTLS(), isQuiet());
    fetch(tag(), std::move(req), m_httpListener);

    return m_sequence++;
}


void xmrig::DaemonClient::retry()
{
    m_failures++;
    m_listener->onClose(this, static_cast<int>(m_failures));

    if (m_failures == -1) {
        return;
    }

    if (m_state == ConnectedState) {
        setState(ConnectingState);
    }

#   ifdef XMRIG_FEATURE_TLS
    if (m_wss.m_socket) {
        uv_close(reinterpret_cast<uv_handle_t*>(m_wss.m_socket), onWSSClose);
    }
    else
#   endif
    if ((m_ZMQConnectionState != ZMQ_NOT_CONNECTED) && (m_ZMQConnectionState != ZMQ_DISCONNECTING)) {
        uv_close(reinterpret_cast<uv_handle_t*>(m_ZMQSocket), onZMQClose);
    }

    m_timer->stop();
    m_timer->start(m_retryPause, 0);
}


void xmrig::DaemonClient::send(const char *path)
{
    FetchRequest req(HTTP_GET, m_pool.host(), m_pool.port(), path, m_pool.isTLS(), isQuiet());
    fetch(tag(), std::move(req), m_httpListener);
}


void xmrig::DaemonClient::setState(SocketState state)
{
    if (m_state == state) {
        return;
    }

    m_state = state;

    switch (state) {
    case ConnectedState:
        {
            m_failures = 0;
            m_listener->onLoginSuccess(this);

            if (m_pool.zmq_port() < 0) {
                const uint64_t interval = std::max<uint64_t>(20, m_pool.pollInterval());
                m_timer->start(interval, interval);
            }
        }
        break;

    case UnconnectedState:
        m_failures = -1;
        m_timer->stop();
        break;

    default:
        break;
    }
}


void xmrig::DaemonClient::onZMQConnect(uv_connect_t* req, int status)
{
    DaemonClient* client = getClient(req->data);
    delete req;

    if (!client) {
        return;
    }

    if (status < 0) {
        LOG_ERR("%s " RED("ZMQ connect error: ") RED_BOLD("\"%s\""), client->tag(), uv_strerror(status));
        client->retry();
        return;
    }

    client->ZMQConnected();
}


void xmrig::DaemonClient::onZMQRead(uv_stream_t* stream, ssize_t nread, const uv_buf_t* buf)
{
    DaemonClient* client = getClient(stream->data);
    if (client) {
        client->ZMQRead(nread, buf);
    }

    NetBuffer::release(buf);
}


void xmrig::DaemonClient::onZMQClose(uv_handle_t* handle)
{
    DaemonClient* client = getClient(handle->data);
    if (client) {
#       ifdef APP_DEBUG
        LOG_DEBUG(CYAN("tcp-zmq://%s:%u") BLACK_BOLD(" disconnected"), client->m_pool.host().data(), client->m_pool.zmq_port());
#       endif
        client->m_ZMQConnectionState = ZMQ_NOT_CONNECTED;
    }
}


void xmrig::DaemonClient::onZMQShutdown(uv_handle_t* handle)
{
    DaemonClient* client = getClient(handle->data);
    if (client) {
#       ifdef APP_DEBUG
        LOG_DEBUG(CYAN("tcp-zmq://%s:%u") BLACK_BOLD(" shutdown"), client->m_pool.host().data(), client->m_pool.zmq_port());
#       endif
        client->m_ZMQConnectionState = ZMQ_NOT_CONNECTED;
        m_storage.remove(client->m_key);
    }
}


void xmrig::DaemonClient::ZMQConnected()
{
#   ifdef APP_DEBUG
    LOG_DEBUG(CYAN("tcp-zmq://%s:%u") BLACK_BOLD(" connected"), m_pool.host().data(), m_pool.zmq_port());
#   endif

    m_ZMQConnectionState = ZMQ_GREETING_1;
    m_ZMQSendBuf.reserve(256);
    m_ZMQRecvBuf.reserve(256);

    if (ZMQWrite(kZMQGreeting, kZMQGreetingSize1)) {
        uv_read_start(reinterpret_cast<uv_stream_t*>(m_ZMQSocket), NetBuffer::onAlloc, onZMQRead);
    }
}


bool xmrig::DaemonClient::ZMQWrite(const char* data, size_t size)
{
    m_ZMQSendBuf.assign(data, data + size);

    uv_buf_t buf;
    buf.base = m_ZMQSendBuf.data();
    buf.len = static_cast<uint32_t>(m_ZMQSendBuf.size());

    const int rc = uv_try_write(reinterpret_cast<uv_stream_t*>(m_ZMQSocket), &buf, 1);

    if (static_cast<size_t>(rc) == buf.len) {
        return true;
    }

    LOG_ERR("%s " RED("ZMQ write failed, rc = %d"), tag(), rc);
    ZMQClose();
    return false;
}


void xmrig::DaemonClient::ZMQRead(ssize_t nread, const uv_buf_t* buf)
{
    if (nread <= 0) {
        LOG_ERR("%s " RED("ZMQ read failed, nread = %" PRId64), tag(), nread);
        ZMQClose();
        return;
    }

    m_ZMQRecvBuf.insert(m_ZMQRecvBuf.end(), buf->base, buf->base + nread);

    do {
        switch (m_ZMQConnectionState) {
        case ZMQ_GREETING_1:
            if (m_ZMQRecvBuf.size() >= kZMQGreetingSize1) {
                if ((m_ZMQRecvBuf[0] == static_cast<char>(-1)) && (m_ZMQRecvBuf[9] == 127) && (m_ZMQRecvBuf[10] == 3)) {
                    ZMQWrite(kZMQGreeting + kZMQGreetingSize1, sizeof(kZMQGreeting) - kZMQGreetingSize1);
                    m_ZMQConnectionState = ZMQ_GREETING_2;
                    break;
                }

                LOG_ERR("%s " RED("ZMQ handshake failed: invalid greeting format"), tag());
                ZMQClose();
            }
            return;

        case ZMQ_GREETING_2:
            if (m_ZMQRecvBuf.size() >= sizeof(kZMQGreeting)) {
                if (memcmp(m_ZMQRecvBuf.data() + 12, kZMQGreeting + 12, 20) == 0) {
                    m_ZMQConnectionState = ZMQ_HANDSHAKE;
                    m_ZMQRecvBuf.erase(m_ZMQRecvBuf.begin(), m_ZMQRecvBuf.begin() + sizeof(kZMQGreeting));

                    ZMQWrite(kZMQHandshake, sizeof(kZMQHandshake) - 1);
                    break;
                }

                LOG_ERR("%s " RED("ZMQ handshake failed: invalid greeting format 2"), tag());
                ZMQClose();

            }
            return;

        case ZMQ_HANDSHAKE:
            if (m_ZMQRecvBuf.size() >= 2) {
                if (m_ZMQRecvBuf[0] != 4) {
                    LOG_ERR("%s " RED("ZMQ handshake failed: invalid handshake format"), tag());
                    ZMQClose();
                    return;
                }

                const size_t size = static_cast<unsigned char>(m_ZMQRecvBuf[1]);
                if (size < 18) {
                    LOG_ERR("%s " RED("ZMQ handshake failed: invalid handshake size"), tag());
                    ZMQClose();
                    return;
                }

                if (m_ZMQRecvBuf.size() < size + 2) {
                    return;
                }

                if (memcmp(m_ZMQRecvBuf.data() + 2, kZMQHandshake + 2, 18) != 0) {
                    LOG_ERR("%s " RED("ZMQ handshake failed: invalid handshake data"), tag());
                    ZMQClose();
                    return;
                }

                ZMQWrite(kZMQSubscribe, sizeof(kZMQSubscribe) - 1);

                m_ZMQConnectionState = ZMQ_CONNECTED;
                m_ZMQRecvBuf.erase(m_ZMQRecvBuf.begin(), m_ZMQRecvBuf.begin() + size + 2);

                getBlockTemplate();
                break;
            }
            return;

        case ZMQ_CONNECTED:
            ZMQParse();
            return;

        default:
            return;
        }
    } while (true);
}


void xmrig::DaemonClient::ZMQParse()
{
#   ifdef APP_DEBUG
    std::vector<char> msg;
#   endif

    size_t msg_size = 0;

    char *data   = m_ZMQRecvBuf.data();
    size_t avail = m_ZMQRecvBuf.size();
    bool more    = false;

    do {
        if (avail < 1) {
            return;
        }

        more                 = (data[0] & 1) != 0;
        const bool long_size = (data[0] & 2) != 0;
        const bool command   = (data[0] & 4) != 0;

        ++data;
        --avail;

        uint64_t size = 0;
        if (long_size)
        {
            if (avail < sizeof(uint64_t)) {
                return;
            }
            size = bswap_64(*((uint64_t*)data));
            data += sizeof(uint64_t);
            avail -= sizeof(uint64_t);
        }
        else
        {
            if (avail < sizeof(uint8_t)) {
                return;
            }
            size = static_cast<uint8_t>(*data);
            ++data;
            --avail;
        }

        if (size > 1024U - msg_size)
        {
            LOG_ERR("%s " RED("ZMQ message is too large, size = %" PRIu64 " bytes"), tag(), size);
            ZMQClose();
            return;
        }

        if (avail < size) {
            return;
        }

        if (!command) {
#           ifdef APP_DEBUG
            msg.insert(msg.end(), data, data + size);
#           endif

            msg_size += size;
        }

        data += size;
        avail -= size;
    } while (more);

    m_ZMQRecvBuf.erase(m_ZMQRecvBuf.begin(), m_ZMQRecvBuf.begin() + (data - m_ZMQRecvBuf.data()));

#   ifdef APP_DEBUG
    LOG_DEBUG(CYAN("tcp-zmq://%s:%u") BLACK_BOLD(" read ") CYAN_BOLD("%zu") BLACK_BOLD(" bytes") " %s", m_pool.host().data(), m_pool.zmq_port(), msg.size(), msg.data());
#   endif

    getBlockTemplate();
}


bool xmrig::DaemonClient::ZMQClose(bool shutdown)
{
    if ((m_ZMQConnectionState == ZMQ_NOT_CONNECTED) || (m_ZMQConnectionState == ZMQ_DISCONNECTING)) {
        if (shutdown) {
            m_storage.remove(m_key);
        }
        return false;
    }

    m_ZMQConnectionState = ZMQ_DISCONNECTING;

    if (uv_is_closing(reinterpret_cast<uv_handle_t*>(m_ZMQSocket)) == 0) {
        uv_close(reinterpret_cast<uv_handle_t*>(m_ZMQSocket), shutdown ? onZMQShutdown : onZMQClose);
        if (!shutdown) {
            retry();
        }
        return true;
    }

    return false;
}


#ifdef XMRIG_FEATURE_TLS
void xmrig::DaemonClient::onWSSConnect(uv_connect_t* req, int status)
{
    DaemonClient* client = getClient(req->data);
    delete req;

    if (!client) {
        return;
    }

    if (status < 0) {
        LOG_ERR("%s " RED("WSS connect error: ") RED_BOLD("\"%s\""), client->tag(), uv_strerror(status));
        client->retry();
        return;
    }

    client->WSSConnected();
}


void xmrig::DaemonClient::onWSSRead(uv_stream_t* stream, ssize_t nread, const uv_buf_t* buf)
{
    DaemonClient* client = getClient(stream->data);
    if (client) {
        client->WSSRead(nread, buf);
    }

    NetBuffer::release(buf);
}


void xmrig::DaemonClient::onWSSClose(uv_handle_t* handle)
{
    DaemonClient* client = getClient(handle->data);
    if (client) {
#       ifdef APP_DEBUG
        LOG_DEBUG(CYAN("%s") BLACK_BOLD(" disconnected"), client->m_pool.url().data());
#       endif
        client->m_wss.cleanup();
        client->retry();
    }
}


void xmrig::DaemonClient::onWSSShutdown(uv_handle_t* handle)
{
    DaemonClient* client = getClient(handle->data);
    if (client) {
#       ifdef APP_DEBUG
        LOG_DEBUG(CYAN("%s") BLACK_BOLD(" shutdown"), client->m_pool.url().data());
#       endif
        client->m_wss.cleanup();
        m_storage.remove(client->m_key);
    }
}


void xmrig::DaemonClient::WSSConnected()
{
    m_wss.m_ctx   = SSL_CTX_new(SSLv23_method());
    m_wss.m_write = BIO_new(BIO_s_mem());
    m_wss.m_read  = BIO_new(BIO_s_mem());

    SSL_CTX_set_options(m_wss.m_ctx, SSL_OP_NO_SSLv2 | SSL_OP_NO_SSLv3);
    m_wss.m_ssl = SSL_new(m_wss.m_ctx);
    SSL_set_connect_state(m_wss.m_ssl);
    SSL_set_bio(m_wss.m_ssl, m_wss.m_read, m_wss.m_write);
    SSL_do_handshake(m_wss.m_ssl);

    if (WSSWrite(nullptr, 0)) {
        uv_read_start(reinterpret_cast<uv_stream_t*>(m_wss.m_socket), NetBuffer::onAlloc, onWSSRead);
    }
}


bool xmrig::DaemonClient::WSSWrite(const char* data, size_t size)
{
    if (!m_wss.m_socket) {
        return false;
    }

    if (data && size) {
#   ifdef APP_DEBUG
        LOG_DEBUG(CYAN("%s") BLACK_BOLD(" write ") CYAN_BOLD("%zu") BLACK_BOLD(" bytes") " %s", m_pool.url().data(), size, data);
#   endif

        if (!m_wss.m_handshake) {
            WSS::Header h{};
            h.fin = 1;
            h.mask = 1;
            h.opcode = 1;

            uint8_t size_buf[8];
            if (size < 126) {
                h.payload_len = static_cast<uint8_t>(size);
            }
            else if (size < 65536) {
                h.payload_len = 126;
                size_buf[0] = static_cast<uint8_t>(size >> 8);
                size_buf[1] = static_cast<uint8_t>(size & 0xFF);
            }
            else {
                h.payload_len = 127;
                uint64_t k = size;
                for (int i = 7; i >= 0; --i, k >>= 8) {
                    size_buf[i] = static_cast<uint8_t>(k & 0xFF);
                }
            }

            // Header
            SSL_write(m_wss.m_ssl, &h, sizeof(h));

            // Optional extended payload length
            if (h.payload_len == 126) SSL_write(m_wss.m_ssl, size_buf, 2);
            if (h.payload_len == 127) SSL_write(m_wss.m_ssl, size_buf, 8);

            // Masking-key
            SSL_write(m_wss.m_ssl, "\0\0\0\0", 4);
        }

        SSL_write(m_wss.m_ssl, data, static_cast<int>(size));
    }

    uv_buf_t buf;
    buf.len = BIO_get_mem_data(m_wss.m_write, &buf.base);

    if (buf.len == 0) {
        return true;
    }

    const int rc = uv_try_write(reinterpret_cast<uv_stream_t*>(m_wss.m_socket), &buf, 1);

    BIO_reset(m_wss.m_write);

    if (static_cast<size_t>(rc) == buf.len) {
        return true;
    }

    LOG_ERR("%s " RED("WSS write failed, rc = %d"), tag(), rc);
    WSSClose();
    return false;
}


void xmrig::DaemonClient::WSSRead(ssize_t nread, const uv_buf_t* read_buf)
{
    if (nread <= 0) {
        LOG_ERR("%s " RED("WSS read failed, nread = %" PRId64), tag(), nread);
        WSSClose();
        return;
    }

    BIO_write(m_wss.m_read, read_buf->base, static_cast<int>(nread));

    if (!SSL_is_init_finished(m_wss.m_ssl)) {
        const int rc = SSL_connect(m_wss.m_ssl);

        if ((rc < 0) && (SSL_get_error(m_wss.m_ssl, rc) == SSL_ERROR_WANT_READ)) {
            WSSWrite(nullptr, 0);
        }
        else if (rc == 1) {
            // login
            static constexpr char Base64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

            char key[25];
            std::random_device r;

            for (int i = 0; i < 21; ++i) {
                key[i] = Base64[r() % 64];
            }

            key[21] = Base64[0];
            key[22] = '=';
            key[23] = '=';
            key[24] = '\0';

            const int n = snprintf(m_wss.m_buf, sizeof(m_wss.m_buf), kWSSLogin, m_pool.user().data(), m_pool.host().data(), key);
            if (0 <= n && n < static_cast<int>(sizeof(m_wss.m_buf))) {
                WSSWrite(m_wss.m_buf, n);
            }
            else {
                WSSClose();
            }
        }

        return;
    }

    int n = 0;
    while ((n = SSL_read(m_wss.m_ssl, m_wss.m_buf, sizeof(m_wss.m_buf))) > 0) {
        m_wss.m_data.insert(m_wss.m_data.end(), m_wss.m_buf, m_wss.m_buf + n);

        // Skip the first message (HTTP upgrade response)
        if (m_wss.m_handshake) {
            const size_t len = m_wss.m_data.size();
            if (len >= 4) {
                for (size_t k = 0; k <= len - 4; ++k) {
                    if (memcmp(m_wss.m_data.data() + k, "\r\n\r\n", 4) == 0) {
                        m_wss.m_handshake = false;
                        m_wss.m_data.erase(m_wss.m_data.begin(), m_wss.m_data.begin() + k + 4);
                        break;
                    }
                }
            }
            continue;
        }

        const uint8_t* p0 = reinterpret_cast<uint8_t*>(m_wss.m_data.data());
        const uint8_t* p = p0;
        const uint8_t* e = p0 + m_wss.m_data.size();

        if (e - p < static_cast<int>(sizeof(WSS::Header)))
            continue;

        const WSS::Header* h = reinterpret_cast<const WSS::Header*>(p);
        p += sizeof(WSS::Header);

        uint64_t len = h->payload_len;

        if (len == 126) {
            if (e - p < static_cast<int>(sizeof(uint16_t))) {
                continue;
            }
            len = 0;
            for (size_t i = 0; i < sizeof(uint16_t); ++i, ++p) {
                len = (len << 8) | *p;
            }
        }
        else if (len == 127) {
            if (e - p < static_cast<int>(sizeof(uint64_t))) {
                continue;
            }
            len = 0;
            for (size_t i = 0; i < sizeof(uint64_t); ++i, ++p) {
                len = (len << 8) | *p;
            }
        }

        uint8_t mask_key[4] = {};
        if (h->mask) {
            if (e - p < 4)
                continue;
            memcpy(mask_key, p, 4);
            p += 4;
        }

        if (static_cast<uint64_t>(e - p) < len)
            continue;

        for (uint64_t i = 0; i < len; ++i) {
            m_wss.m_message.push_back(p[i] ^ mask_key[i % 4]);
        }
        p += len;

        m_wss.m_data.erase(m_wss.m_data.begin(), m_wss.m_data.begin() + (p - p0));

        if (h->fin) {
            if (m_wss.m_message.back() == '\n') {
                m_wss.m_message.back() = '\0';
            }
            else {
                m_wss.m_message.push_back('\0');
            }
            WSSParse();
            m_wss.m_message.clear();
        }
    }
}


void xmrig::DaemonClient::WSSParse()
{
#   ifdef APP_DEBUG
    LOG_DEBUG(CYAN("%s") BLACK_BOLD(" read ") CYAN_BOLD("%zu") BLACK_BOLD(" bytes") " %s", m_pool.url().data(), m_wss.m_message.size(), m_wss.m_message.data());
#   endif

    using namespace rapidjson;

    Document doc;
    if (doc.ParseInsitu(m_wss.m_message.data()).HasParseError() || !doc.IsObject()) {
        if (!isQuiet()) {
            LOG_ERR("%s " RED("JSON decode failed: ") RED_BOLD("\"%s\""), tag(), GetParseError_En(doc.GetParseError()));
        }

        return retry();
    }

    if (doc.HasMember(kLastError)) {
        String err = Json::getString(doc, kLastError, "");
        if (!err.isEmpty()) {
            LOG_ERR("%s " RED_BOLD("\"%s\""), tag(), err.data());
            return;
        }
    }

    if (doc.HasMember(kBlockhashingBlob)) {
        Job job(false, m_pool.algorithm(), String());

        m_blockhashingblob = Json::getString(doc, kBlockhashingBlob, "");
        if (m_blockhashingblob.isEmpty()) {
            LOG_ERR("%s " RED_BOLD("blockhashing_blob is empty"), tag());
            return;
        }
        job.setBlob(m_blockhashingblob);
        memset(job.blob() + job.nonceOffset(), 0, job.nonceSize());

        const uint64_t height = Json::getUint64(doc, kHeight);

        job.setHeight(height);
        job.setDiff(Json::getUint64(doc, "difficultyuint64"));
        //job.setDiff(100000);

        m_currentJobId = Json::getString(doc, "jobid");
        job.setId(m_currentJobId);

        m_job = std::move(job);

        if (m_state == ConnectingState) {
            setState(ConnectedState);
        }

        const uint64_t blocks = Json::getUint64(doc, "blocks");
        const uint64_t miniblocks = Json::getUint64(doc, "miniblocks");

        if ((blocks != m_wss.m_blocks) || (miniblocks != m_wss.m_miniblocks) || (height != m_wss.m_height)) {
            LOG_INFO("%s " GREEN_BOLD("%" PRIu64 " blocks, %" PRIu64 " mini blocks"), tag(), blocks, miniblocks);
            m_wss.m_blocks = blocks;
            m_wss.m_miniblocks = miniblocks;
            m_wss.m_height = height;
        }

        m_listener->onJobReceived(this, m_job, doc);
        return;
    }
}


bool xmrig::DaemonClient::WSSClose(bool shutdown)
{
    if (m_wss.m_socket && (uv_is_closing(reinterpret_cast<uv_handle_t*>(m_wss.m_socket)) == 0)) {
        uv_close(reinterpret_cast<uv_handle_t*>(m_wss.m_socket), shutdown ? onWSSShutdown : onWSSClose);
        return true;
    }

    return false;
}


void xmrig::DaemonClient::WSS::cleanup()
{
    delete m_socket;
    m_socket = nullptr;

    if (m_ctx) {
        SSL_CTX_free(m_ctx);
        m_ctx = nullptr;
    }
    if (m_ssl) {
        SSL_free(m_ssl);
        m_ssl = nullptr;
    }

    m_read = nullptr;
    m_write = nullptr;
    m_handshake = true;
    m_blocks = 0;
    m_miniblocks = 0;
    m_height = 0;
    m_data.clear();
    m_message.clear();
}
#endif
