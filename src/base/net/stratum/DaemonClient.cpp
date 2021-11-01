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


#include <algorithm>
#include <cassert>


namespace xmrig {


Storage<DaemonClient> DaemonClient::m_storage;


static const char* kBlocktemplateBlob = "blocktemplate_blob";
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
}


void xmrig::DaemonClient::deleteLater()
{
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

    if (!m_walletAddress.isValid()) {
        return connectError("Invalid wallet address.");
    }

    if (!m_coin.isValid() && !m_pool.algorithm().isValid()) {
        return connectError("Invalid algorithm.");
    }

    if ((m_pool.algorithm() == Algorithm::ASTROBWT_DERO) || (m_coin == Coin::DERO)) {
        m_apiVersion = API_DERO;
    }

    if (m_pool.zmq_port() >= 0) {
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


    delete m_ZMQSocket;


    const auto &record = records.get();
    m_ip = record.ip();

    auto req = new uv_connect_t;
    req->data = m_storage.ptr(m_key);

    m_ZMQSocket = new uv_tcp_t;
    m_ZMQSocket->data = m_storage.ptr(m_key);

    uv_tcp_init(uv_default_loop(), m_ZMQSocket);
    uv_tcp_nodelay(m_ZMQSocket, 1);

#   ifndef WIN32
    uv_tcp_keepalive(m_ZMQSocket, 1, 60);
#   endif

    uv_tcp_connect(req, m_ZMQSocket, record.addr(m_pool.zmq_port()), onZMQConnect);
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

    m_blockhashingblob = Json::getString(params, "blockhashing_blob");

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
