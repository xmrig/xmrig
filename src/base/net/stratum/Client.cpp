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

#include <cassert>
#include <cinttypes>
#include <iterator>
#include <cstdio>
#include <cstring>
#include <utility>


#ifdef XMRIG_FEATURE_TLS
#   include <openssl/ssl.h>
#   include <openssl/err.h>
#   include "base/net/stratum/Tls.h"
#endif


#include "base/io/json/Json.h"
#include "base/io/json/JsonRequest.h"
#include "base/io/log/Log.h"
#include "base/kernel/interfaces/IClientListener.h"
#include "base/net/dns/Dns.h"
#include "base/net/stratum/Client.h"
#include "base/tools/Buffer.h"
#include "base/tools/Chrono.h"
#include "net/JobResult.h"
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"


#ifdef _MSC_VER
#   define strncasecmp(x,y,z) _strnicmp(x,y,z)
#endif


namespace xmrig {

Storage<Client> Client::m_storage;

} /* namespace xmrig */


#ifdef APP_DEBUG
static const char *states[] = {
    "unconnected",
    "host-lookup",
    "connecting",
    "connected",
    "closing",
    "reconnecting"
};
#endif


xmrig::Client::Client(int id, const char *agent, IClientListener *listener) :
    BaseClient(id, listener),
    m_agent(agent)
{
    m_key = m_storage.add(this);
    m_dns = new Dns(this);
}


xmrig::Client::~Client()
{
    delete m_dns;
    delete m_socket;
}


bool xmrig::Client::disconnect()
{
    m_keepAlive = 0;
    m_expire    = 0;
    m_failures  = -1;

    return close();
}


bool xmrig::Client::isTLS() const
{
#   ifdef XMRIG_FEATURE_TLS
    return m_pool.isTLS() && m_tls;
#   else
    return false;
#   endif
}


const char *xmrig::Client::tlsFingerprint() const
{
#   ifdef XMRIG_FEATURE_TLS
    if (isTLS() && m_pool.fingerprint() == nullptr) {
        return m_tls->fingerprint();
    }
#   endif

    return nullptr;
}


const char *xmrig::Client::tlsVersion() const
{
#   ifdef XMRIG_FEATURE_TLS
    if (isTLS()) {
        return m_tls->version();
    }
#   endif

    return nullptr;
}


int64_t xmrig::Client::send(const rapidjson::Value &obj, Callback callback)
{
    assert(obj["id"] == sequence());

    m_callbacks.insert({ sequence(), std::move(callback) });

    return send(obj);
}


int64_t xmrig::Client::send(const rapidjson::Value &obj)
{
    using namespace rapidjson;

    Value value;

    StringBuffer buffer(nullptr, 512);
    Writer<StringBuffer> writer(buffer);
    obj.Accept(writer);

    const size_t size = buffer.GetSize();
    if (size > (sizeof(m_sendBuf) - 2)) {
        LOG_ERR("[%s] send failed: \"send buffer overflow: %zu > %zu\"", url(), size, (sizeof(m_sendBuf) - 2));
        close();
        return -1;
    }

    memcpy(m_sendBuf, buffer.GetString(), size);
    m_sendBuf[size]     = '\n';
    m_sendBuf[size + 1] = '\0';

    return send(size + 1);
}


int64_t xmrig::Client::submit(const JobResult &result)
{
#   ifndef XMRIG_PROXY_PROJECT
    if (result.clientId != m_rpcId) {
        return -1;
    }
#   endif

    using namespace rapidjson;

#   ifdef XMRIG_PROXY_PROJECT
    const char *nonce = result.nonce;
    const char *data  = result.result;
#   else
    char *nonce = m_sendBuf;
    char *data  = m_sendBuf + 16;

    Buffer::toHex(reinterpret_cast<const char*>(&result.nonce), 4, nonce);
    nonce[8] = '\0';

    Buffer::toHex(result.result(), 32, data);
    data[64] = '\0';
#   endif

    Document doc(kObjectType);
    auto &allocator = doc.GetAllocator();

    Value params(kObjectType);
    params.AddMember("id",     StringRef(m_rpcId.data()), allocator);
    params.AddMember("job_id", StringRef(result.jobId.data()), allocator);
    params.AddMember("nonce",  StringRef(nonce), allocator);
    params.AddMember("result", StringRef(data), allocator);

    if (has<EXT_ALGO>() && result.algorithm.isValid()) {
        params.AddMember("algo", StringRef(result.algorithm.shortName()), allocator);
    }

    JsonRequest::create(doc, m_sequence, "submit", params);

#   ifdef XMRIG_PROXY_PROJECT
    m_results[m_sequence] = SubmitResult(m_sequence, result.diff, result.actualDiff(), result.id, 0);
#   else
    m_results[m_sequence] = SubmitResult(m_sequence, result.diff, result.actualDiff(), 0, result.backend);
#   endif

    return send(doc);
}


void xmrig::Client::connect()
{
#   ifdef XMRIG_FEATURE_TLS
    if (m_pool.isTLS()) {
        m_tls = new Tls(this);
    }
#   endif

    resolve(m_pool.host());
}


void xmrig::Client::connect(const Pool &pool)
{
    setPool(pool);
    connect();
}


void xmrig::Client::deleteLater()
{
    if (!m_listener) {
        return;
    }

    m_listener = nullptr;

    if (!disconnect()) {
        m_storage.remove(m_key);
    }
}


void xmrig::Client::tick(uint64_t now)
{
    if (m_state == ConnectedState) {
        if (m_expire && now > m_expire) {
            LOG_DEBUG_ERR("[%s] timeout", url());
            close();
        }
        else if (m_keepAlive && now > m_keepAlive) {
            ping();
        }

        return;
    }

    if (m_state == ReconnectingState && m_expire && now > m_expire) {
        return connect();
    }

    if (m_state == ConnectingState && m_expire && now > m_expire) {
        return reconnect();
    }
}


void xmrig::Client::onResolved(const Dns &dns, int status)
{
    assert(m_listener != nullptr);
    if (!m_listener) {
        return reconnect();
    }

    if (status < 0 && dns.isEmpty()) {
        if (!isQuiet()) {
            LOG_ERR("[%s] DNS error: \"%s\"", url(), uv_strerror(status));
        }

        return reconnect();
    }

    const DnsRecord &record = dns.get();
    m_ip = record.ip();

    connect(record.addr(m_pool.port()));
}


bool xmrig::Client::close()
{
    if (m_state == ClosingState) {
        return m_socket != nullptr;
    }

    if (m_state == UnconnectedState || m_socket == nullptr) {
        return false;
    }

    setState(ClosingState);

    if (uv_is_closing(reinterpret_cast<uv_handle_t*>(m_socket)) == 0) {
        uv_close(reinterpret_cast<uv_handle_t*>(m_socket), Client::onClose);
    }

    return true;
}


bool xmrig::Client::isCriticalError(const char *message)
{
    if (!message) {
        return false;
    }

    if (strncasecmp(message, "Unauthenticated", 15) == 0) {
        return true;
    }

    if (strncasecmp(message, "your IP is banned", 17) == 0) {
        return true;
    }

    if (strncasecmp(message, "IP Address currently banned", 27) == 0) {
        return true;
    }

    if (strncasecmp(message, "Invalid job id", 14) == 0) {
        return true;
    }

    return false;
}


bool xmrig::Client::parseJob(const rapidjson::Value &params, int *code)
{
    if (!params.IsObject()) {
        *code = 2;
        return false;
    }

    Job job(has<EXT_NICEHASH>(), m_pool.algorithm(), m_rpcId);

    if (!job.setId(params["job_id"].GetString())) {
        *code = 3;
        return false;
    }

#   ifdef XMRIG_FEATURE_HTTP
    if (m_pool.mode() == Pool::MODE_SELF_SELECT) {
        job.setExtraNonce(Json::getString(params, "extra_nonce"));
        job.setPoolWallet(Json::getString(params, "pool_wallet"));

        if (job.extraNonce().isNull() || job.poolWallet().isNull()) {
            *code = 4;
            return false;
        }
    }
    else
#   endif
    {
        if (!job.setBlob(params["blob"].GetString())) {
            *code = 4;
            return false;
        }
    }

    if (!job.setTarget(params["target"].GetString())) {
        *code = 5;
        return false;
    }

    const char *algo = Json::getString(params, "algo");
    if (algo) {
        job.setAlgorithm(algo);
    }
    else if (m_pool.coin().isValid()) {
        job.setAlgorithm(m_pool.coin().algorithm(job.blob()[0]));
    }

    job.setHeight(Json::getUint64(params, "height"));

    if (!verifyAlgorithm(job.algorithm(), algo)) {
        *code = 6;
        return false;
    }

    if (m_pool.mode() != Pool::MODE_SELF_SELECT && job.algorithm().family() == Algorithm::RANDOM_X && !job.setSeedHash(Json::getString(params, "seed_hash"))) {
        if (!isQuiet()) {
            LOG_ERR("[%s] failed to parse field \"seed_hash\" required by RandomX", url(), algo);
        }

        *code = 7;
        return false;
    }

    m_job.setClientId(m_rpcId);

    if (m_job != job) {
        m_jobs++;
        m_job = std::move(job);
        return true;
    }

    if (m_jobs == 0) { // https://github.com/xmrig/xmrig/issues/459
        return false;
    }

    if (!isQuiet()) {
        LOG_WARN("[%s] duplicate job received, reconnect", url());
    }

    close();
    return false;
}


bool xmrig::Client::parseLogin(const rapidjson::Value &result, int *code)
{
    m_rpcId = result["id"].GetString();
    if (m_rpcId.isNull()) {
        *code = 1;
        return false;
    }

    parseExtensions(result);

    const bool rc = parseJob(result["job"], code);
    m_jobs = 0;

    return rc;
}


bool xmrig::Client::send(BIO *bio)
{
#   ifdef XMRIG_FEATURE_TLS
    uv_buf_t buf;
    buf.len = BIO_get_mem_data(bio, &buf.base);

    if (buf.len == 0) {
        return true;
    }

    LOG_DEBUG("[%s] TLS send     (%d bytes)", url(), static_cast<int>(buf.len));

    bool result = false;
    if (state() == ConnectedState && uv_is_writable(m_stream)) {
        result = uv_try_write(m_stream, &buf, 1) > 0;

        if (!result) {
            close();
        }
    }
    else {
        LOG_DEBUG_ERR("[%s] send failed, invalid state: %d", url(), m_state);
    }

    (void) BIO_reset(bio);

    return result;
#   else
    return false;
#   endif
}


bool xmrig::Client::verifyAlgorithm(const Algorithm &algorithm, const char *algo) const
{
    if (!algorithm.isValid()) {
        if (!isQuiet()) {
            if (algo == nullptr) {
                LOG_ERR("[%s] unknown algorithm, make sure you set \"algo\" or \"coin\" option", url(), algo);
            }
            else {
                LOG_ERR("[%s] unsupported algorithm \"%s\" detected, reconnect", url(), algo);
            }
        }

        return false;
    }

    bool ok = true;
    m_listener->onVerifyAlgorithm(this, algorithm, &ok);

    if (!ok && !isQuiet()) {
        LOG_ERR("[%s] incompatible/disabled algorithm \"%s\" detected, reconnect", url(), algorithm.shortName());
    }

    return ok;
}


int xmrig::Client::resolve(const String &host)
{
    setState(HostLookupState);

    m_recvBuf.reset();

    if (m_failures == -1) {
        m_failures = 0;
    }

    if (!m_dns->resolve(host)) {
        if (!isQuiet()) {
            LOG_ERR("[%s:%u] getaddrinfo error: \"%s\"", host.data(), m_pool.port(), uv_strerror(m_dns->status()));
        }

        return 1;
    }

    return 0;
}


int64_t xmrig::Client::send(size_t size)
{
    LOG_DEBUG("[%s] send (%d bytes): \"%.*s\"", url(), size, static_cast<int>(size) - 1, m_sendBuf);

#   ifdef XMRIG_FEATURE_TLS
    if (isTLS()) {
        if (!m_tls->send(m_sendBuf, size)) {
            return -1;
        }
    }
    else
#   endif
    {
        if (state() != ConnectedState || !uv_is_writable(m_stream)) {
            LOG_DEBUG_ERR("[%s] send failed, invalid state: %d", url(), m_state);
            return -1;
        }

        uv_buf_t buf = uv_buf_init(m_sendBuf, (unsigned int) size);

        if (uv_try_write(m_stream, &buf, 1) < 0) {
            close();
            return -1;
        }
    }

    m_expire = Chrono::steadyMSecs() + kResponseTimeout;
    return m_sequence++;
}


void xmrig::Client::connect(sockaddr *addr)
{
    setState(ConnectingState);

    auto req = new uv_connect_t;
    req->data = m_storage.ptr(m_key);

    m_socket = new uv_tcp_t;
    m_socket->data = m_storage.ptr(m_key);

    uv_tcp_init(uv_default_loop(), m_socket);
    uv_tcp_nodelay(m_socket, 1);

#   ifndef WIN32
    uv_tcp_keepalive(m_socket, 1, 60);
#   endif

    uv_tcp_connect(req, m_socket, addr, onConnect);

    delete addr;
}


void xmrig::Client::handshake()
{
#   ifdef XMRIG_FEATURE_TLS
    if (isTLS()) {
        m_expire = Chrono::steadyMSecs() + kResponseTimeout;

        m_tls->handshake();
    }
    else
#   endif
    {
        login();
    }
}


void xmrig::Client::login()
{
    using namespace rapidjson;
    m_results.clear();

    Document doc(kObjectType);
    auto &allocator = doc.GetAllocator();

    Value params(kObjectType);
    params.AddMember("login", m_pool.user().toJSON(),     allocator);
    params.AddMember("pass",  m_pool.password().toJSON(), allocator);
    params.AddMember("agent", StringRef(m_agent),         allocator);

    if (!m_pool.rigId().isNull()) {
        params.AddMember("rigid", m_pool.rigId().toJSON(), allocator);
    }

    m_listener->onLogin(this, doc, params);

    JsonRequest::create(doc, 1, "login", params);

    send(doc);
}


void xmrig::Client::onClose()
{
    delete m_socket;

    m_stream = nullptr;
    m_socket = nullptr;
    setState(UnconnectedState);

#   ifdef XMRIG_FEATURE_TLS
    if (m_tls) {
        delete m_tls;
        m_tls = nullptr;
    }
#   endif

    reconnect();
}


void xmrig::Client::parse(char *line, size_t len)
{
    startTimeout();

    LOG_DEBUG("[%s] received (%d bytes): \"%.*s\"", url(), len, static_cast<int>(len), line);

    if (len < 32 || line[0] != '{') {
        if (!isQuiet()) {
            LOG_ERR("[%s] JSON decode failed", url());
        }

        return;
    }

    rapidjson::Document doc;
    if (doc.ParseInsitu(line).HasParseError()) {
        if (!isQuiet()) {
            LOG_ERR("[%s] JSON decode failed: \"%s\"", url(), rapidjson::GetParseError_En(doc.GetParseError()));
        }

        return;
    }

    if (!doc.IsObject()) {
        return;
    }

    const rapidjson::Value &id = doc["id"];
    if (id.IsInt64()) {
        parseResponse(id.GetInt64(), doc["result"], doc["error"]);
    }
    else {
        parseNotification(doc["method"].GetString(), doc["params"], doc["error"]);
    }
}


void xmrig::Client::parseExtensions(const rapidjson::Value &result)
{
    m_extensions.reset();

    if (!result.HasMember("extensions")) {
        return;
    }

    const rapidjson::Value &extensions = result["extensions"];
    if (!extensions.IsArray()) {
        return;
    }

    for (const rapidjson::Value &ext : extensions.GetArray()) {
        if (!ext.IsString()) {
            continue;
        }

        const char *name = ext.GetString();

        if (strcmp(name, "algo") == 0) {
            setExtension(EXT_ALGO, true);
        }
        else if (strcmp(name, "nicehash") == 0) {
            setExtension(EXT_NICEHASH, true);
        }
        else if (strcmp(name, "connect") == 0) {
            setExtension(EXT_CONNECT, true);
        }
        else if (strcmp(name, "keepalive") == 0) {
            setExtension(EXT_KEEPALIVE, true);
        }
#       ifdef XMRIG_FEATURE_TLS
        else if (strcmp(name, "tls") == 0) {
            setExtension(EXT_TLS, true);
        }
#       endif
    }
}


void xmrig::Client::parseNotification(const char *method, const rapidjson::Value &params, const rapidjson::Value &error)
{
    if (error.IsObject()) {
        if (!isQuiet()) {
            LOG_ERR("[%s] error: \"%s\", code: %d", url(), error["message"].GetString(), error["code"].GetInt());
        }
        return;
    }

    if (!method) {
        return;
    }

    if (strcmp(method, "job") == 0) {
        int code = -1;
        if (parseJob(params, &code)) {
            m_listener->onJobReceived(this, m_job, params);
        }
        else {
            close();
        }

        return;
    }

    LOG_WARN("[%s] unsupported method: \"%s\"", url(), method);
}


void xmrig::Client::parseResponse(int64_t id, const rapidjson::Value &result, const rapidjson::Value &error)
{
    if (handleResponse(id, result, error)) {
        return;
    }

    if (error.IsObject()) {
        const char *message = error["message"].GetString();

        if (!handleSubmitResponse(id, message) && !isQuiet()) {
            LOG_ERR("[%s] error: " RED_BOLD("\"%s\"") RED_S ", code: %d", url(), message, error["code"].GetInt());
        }

        if (m_id == 1 || isCriticalError(message)) {
            close();
        }

        return;
    }

    if (!result.IsObject()) {
        return;
    }

    if (id == 1) {
        int code = -1;
        if (!parseLogin(result, &code)) {
            if (!isQuiet()) {
                LOG_ERR("[%s] login error code: %d", url(), code);
            }

            close();
            return;
        }

        m_failures = 0;
        m_listener->onLoginSuccess(this);
        m_listener->onJobReceived(this, m_job, result["job"]);
        return;
    }

    handleSubmitResponse(id);
}


void xmrig::Client::ping()
{
    send(snprintf(m_sendBuf, sizeof(m_sendBuf), "{\"id\":%" PRId64 ",\"jsonrpc\":\"2.0\",\"method\":\"keepalived\",\"params\":{\"id\":\"%s\"}}\n", m_sequence, m_rpcId.data()));

    m_keepAlive = 0;
}


void xmrig::Client::read(ssize_t nread)
{
    const auto size = static_cast<size_t>(nread);

    if (nread > 0 && size > m_recvBuf.available()) {
        nread = UV_ENOBUFS;
    }

    if (nread < 0) {
        if (!isQuiet()) {
            LOG_ERR("[%s] read error: \"%s\"", url(), uv_strerror(static_cast<int>(nread)));
        }

        close();
        return;
    }

    assert(m_listener != nullptr);
    if (!m_listener) {
        return reconnect();
    }

    m_recvBuf.nread(size);

#   ifdef XMRIG_FEATURE_TLS
    if (isTLS()) {
        LOG_DEBUG("[%s] TLS received (%d bytes)", url(), static_cast<int>(nread));

        m_tls->read(m_recvBuf.base(), m_recvBuf.pos());
        m_recvBuf.reset();
    }
    else
#   endif
    {
        m_recvBuf.getline(this);
    }
}


void xmrig::Client::reconnect()
{
    if (!m_listener) {
        m_storage.remove(m_key);

        return;
    }

    m_keepAlive = 0;

    if (m_failures == -1) {
        return m_listener->onClose(this, -1);
    }

    setState(ReconnectingState);

    m_failures++;
    m_listener->onClose(this, static_cast<int>(m_failures));
}


void xmrig::Client::setState(SocketState state)
{
    LOG_DEBUG("[%s] state: \"%s\" -> \"%s\"", url(), states[m_state], states[state]);

    if (m_state == state) {
        return;
    }

    switch (state) {
    case HostLookupState:
        m_expire = 0;
        break;

    case ConnectingState:
        m_expire = Chrono::steadyMSecs() + kConnectTimeout;
        break;

    case ReconnectingState:
        m_expire = Chrono::steadyMSecs() + m_retryPause;
        break;

    default:
        break;
    }

    m_state = state;
}


void xmrig::Client::startTimeout()
{
    m_expire = 0;

    if (has<EXT_KEEPALIVE>()) {
        const uint64_t ms = static_cast<uint64_t>(m_pool.keepAlive() > 0 ? m_pool.keepAlive() : Pool::kKeepAliveTimeout) * 1000;

        m_keepAlive = Chrono::steadyMSecs() + ms;
    }
}


void xmrig::Client::onAllocBuffer(uv_handle_t *handle, size_t, uv_buf_t *buf)
{
    auto client = getClient(handle->data);
    if (!client) {
        return;
    }

    buf->base = client->m_recvBuf.current();

#   ifdef _WIN32
    buf->len = static_cast<ULONG>(client->m_recvBuf.available());
#   else
    buf->len = client->m_recvBuf.available();
#   endif
}


void xmrig::Client::onClose(uv_handle_t *handle)
{
    auto client = getClient(handle->data);
    if (!client) {
        return;
    }

    client->onClose();
}


void xmrig::Client::onConnect(uv_connect_t *req, int status)
{
    auto client = getClient(req->data);
    if (!client) {
        delete req;
        return;
    }

    if (status < 0) {
        if (!client->isQuiet()) {
            LOG_ERR("[%s] connect error: \"%s\"", client->url(), uv_strerror(status));
        }

        if (client->state() == ReconnectingState) {
            return;
        }

        if (client->state() != ConnectingState) {
            if (!client->isQuiet()) {
                LOG_ERR("[%s] connect error: \"invalid state: %d\"", client->url(), client->state());
            }

            return;
        }

        delete req;
        client->close();
        return;
    }

    if (client->state() == ConnectedState) {
        LOG_ERR("[%s] already connected");

        return;
    }

    client->m_stream = static_cast<uv_stream_t*>(req->handle);
    client->m_stream->data = req->data;
    client->setState(ConnectedState);

    uv_read_start(client->m_stream, onAllocBuffer, onRead);
    delete req;

    client->handshake();
}


void xmrig::Client::onRead(uv_stream_t *stream, ssize_t nread, const uv_buf_t *)
{
    auto client = getClient(stream->data);
    if (client) {
        client->read(nread);
    }
}
