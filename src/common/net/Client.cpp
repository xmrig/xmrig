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

#include <assert.h>
#include <inttypes.h>
#include <iterator>
#include <stdio.h>
#include <string.h>
#include <utility>

#ifdef _MSC_VER
#define strncasecmp(x,y,z) _strnicmp(x,y,z)
#include <WinSock2.h>
#endif

#ifndef XMRIG_NO_TLS
#   include <openssl/ssl.h>
#   include <openssl/err.h>
#   include "common/net/Tls.h"
#endif

#include "common/interfaces/IClientListener.h"
#include "common/log/Log.h"
#include "common/net/Client.h"
#include "net/JobResult.h"
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"


namespace xmrig {

int64_t Client::m_sequence = 1;
Storage<Client> Client::m_storage;

} /* namespace xmrig */


#ifdef APP_DEBUG
static const char *states[] = {
    "unconnected",
    "host-lookup",
    "connecting",
    "connected",
    "closing"
};
#endif


xmrig::Client::Client(int id, const char *agent, IClientListener *listener) :
    m_ipv6(false),
    m_nicehash(false),
    m_quiet(false),
    m_agent(agent),
    m_listener(listener),
    m_extensions(0),
    m_id(id),
    m_retries(5),
    m_retryPause(5000),
    m_failures(0),
    m_recvBufPos(0),
    m_state(UnconnectedState),
    m_tls(nullptr),
    m_expire(0),
    m_jobs(0),
    m_keepAlive(0),
    m_key(0),
    m_stream(nullptr),
    m_socket(nullptr)
{
    m_key = m_storage.add(this);

    memset(m_ip, 0, sizeof(m_ip));
    memset(&m_hints, 0, sizeof(m_hints));

    m_resolver.data = m_storage.ptr(m_key);

    m_hints.ai_family   = AF_UNSPEC;
    m_hints.ai_socktype = SOCK_STREAM;
    m_hints.ai_protocol = IPPROTO_TCP;

    m_recvBuf.base = m_buf;
    m_recvBuf.len  = sizeof(m_buf);
}


xmrig::Client::~Client()
{
    delete m_socket;
}


void xmrig::Client::connect()
{
#   ifndef XMRIG_NO_TLS
    if (m_pool.isTLS()) {
        m_tls = new Tls(this);
    }
#   endif

    resolve(m_pool.host());
}


/**
 * @brief Connect to server.
 *
 * @param url
 */
void xmrig::Client::connect(const Pool &url)
{
    setPool(url);
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



void xmrig::Client::setPool(const Pool &pool)
{
    if (!pool.isValid()) {
        return;
    }

    m_pool = pool;
}


void xmrig::Client::tick(uint64_t now)
{
    if (m_state == ConnectedState) {
        if (m_expire && now > m_expire) {
            LOG_DEBUG_ERR("[%s] timeout", m_pool.url());
            close();
        }
        else if (m_keepAlive && now > m_keepAlive) {
            ping();
        }
    }

    if (m_expire && now > m_expire && m_state == ConnectingState) {
        connect();
    }
}


bool xmrig::Client::disconnect()
{
    m_keepAlive = 0;
    m_expire    = 0;
    m_failures  = -1;

    return close();
}


const char *xmrig::Client::tlsFingerprint() const
{
#   ifndef XMRIG_NO_TLS
    if (isTLS() && m_pool.fingerprint() == nullptr) {
        return m_tls->fingerprint();
    }
#   endif

    return nullptr;
}


const char *xmrig::Client::tlsVersion() const
{
#   ifndef XMRIG_NO_TLS
    if (isTLS()) {
        return m_tls->version();
    }
#   endif

    return nullptr;
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

    Job::toHex(reinterpret_cast<const unsigned char*>(&result.nonce), 4, nonce);
    nonce[8] = '\0';

    Job::toHex(result.result, 32, data);
    data[64] = '\0';
#   endif

    Document doc(kObjectType);
    auto &allocator = doc.GetAllocator();

    doc.AddMember("id",      m_sequence, allocator);
    doc.AddMember("jsonrpc", "2.0", allocator);
    doc.AddMember("method",  "submit", allocator);

    Value params(kObjectType);
    params.AddMember("id",     StringRef(m_rpcId.data()), allocator);
    params.AddMember("job_id", StringRef(result.jobId.data()), allocator);
    params.AddMember("nonce",  StringRef(nonce), allocator);
    params.AddMember("result", StringRef(data), allocator);

    if (m_extensions & AlgoExt) {
        params.AddMember("algo", StringRef(result.algorithm.shortName()), allocator);
    }

    doc.AddMember("params", params, allocator);

#   ifdef XMRIG_PROXY_PROJECT
    m_results[m_sequence] = SubmitResult(m_sequence, result.diff, result.actualDiff(), result.id);
#   else
    m_results[m_sequence] = SubmitResult(m_sequence, result.diff, result.actualDiff());
#   endif

    return send(doc);
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

    return false;
}


bool xmrig::Client::isTLS() const
{
#   ifndef XMRIG_NO_TLS
    return m_pool.isTLS() && m_tls;
#   else
    return false;
#   endif
}


bool xmrig::Client::parseJob(const rapidjson::Value &params, int *code)
{
    if (!params.IsObject()) {
        *code = 2;
        return false;
    }

    Job job(m_id, m_nicehash, m_pool.algorithm(), m_rpcId);

    if (!job.setId(params["job_id"].GetString())) {
        *code = 3;
        return false;
    }

    if (!job.setBlob(params["blob"].GetString())) {
        *code = 4;
        return false;
    }

    if (!job.setTarget(params["target"].GetString())) {
        *code = 5;
        return false;
    }

    if (params.HasMember("algo")) {
        job.setAlgorithm(params["algo"].GetString());
    }

    if (params.HasMember("variant")) {
        const rapidjson::Value &variant = params["variant"];

        if (variant.IsInt()) {
            job.setVariant(variant.GetInt());
        }
        else if (variant.IsString()){
            job.setVariant(variant.GetString());
        }
    }

    if (params.HasMember("height")) {
        const rapidjson::Value &variant = params["height"];

        if (variant.IsUint64()) {
            job.setHeight(variant.GetUint64());
        }
    }

    if (!verifyAlgorithm(job.algorithm())) {
        *code = 6;

        close();
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
        LOG_WARN("[%s] duplicate job received, reconnect", m_pool.url());
    }

    close();
    return false;
}


bool xmrig::Client::parseLogin(const rapidjson::Value &result, int *code)
{
    if (!m_rpcId.setId(result["id"].GetString())) {
        *code = 1;
        return false;
    }

    m_nicehash = m_pool.isNicehash();

    if (result.HasMember("extensions")) {
        parseExtensions(result["extensions"]);
    }

    const bool rc = parseJob(result["job"], code);
    m_jobs = 0;

    return rc;
}


bool xmrig::Client::send(BIO *bio)
{
#   ifndef XMRIG_NO_TLS
    uv_buf_t buf;
    buf.len = BIO_get_mem_data(bio, &buf.base);

    if (buf.len == 0) {
        return true;
    }

    LOG_DEBUG("[%s] TLS send     (%d bytes)", m_pool.url(), static_cast<int>(buf.len));

    bool result = false;
    if (state() == ConnectedState && uv_is_writable(m_stream)) {
        result = uv_try_write(m_stream, &buf, 1) > 0;

        if (!result) {
            close();
        }
    }
    else {
        LOG_DEBUG_ERR("[%s] send failed, invalid state: %d", m_pool.url(), m_state);
    }

    (void) BIO_reset(bio);

    return result;
#   else
    return false;
#   endif
}


bool xmrig::Client::verifyAlgorithm(const Algorithm &algorithm) const
{
#   ifdef XMRIG_PROXY_PROJECT
    if (m_pool.algorithm().variant() == VARIANT_AUTO || m_id == -1) {
        return true;
    }
#   endif

    if (m_pool.isCompatible(algorithm)) {
        return true;
    }

    if (isQuiet()) {
        return false;
    }

    if (algorithm.isValid()) {
        LOG_ERR("Incompatible algorithm \"%s\" detected, reconnect", algorithm.name());
    }
    else {
        LOG_ERR("Unknown/unsupported algorithm detected, reconnect");
    }

    return false;
}


int xmrig::Client::resolve(const char *host)
{
    setState(HostLookupState);

    m_expire     = 0;
    m_recvBufPos = 0;

    if (m_failures == -1) {
        m_failures = 0;
    }

    const int r = uv_getaddrinfo(uv_default_loop(), &m_resolver, Client::onResolved, host, nullptr, &m_hints);
    if (r) {
        if (!isQuiet()) {
            LOG_ERR("[%s:%u] getaddrinfo error: \"%s\"", host, m_pool.port(), uv_strerror(r));
        }
        return 1;
    }

    return 0;
}


int64_t xmrig::Client::send(const rapidjson::Document &doc)
{
    using namespace rapidjson;

    StringBuffer buffer(0, 512);
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);

    const size_t size = buffer.GetSize();
    if (size > (sizeof(m_sendBuf) - 2)) {
        LOG_ERR("[%s] send failed: \"send buffer overflow: %zu > %zu\"", m_pool.url(), size, (sizeof(m_sendBuf) - 2));
        close();
        return -1;
    }

    memcpy(m_sendBuf, buffer.GetString(), size);
    m_sendBuf[size]     = '\n';
    m_sendBuf[size + 1] = '\0';

    return send(size + 1);
}


int64_t xmrig::Client::send(size_t size)
{
    LOG_DEBUG("[%s] send (%d bytes): \"%s\"", m_pool.url(), size, m_sendBuf);

#   ifndef XMRIG_NO_TLS
    if (isTLS()) {
        if (!m_tls->send(m_sendBuf, size)) {
            return -1;
        }
    }
    else
#   endif
    {
        if (state() != ConnectedState || !uv_is_writable(m_stream)) {
            LOG_DEBUG_ERR("[%s] send failed, invalid state: %d", m_pool.url(), m_state);
            return -1;
        }

        uv_buf_t buf = uv_buf_init(m_sendBuf, (unsigned int) size);

        if (uv_try_write(m_stream, &buf, 1) < 0) {
            close();
            return -1;
        }
    }

    m_expire = uv_now(uv_default_loop()) + kResponseTimeout;
    return m_sequence++;
}


void xmrig::Client::connect(const std::vector<addrinfo*> &ipv4, const std::vector<addrinfo*> &ipv6)
{
    addrinfo *addr = nullptr;
    m_ipv6         = ipv4.empty() && !ipv6.empty();

    if (m_ipv6) {
        addr = ipv6[ipv6.size() == 1 ? 0 : rand() % ipv6.size()];
        uv_ip6_name(reinterpret_cast<sockaddr_in6*>(addr->ai_addr), m_ip, 45);
    }
    else {
        addr = ipv4[ipv4.size() == 1 ? 0 : rand() % ipv4.size()];
        uv_ip4_name(reinterpret_cast<sockaddr_in*>(addr->ai_addr), m_ip, 16);
    }

    connect(addr->ai_addr);
}


void xmrig::Client::connect(sockaddr *addr)
{
    setState(ConnectingState);

    reinterpret_cast<sockaddr_in*>(addr)->sin_port = htons(m_pool.port());

    uv_connect_t *req = new uv_connect_t;
    req->data = m_storage.ptr(m_key);

    m_socket = new uv_tcp_t;
    m_socket->data = m_storage.ptr(m_key);

    uv_tcp_init(uv_default_loop(), m_socket);
    uv_tcp_nodelay(m_socket, 1);

#   ifndef WIN32
    uv_tcp_keepalive(m_socket, 1, 60);
#   endif

    uv_tcp_connect(req, m_socket, reinterpret_cast<const sockaddr*>(addr), Client::onConnect);
}


void xmrig::Client::handshake()
{
#   ifndef XMRIG_NO_TLS
    if (isTLS()) {
        m_expire = uv_now(uv_default_loop()) + kResponseTimeout;

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

    doc.AddMember("id",      1,       allocator);
    doc.AddMember("jsonrpc", "2.0",   allocator);
    doc.AddMember("method",  "login", allocator);

    Value params(kObjectType);
    params.AddMember("login", StringRef(m_pool.user()),     allocator);
    params.AddMember("pass",  StringRef(m_pool.password()), allocator);
    params.AddMember("agent", StringRef(m_agent),           allocator);

    if (m_pool.rigId()) {
        params.AddMember("rigid", StringRef(m_pool.rigId()), allocator);
    }

#   ifdef XMRIG_PROXY_PROJECT
    if (m_pool.algorithm().variant() != xmrig::VARIANT_AUTO)
#   endif
    {
        Value algo(kArrayType);

        for (const auto &a : m_pool.algorithms()) {
            algo.PushBack(StringRef(a.shortName()), allocator);
        }

        params.AddMember("algo", algo, allocator);
    }

    doc.AddMember("params", params, allocator);

    send(doc);
}


void xmrig::Client::onClose()
{
    delete m_socket;

    m_stream = nullptr;
    m_socket = nullptr;
    setState(UnconnectedState);

#   ifndef XMRIG_NO_TLS
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

    line[len - 1] = '\0';

    LOG_DEBUG("[%s] received (%d bytes): \"%s\"", m_pool.url(), len, line);

    if (len < 32 || line[0] != '{') {
        if (!isQuiet()) {
            LOG_ERR("[%s] JSON decode failed", m_pool.url());
        }

        return;
    }

    rapidjson::Document doc;
    if (doc.ParseInsitu(line).HasParseError()) {
        if (!isQuiet()) {
            LOG_ERR("[%s] JSON decode failed: \"%s\"", m_pool.url(), rapidjson::GetParseError_En(doc.GetParseError()));
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


void xmrig::Client::parseExtensions(const rapidjson::Value &value)
{
    m_extensions = 0;

    if (!value.IsArray()) {
        return;
    }

    for (const rapidjson::Value &ext : value.GetArray()) {
        if (!ext.IsString()) {
            continue;
        }

        if (strcmp(ext.GetString(), "algo") == 0) {
            m_extensions |= AlgoExt;
            continue;
        }

        if (strcmp(ext.GetString(), "nicehash") == 0) {
            m_extensions |= NicehashExt;
            m_nicehash = true;
            continue;
        }
    }
}


void xmrig::Client::parseNotification(const char *method, const rapidjson::Value &params, const rapidjson::Value &error)
{
    if (error.IsObject()) {
        if (!isQuiet()) {
            LOG_ERR("[%s] error: \"%s\", code: %d", m_pool.url(), error["message"].GetString(), error["code"].GetInt());
        }
        return;
    }

    if (!method) {
        return;
    }

    if (strcmp(method, "job") == 0) {
        int code = -1;
        if (parseJob(params, &code)) {
            m_listener->onJobReceived(this, m_job);
        }

        return;
    }

    LOG_WARN("[%s] unsupported method: \"%s\"", m_pool.url(), method);
}


void xmrig::Client::parseResponse(int64_t id, const rapidjson::Value &result, const rapidjson::Value &error)
{
    if (error.IsObject()) {
        const char *message = error["message"].GetString();

        auto it = m_results.find(id);
        if (it != m_results.end()) {
            it->second.done();
            m_listener->onResultAccepted(this, it->second, message);
            m_results.erase(it);
        }
        else if (!isQuiet()) {
            LOG_ERR("[%s] error: \"%s\", code: %d", m_pool.url(), message, error["code"].GetInt());
        }

        if (isCriticalError(message)) {
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
                LOG_ERR("[%s] login error code: %d", m_pool.url(), code);
            }

            close();
            return;
        }

        m_failures = 0;
        m_listener->onLoginSuccess(this);
        m_listener->onJobReceived(this, m_job);
        return;
    }

    auto it = m_results.find(id);
    if (it != m_results.end()) {
        it->second.done();
        m_listener->onResultAccepted(this, it->second, nullptr);
        m_results.erase(it);
    }
}


void xmrig::Client::ping()
{
    send(snprintf(m_sendBuf, sizeof(m_sendBuf), "{\"id\":%" PRId64 ",\"jsonrpc\":\"2.0\",\"method\":\"keepalived\",\"params\":{\"id\":\"%s\"}}\n", m_sequence, m_rpcId.data()));
}


void xmrig::Client::read()
{
    char* end;
    char* start = m_recvBuf.base;
    size_t remaining = m_recvBufPos;

    while ((end = static_cast<char*>(memchr(start, '\n', remaining))) != nullptr) {
        end++;
        size_t len = end - start;
        parse(start, len);

        remaining -= len;
        start = end;
    }

    if (remaining == 0) {
        m_recvBufPos = 0;
        return;
    }

    if (start == m_recvBuf.base) {
        return;
    }

    memcpy(m_recvBuf.base, start, remaining);
    m_recvBufPos = remaining;
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

    setState(ConnectingState);

    m_failures++;
    m_listener->onClose(this, (int) m_failures);

    m_expire = uv_now(uv_default_loop()) + m_retryPause;
}


void xmrig::Client::setState(SocketState state)
{
    LOG_DEBUG("[%s] state: \"%s\"", m_pool.url(), states[state]);

    if (m_state == state) {
        return;
    }

    m_state = state;
}


void xmrig::Client::startTimeout()
{
    m_expire = 0;

    if (m_pool.keepAlive()) {
        m_keepAlive = uv_now(uv_default_loop()) + (m_pool.keepAlive() * 1000);
    }
}


void xmrig::Client::onAllocBuffer(uv_handle_t *handle, size_t suggested_size, uv_buf_t *buf)
{
    auto client = getClient(handle->data);
    if (!client) {
        return;
    }

    buf->base = &client->m_recvBuf.base[client->m_recvBufPos];
    buf->len  = client->m_recvBuf.len - client->m_recvBufPos;
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
            LOG_ERR("[%s] connect error: \"%s\"", client->m_pool.url(), uv_strerror(status));
        }

        delete req;
        client->close();
        return;
    }

    client->m_stream = static_cast<uv_stream_t*>(req->handle);
    client->m_stream->data = req->data;
    client->setState(ConnectedState);

    uv_read_start(client->m_stream, Client::onAllocBuffer, Client::onRead);
    delete req;

    client->handshake();
}


void xmrig::Client::onRead(uv_stream_t *stream, ssize_t nread, const uv_buf_t *buf)
{
    auto client = getClient(stream->data);
    if (!client) {
        return;
    }

    if (nread < 0) {
        if (!client->isQuiet()) {
            LOG_ERR("[%s] read error: \"%s\"", client->m_pool.url(), uv_strerror((int) nread));
        }

        client->close();
        return;
    }

    if ((size_t) nread > (sizeof(m_buf) - 8 - client->m_recvBufPos)) {
        client->close();
        return;
    }

    assert(client->m_listener != nullptr);
    if (!client->m_listener) {
        return client->reconnect();
    }

    client->m_recvBufPos += nread;

#   ifndef XMRIG_NO_TLS
    if (client->isTLS()) {
        LOG_DEBUG("[%s] TLS received (%d bytes)", client->m_pool.url(), static_cast<int>(nread));

        client->m_tls->read(client->m_recvBuf.base, client->m_recvBufPos);
        client->m_recvBufPos = 0;
    }
    else
#   endif
    {
        client->read();
    }
}


void xmrig::Client::onResolved(uv_getaddrinfo_t *req, int status, struct addrinfo *res)
{
    auto client = getClient(req->data);
    if (!client) {
        return;
    }

    assert(client->m_listener != nullptr);
    if (!client->m_listener) {
        return client->reconnect();
    }

    if (status < 0) {
        if (!client->isQuiet()) {
            LOG_ERR("[%s] DNS error: \"%s\"", client->m_pool.url(), uv_strerror(status));
        }

        return client->reconnect();
    }

    addrinfo *ptr = res;
    std::vector<addrinfo*> ipv4;
    std::vector<addrinfo*> ipv6;

    while (ptr != nullptr) {
        if (ptr->ai_family == AF_INET) {
            ipv4.push_back(ptr);
        }

        if (ptr->ai_family == AF_INET6) {
            ipv6.push_back(ptr);
        }

        ptr = ptr->ai_next;
    }

    if (ipv4.empty() && ipv6.empty()) {
        if (!client->isQuiet()) {
            LOG_ERR("[%s] DNS error: \"No IPv4 (A) or IPv6 (AAAA) records found\"", client->m_pool.url());
        }

        uv_freeaddrinfo(res);
        return client->reconnect();
    }

    client->connect(ipv4, ipv6);
    uv_freeaddrinfo(res);
}
