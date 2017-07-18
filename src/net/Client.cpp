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

#include <inttypes.h>
#include <iterator>
#include <string.h>
#include <utility>

#include "log/Log.h"
#include "interfaces/IClientListener.h"
#include "net/Client.h"
#include "net/Url.h"


#ifdef XMRIG_PROXY_PROJECT
#   include "proxy/JobResult.h"
#else
#   include "net/JobResult.h"
#endif


#ifdef _MSC_VER
#   define strncasecmp(x,y,z) _strnicmp(x,y,z)
#endif


int64_t Client::m_sequence = 1;


Client::Client(int id, const char *agent, IClientListener *listener) :
    m_quiet(false),
    m_agent(agent),
    m_listener(listener),
    m_id(id),
    m_retryPause(5000),
    m_failures(0),
    m_recvBufPos(0),
    m_state(UnconnectedState),
    m_stream(nullptr),
    m_socket(nullptr)
{
    memset(m_ip, 0, sizeof(m_ip));
    memset(&m_hints, 0, sizeof(m_hints));

    m_resolver.data = m_responseTimer.data = m_retriesTimer.data = m_keepAliveTimer.data = this;

    m_hints.ai_family   = PF_INET;
    m_hints.ai_socktype = SOCK_STREAM;
    m_hints.ai_protocol = IPPROTO_TCP;

    m_recvBuf.base = static_cast<char*>(malloc(kRecvBufSize));
    m_recvBuf.len  = kRecvBufSize;

    auto loop = uv_default_loop();
    uv_timer_init(loop, &m_retriesTimer);
    uv_timer_init(loop, &m_responseTimer);
    uv_timer_init(loop, &m_keepAliveTimer);
}


Client::~Client()
{
    free(m_recvBuf.base);
    free(m_socket);
}


/**
 * @brief Send raw data to server.
 *
 * @param data
 */
int64_t Client::send(char *data, size_t size)
{
    LOG_DEBUG("[%s:%u] send (%d bytes): \"%s\"", m_url.host(), m_url.port(), size ? size : strlen(data), data);
    if (state() != ConnectedState) {
        LOG_DEBUG_ERR("[%s:%u] send failed, invalid state: %d", m_url.host(), m_url.port(), m_state);
        return -1;
    }

    uv_buf_t buf = uv_buf_init(data, size ? size : strlen(data));

    uv_write_t *req = static_cast<uv_write_t*>(malloc(sizeof(uv_write_t)));
    req->data = buf.base;

    uv_write(req, m_stream, &buf, 1, [](uv_write_t *req, int status) {
        free(req->data);
        free(req);
    });

    uv_timer_start(&m_responseTimer, [](uv_timer_t *handle) { getClient(handle->data)->close(); }, kResponseTimeout, 0);

    return m_sequence++;
}


void Client::connect()
{
    resolve(m_url.host());
}


/**
 * @brief Connect to server.
 *
 * @param url
 */
void Client::connect(const Url *url)
{
    setUrl(url);
    resolve(m_url.host());
}


void Client::disconnect()
{
    uv_timer_stop(&m_keepAliveTimer);
    uv_timer_stop(&m_responseTimer);
    uv_timer_stop(&m_retriesTimer);
    m_failures = -1;

    close();
}


void Client::setUrl(const Url *url)
{
    if (!url || !url->isValid()) {
        return;
    }

    m_url = url;
}


int64_t Client::submit(const JobResult &result)
{
    char *req = static_cast<char*>(malloc(345));

#   ifdef XMRIG_PROXY_PROJECT
    const char *nonce = result.nonce;
    const char *data  = result.result;
#   else
    char nonce[9];
    char data[65];

    Job::toHex(reinterpret_cast<const unsigned char*>(&result.nonce), 4, nonce);
    nonce[8] = '\0';

    Job::toHex(result.result, 32, data);
    data[64] = '\0';
#   endif

    snprintf(req, 345, "{\"id\":%" PRIu64 ",\"jsonrpc\":\"2.0\",\"method\":\"submit\",\"params\":{\"id\":\"%s\",\"job_id\":\"%s\",\"nonce\":\"%s\",\"result\":\"%s\"}}\n",
             m_sequence, m_rpcId, result.jobId, nonce, data);

    m_results[m_sequence] = SubmitResult(m_sequence, result.diff);
    return send(req);
}


bool Client::parseJob(const json_t *params, int *code)
{
    if (!json_is_object(params)) {
        *code = 2;
        return false;
    }

    Job job(m_id, m_url.isNicehash());
    if (!job.setId(json_string_value(json_object_get(params, "job_id")))) {
        *code = 3;
        return false;
    }

    if (!job.setBlob(json_string_value(json_object_get(params, "blob")))) {
        *code = 4;
        return false;
    }

    if (!job.setTarget(json_string_value(json_object_get(params, "target")))) {
        *code = 5;
        return false;
    }

    m_job = std::move(job);
    return true;
}


bool Client::parseLogin(const json_t *result, int *code)
{
    const char *id = json_string_value(json_object_get(result, "id"));
    if (!id || strlen(id) >= sizeof(m_rpcId)) {
        *code = 1;
        return false;
    }

    memset(m_rpcId, 0, sizeof(m_rpcId));
    memcpy(m_rpcId, id, strlen(id));

    return parseJob(json_object_get(result, "job"), code);
}


int Client::resolve(const char *host)
{
    setState(HostLookupState);

    m_recvBufPos = 0;

    if (m_failures == -1) {
        m_failures = 0;
    }

    const int r = uv_getaddrinfo(uv_default_loop(), &m_resolver, Client::onResolved, host, NULL, &m_hints);
    if (r) {
        if (!m_quiet) {
            LOG_ERR("[%s:%u] getaddrinfo error: \"%s\"", host, m_url.port(), uv_strerror(r));
        }
        return 1;
    }

    return 0;
}


void Client::close()
{
    if (m_state == UnconnectedState || m_state == ClosingState || !m_socket) {
        return;
    }

    setState(ClosingState);

    if (uv_is_closing(reinterpret_cast<uv_handle_t*>(m_socket)) == 0) {
        uv_close(reinterpret_cast<uv_handle_t*>(m_socket), Client::onClose);
    }
}


void Client::connect(struct sockaddr *addr)
{
    setState(ConnectingState);

    reinterpret_cast<struct sockaddr_in*>(addr)->sin_port = htons(m_url.port());
    free(m_socket);

    uv_connect_t *req = (uv_connect_t*) malloc(sizeof(uv_connect_t));
    req->data = this;

    m_socket = static_cast<uv_tcp_t*>(malloc(sizeof(uv_tcp_t)));
    m_socket->data = this;

    uv_tcp_init(uv_default_loop(), m_socket);
    uv_tcp_nodelay(m_socket, 1);

#   ifndef WIN32
    uv_tcp_keepalive(m_socket, 1, 60);
#   endif

    uv_tcp_connect(req, m_socket, reinterpret_cast<const sockaddr*>(addr), Client::onConnect);
}


void Client::login()
{
    m_results.clear();

    json_t *req = json_object();
    json_object_set(req, "id", json_integer(1));
    json_object_set(req, "jsonrpc", json_string("2.0"));
    json_object_set(req, "method", json_string("login"));

    json_t *params = json_object();
    json_object_set(params, "login", json_string(m_url.user()));
    json_object_set(params, "pass", json_string(m_url.password()));
    json_object_set(params, "agent", json_string(m_agent));

    json_object_set(req, "params", params);

    char *buf = json_dumps(req, JSON_COMPACT);
    const size_t size = strlen(buf);

    buf[size] = '\n';

    json_decref(req);

    send(buf, size + 1);
}


void Client::parse(char *line, size_t len)
{
    startTimeout();

    line[len - 1] = '\0';

    LOG_DEBUG("[%s:%u] received (%d bytes): \"%s\"", m_url.host(), m_url.port(), len, line);

    json_error_t err;
    json_t *val = json_loads(line, 0, &err);

    if (!val) {
        if (!m_quiet) {
            LOG_ERR("[%s:%u] JSON decode failed: \"%s\"", m_url.host(), m_url.port(), err.text);
        }
        return;
    }

    const json_t *id = json_object_get(val, "id");
    if (json_is_integer(id)) {
        parseResponse(json_integer_value(id), json_object_get(val, "result"), json_object_get(val, "error"));
    }
    else {
        parseNotification(json_string_value(json_object_get(val, "method")), json_object_get(val, "params"), json_object_get(val, "error"));
    }

    json_decref(val);
}


void Client::parseNotification(const char *method, const json_t *params, const json_t *error)
{
    if (json_is_object(error)) {
        if (!m_quiet) {
            LOG_ERR("[%s:%u] error: \"%s\", code: %" PRId64, m_url.host(), m_url.port(), json_string_value(json_object_get(error, "message")), json_integer_value(json_object_get(error, "code")));
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

    LOG_WARN("[%s:%u] unsupported method: \"%s\"", m_url.host(), m_url.port(), method);
}


void Client::parseResponse(int64_t id, const json_t *result, const json_t *error)
{
    if (json_is_object(error)) {
        const char *message = json_string_value(json_object_get(error, "message"));

        auto it = m_results.find(id);
        if (it != m_results.end()) {
            m_listener->onResultAccepted(this, it->second.seq, it->second.diff, it->second.elapsed(), message);
            m_results.erase(it);
        }
        else if (!m_quiet) {
            LOG_ERR("[%s:%u] error: \"%s\", code: %" PRId64, m_url.host(), m_url.port(), message, json_integer_value(json_object_get(error, "code")));
        }

        if (id == 1 || (message && strncasecmp(message, "Unauthenticated", 15) == 0)) {
            close();
        }

        return;
    }

    if (!json_is_object(result)) {
        return;
    }

    if (id == 1) {
        int code = -1;
        if (!parseLogin(result, &code)) {
            if (!m_quiet) {
                LOG_ERR("[%s:%u] login error code: %d", m_url.host(), m_url.port(), code);
            }

            return close();
        }

        m_failures = 0;
        m_listener->onLoginSuccess(this);
        m_listener->onJobReceived(this, m_job);
        return;
    }

    auto it = m_results.find(id);
    if (it != m_results.end()) {
        m_listener->onResultAccepted(this, it->second.seq, it->second.diff, it->second.elapsed(), nullptr);
        m_results.erase(it);
    }
}


void Client::ping()
{
    char *req = static_cast<char*>(malloc(128));
    snprintf(req, 128, "{\"id\":%" PRId64 ",\"jsonrpc\":\"2.0\",\"method\":\"keepalived\",\"params\":{\"id\":\"%s\"}}\n", m_sequence, m_rpcId);

    send(req);
}


void Client::reconnect()
{
    setState(ConnectingState);

    uv_timer_stop(&m_responseTimer);
    if (m_url.isKeepAlive()) {
        uv_timer_stop(&m_keepAliveTimer);
    }

    if (m_failures == -1) {
        return m_listener->onClose(this, -1);
    }

    m_failures++;
    m_listener->onClose(this, m_failures);

    uv_timer_start(&m_retriesTimer, [](uv_timer_t *handle) { getClient(handle->data)->connect(); }, m_retryPause, 0);
}


void Client::setState(SocketState state)
{
    LOG_DEBUG("[%s:%u] state: %d", m_url.host(), m_url.port(), state);

    if (m_state == state) {
        return;
    }

    m_state = state;
}


void Client::startTimeout()
{
    uv_timer_stop(&m_responseTimer);
    if (!m_url.isKeepAlive()) {
        return;
    }

    uv_timer_start(&m_keepAliveTimer, [](uv_timer_t *handle) { getClient(handle->data)->ping(); }, kKeepAliveTimeout, 0);
}


void Client::onAllocBuffer(uv_handle_t *handle, size_t suggested_size, uv_buf_t *buf)
{
    auto client = getClient(handle->data);

    buf->base = &client->m_recvBuf.base[client->m_recvBufPos];
    buf->len  = client->m_recvBuf.len - client->m_recvBufPos;
}


void Client::onClose(uv_handle_t *handle)
{
    auto client = getClient(handle->data);

    free(client->m_socket);

    client->m_stream = nullptr;
    client->m_socket = nullptr;
    client->setState(UnconnectedState);

    client->reconnect();
}


void Client::onConnect(uv_connect_t *req, int status)
{
    auto client = getClient(req->data);
    if (status < 0) {
        if (!client->m_quiet) {
            LOG_ERR("[%s:%u] connect error: \"%s\"", client->m_url.host(), client->m_url.port(), uv_strerror(status));
        }

        free(req);
        client->close();
        return;
    }

    client->m_stream = static_cast<uv_stream_t*>(req->handle);
    client->m_stream->data = req->data;
    client->setState(ConnectedState);

    uv_read_start(client->m_stream, Client::onAllocBuffer, Client::onRead);
    free(req);

    client->login();
}


void Client::onRead(uv_stream_t *stream, ssize_t nread, const uv_buf_t *buf)
{
    auto client = getClient(stream->data);
    if (nread < 0) {
        if (nread != UV_EOF && !client->m_quiet) {
            LOG_ERR("[%s:%u] read error: \"%s\"", client->m_url.host(), client->m_url.port(), uv_strerror(nread));
        }

        return client->close();;
    }

    client->m_recvBufPos += nread;

    char* end;
    char* start = client->m_recvBuf.base;
    size_t remaining = client->m_recvBufPos;

    while ((end = static_cast<char*>(memchr(start, '\n', remaining))) != nullptr) {
        end++;
        size_t len = end - start;
        client->parse(start, len);

        remaining -= len;
        start = end;
    }

    if (remaining == 0) {
        client->m_recvBufPos = 0;
        return;
    }

    if (start == client->m_recvBuf.base) {
        return;
    }

    memcpy(client->m_recvBuf.base, start, remaining);
    client->m_recvBufPos = remaining;
}


void Client::onResolved(uv_getaddrinfo_t *req, int status, struct addrinfo *res)
{
    auto client = getClient(req->data);
    if (status < 0) {
        LOG_ERR("[%s:%u] DNS error: \"%s\"", client->m_url.host(), client->m_url.port(), uv_strerror(status));
        return client->reconnect();;
    }

    addrinfo *ptr = res;
    std::vector<addrinfo*> ipv4;

    while (ptr != nullptr) {
        if (ptr->ai_family == AF_INET) {
            ipv4.push_back(ptr);
        }

        ptr = ptr->ai_next;
    }

    if (ipv4.empty()) {
        LOG_ERR("[%s:%u] DNS error: \"No IPv4 records found\"", client->m_url.host(), client->m_url.port());
        return client->reconnect();
    }

    ptr = ipv4[rand() % ipv4.size()];

    uv_ip4_name(reinterpret_cast<sockaddr_in*>(ptr->ai_addr), client->m_ip, 16);

    client->connect(ptr->ai_addr);
    uv_freeaddrinfo(res);
}
