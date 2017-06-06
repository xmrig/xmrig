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


#include "Console.h"
#include "interfaces/IClientListener.h"
#include "net/Client.h"
#include "net/Url.h"


Client::Client(IClientListener *listener) :
    m_host(nullptr),
    m_listener(listener),
    m_retries(0),
    m_sequence(1),
    m_recvBufPos(0),
    m_state(UnconnectedState),
    m_port(0),
    m_stream(nullptr),
    m_socket(nullptr)
{
    m_resolver.data = this;

    m_hints.ai_family   = PF_INET;
    m_hints.ai_socktype = SOCK_STREAM;
    m_hints.ai_protocol = IPPROTO_TCP;
    m_hints.ai_flags    = 0;

    m_recvBuf.base = static_cast<char*>(malloc(kRecvBufSize));
    m_recvBuf.len  = kRecvBufSize;
}


Client::~Client()
{
    free(m_recvBuf.base);
    free(m_socket);
    free(m_host);
}


/**
 * @brief Connect to server.
 *
 * @param host
 * @param port
 */
void Client::connect(const char *host, uint16_t port)
{
    m_host = strdup(host);
    m_port = port;

    LOG_DEBUG("[%s:%u] connect", m_host, m_port);

    resolve(host);
}


/**
 * @brief Connect to server.
 *
 * @param url
 */
void Client::connect(const Url *url)
{
    connect(url->host(), url->port());
}


void Client::disconnect()
{
    close();
}


void Client::login(const char *user, const char *pass, const char *agent)
{
    m_sequence = 1;

    const size_t size = 96 + strlen(user) + strlen(pass) + strlen(agent);
    char *req = static_cast<char*>(malloc(size));
    snprintf(req, size, "{\"id\":%llu,\"jsonrpc\":\"2.0\",\"method\":\"login\",\"params\":{\"login\":\"%s\",\"pass\":\"%s\",\"agent\":\"%s\"}}\n", m_sequence, user, pass, agent);

    send(req);
}


/**
 * @brief Send raw data to server.
 *
 * @param data
 */
void Client::send(char *data)
{
    LOG_DEBUG("[%s:%u] send (%d bytes): \"%s\"", m_host, m_port, strlen(data), data);
    if (state() != ConnectedState) {
        LOG_DEBUG_ERR("[%s:%u] send failed, invalid state: %d", m_host, m_port, m_state);
        return;
    }

    m_sequence++;
    uv_buf_t buf = uv_buf_init(data, strlen(data));

    uv_write_t *req = static_cast<uv_write_t*>(malloc(sizeof(uv_write_t)));
    req->data = buf.base;

    uv_write(req, m_stream, &buf, 1, [](uv_write_t *req, int status) {
        if (status) {
            auto client = getClient(req->data);
            LOG_ERR("[%s:%u] write error: \"%s\"", client->m_host, client->m_port, uv_strerror(status));
        }

        free(req->data);
        free(req);
    });
}


int Client::resolve(const char *host)
{
    setState(HostLookupState);

    m_recvBufPos = 0;

    const int r = uv_getaddrinfo(uv_default_loop(), &m_resolver, Client::onResolved, host, NULL, &m_hints);
    if (r) {
        LOG_ERR("[%s:%u] getaddrinfo error: \"%s\"", host, m_port, uv_strerror(r));
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
    uv_close(reinterpret_cast<uv_handle_t*>(m_socket), Client::onClose);
}


void Client::connect(struct sockaddr *addr)
{
    setState(ConnectingState);

    reinterpret_cast<struct sockaddr_in*>(addr)->sin_port = htons(m_port);
    free(m_socket);

    uv_connect_t *req = (uv_connect_t*) malloc(sizeof(uv_connect_t));
    req->data = this;

    m_socket = static_cast<uv_tcp_t*>(malloc(sizeof(uv_tcp_t)));
    m_socket->data = this;

    uv_tcp_init(uv_default_loop(), m_socket);
    uv_tcp_nodelay(m_socket, 1);
    uv_tcp_keepalive(m_socket, 1, 60);

    uv_tcp_connect(req, m_socket, (const sockaddr*) addr, Client::onConnect);
}


void Client::parse(char *line, size_t len)
{
    line[len - 1] = '\0';

    LOG_DEBUG("[%s:%u] received (%d bytes): \"%s\"", m_host, m_port, len, line);

    json_error_t err;
    json_t *val = json_loads(line, 0, &err);

    if (!val) {
        LOG_ERR("[%s:%u] JSON decode failed: \"%s\"", m_host, m_port, err.text);
        return;
    }

    const json_t *id = json_object_get(val, "id");
    if (json_is_integer(id)) {
        parseResponse(json_integer_value(id), json_object_get(val, "result"), json_object_get(val, "error"));
    }
    else {
        parseNotification(json_string_value(json_object_get(val, "method")), json_object_get(val, "params"));
    }

    json_decref(val);
}


void Client::parseNotification(const char *method, const json_t *params)
{

}


void Client::parseResponse(int64_t id, const json_t *result, const json_t *error)
{
    if (json_is_object(error)) {
        LOG_ERR("[%s:%u] error: \"%s\", code: %lld", m_host, m_port, json_string_value(json_object_get(error, "message")), json_integer_value(json_object_get(error, "code")));

        if (id == 1) {
            close();
        }

        return;
    }
}


void Client::setState(SocketState state)
{
    LOG_DEBUG("[%s:%u] state: %d", m_host, m_port, state);

    if (m_state == state) {
        return;
    }

    m_state = state;
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
}


void Client::onConnect(uv_connect_t *req, int status)
{
    auto client = getClient(req->data);
    if (status < 0) {
        LOG_ERR("[%s:%u] connect error: \"%s\"", client->m_host, client->m_port, uv_strerror(status));
        free(req);
        return;
    }

    client->m_stream = static_cast<uv_stream_t*>(req->handle);
    client->m_stream->data = req->data;
    client->setState(ConnectedState);

    uv_read_start(client->m_stream, Client::onAllocBuffer, Client::onRead);
    free(req);

    client->m_listener->onLoginCredentialsRequired(client);
}


void Client::onRead(uv_stream_t *stream, ssize_t nread, const uv_buf_t *buf)
{
    auto client = getClient(stream->data);
    if (nread < 0) {
        if (nread != UV_EOF) {
            LOG_ERR("[%s:%u] read error: \"%s\"", client->m_host, client->m_port, uv_strerror(nread));
        }

        client->close();
        return;
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
        LOG_ERR("[%s:%u] DNS error: \"%s\"", client->m_host, client->m_port, uv_strerror(status));
        return;
    }

    client->connect(res->ai_addr);
    uv_freeaddrinfo(res);
}


Client *Client::getClient(void *data)
{
    return static_cast<Client*>(data);
}
