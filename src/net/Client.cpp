/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
 * Copyright 2018-     BenDr0id    <ben@graef.in>
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
#include <uv.h>
#include <Options.h>

#include "interfaces/IClientListener.h"
#include "log/Log.h"
#include "net/Client.h"
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"


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
    m_nicehash(false),
    m_agent(agent),
    m_listener(listener),
    m_id(id),
    m_retryPause(5000),
    m_failures(0),
    m_jobs(0),
    m_recvBufPos(0),
    m_expire(0)
{
    m_recvBuf.base = m_buf;
    m_recvBuf.len  = sizeof(m_buf);

#   ifndef XMRIG_PROXY_PROJECT
    m_keepAliveTimer.data = this;
    uv_timer_init(uv_default_loop(), &m_keepAliveTimer);
#   endif
}


Client::~Client()
{
    if (m_net) {
        net_free(m_net);
    }
}


/**
 * @brief Connect to server.
 *
 * @param url
 */
void Client::connect(const Url *url)
{
    setUrl(url);
    connect();
}


void Client::disconnect()
{
    LOG_DEBUG("Client::disconnect");

#   ifndef XMRIG_PROXY_PROJECT
    uv_timer_stop(&m_keepAliveTimer);
#   endif

    m_expire   = 0;
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


void Client::tick(uint64_t now)
{
    if (m_expire == 0 || now < m_expire) {
        return;
    }

    if (m_net) {
        LOG_WARN("[%s:%u] timeout", m_url.host(), m_url.port());
        close();
    }
    else {
        LOG_DEBUG("Client::tick -> connect");
        connect();
    }
}


int64_t Client::submit(const JobResult &result)
{
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

    const size_t size = snprintf(m_sendBuf, sizeof(m_sendBuf), "{\"id\":%" PRIu64 ",\"jsonrpc\":\"2.0\",\"method\":\"submit\",\"params\":{\"id\":\"%s\",\"job_id\":\"%s\",\"nonce\":\"%s\",\"result\":\"%s\"}}\n",
                                 m_sequence, m_rpcId, result.jobId.data(), nonce, data);

    m_results[m_sequence] = SubmitResult(m_sequence, result.diff, result.actualDiff());
    return send(size);
}


bool Client::isCriticalError(const char *message)
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


bool Client::parseJob(const rapidjson::Value &params, int *code)
{
    if (!params.IsObject()) {
        *code = 2;
        return false;
    }

    Job job(m_id, m_nicehash);
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

    if (params.HasMember("variant")) {
        int variantFromProxy = params["variant"].GetInt();

        if (Options::i()->forcePowVersion() == Options::POW_AUTODETECT) {
            switch (variantFromProxy) {
                case -1:
                    Options::i()->setForcePowVersion(Options::POW_AUTODETECT);
                    break;
                case 0:
                    Options::i()->setForcePowVersion(Options::POW_V1);
                    break;
                case 1:
                    Options::i()->setForcePowVersion(Options::POW_V2);
                    break;
                default:
                    break;
            }
        }
    }

    if (m_job != job) {
        m_jobs++;
        m_job = std::move(job);
        return true;
    }

    if (m_jobs == 0) { // https://github.com/xmrig/xmrig/issues/459
        return false;
    }

    if (!m_quiet) {
        LOG_WARN("[%s:%u] duplicate job received, reconnect", m_url.host(), m_url.port());
    }

    close();
    return false;
}


bool Client::parseLogin(const rapidjson::Value &result, int *code)
{
    const char *id = result["id"].GetString();
    if (!id || strlen(id) >= sizeof(m_rpcId)) {
        *code = 1;
        return false;
    }

#   ifndef XMRIG_PROXY_PROJECT
    m_nicehash = m_url.isNicehash();
#   endif

    if (result.HasMember("extensions")) {
        parseExtensions(result["extensions"]);
    }

    memset(m_rpcId, 0, sizeof(m_rpcId));
    memcpy(m_rpcId, id, strlen(id));

    const bool rc = parseJob(result["job"], code);
    m_jobs = 0;

    return rc;
}

int64_t Client::send(size_t size)
{
    LOG_DEBUG("Client::send");

    LOG_DEBUG("[%s:%u] send (%d bytes): \"%s\"", m_url.host(), m_url.port(), size, m_sendBuf);
    if (!m_net) {
        LOG_DEBUG_ERR("[%s:%u] send failed", m_url.host(), m_url.port());
        return -1;
    }

    if (net_write2(m_net, m_sendBuf, static_cast<unsigned int>(size)) < 0) {
        close();
        return -1;
    }

    m_expire = uv_now(uv_default_loop()) + kResponseTimeout;
    return m_sequence++;
}


void Client::close()
{
    LOG_DEBUG("Client::close");

    if (m_net) {
        net_close(m_net, nullptr);
    } else {
        reconnect();
    }
}


void Client::connect()
{
    LOG_DEBUG("Client::connect");

    m_net = net_new(const_cast<char *>(m_url.host()), m_url.port());
    m_net->data = this;
    m_net->conn_cb = Client::onConnect;
    m_net->read_cb = Client::onRead;
    m_net->close_cb = Client::onClose;
    m_net->error_cb = Client::onError;

#ifndef XMRIG_NO_TLS
    if (m_url.useTls()) {
        tls_ctx* tls_ctx = tls_ctx_new();
        net_set_tls(m_net, tls_ctx);
    }
#endif

    net_connect(m_net);
}

void Client::onRead(net_t *net, size_t size, char *buf)
{
    LOG_DEBUG("Client::onRead");

    auto client = getClient(net->data);

    if (size == 0) {
        if (size != UV_EOF && !client->m_quiet) {
            LOG_ERR("[%s:%u] read error: \"%s\"", client->m_url.host(), client->m_url.port(), uv_strerror((int) size));
        }

        return client->close();
    }

    client->m_recvBufPos += size;

    char* end;
    char* start = buf;
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

    if (start == buf) {
        return;
    }

    memcpy(buf, start, remaining);
    client->m_recvBufPos = remaining;
}

void Client::onConnect(net_t *net) {
    LOG_DEBUG("Client::onConnect");
    auto client = getClient(net->data);
    client->login();
}

void Client::onError(net_t *net, int err, char *errStr)
{
    LOG_DEBUG("Client::onError");

    if (net) {
        auto client = getClient(net->data);
        if (!client->m_quiet) {
            LOG_ERR("[%s:%u] error: \"%s\"", client->m_url.host(), client->m_url.port(), errStr);
        }

        client->close();
    }
}

void Client::onClose(net_t *net)
{
    LOG_DEBUG("Client::onClose");

    if (net) {
        auto client = getClient(net->data);

        net_free(net);

        client->m_net = nullptr;

        client->reconnect();
    }
}

void Client::login()
{
    LOG_DEBUG("Client::login");

    m_results.clear();

    rapidjson::Document doc;
    doc.SetObject();

    auto &allocator = doc.GetAllocator();

    doc.AddMember("id",      1,       allocator);
    doc.AddMember("jsonrpc", "2.0",   allocator);
    doc.AddMember("method",  "login", allocator);

    rapidjson::Value params(rapidjson::kObjectType);
    params.AddMember("login", rapidjson::StringRef(m_url.user()),     allocator);
    params.AddMember("pass",  rapidjson::StringRef(m_url.password()), allocator);
    params.AddMember("agent", rapidjson::StringRef(m_agent),          allocator);

    doc.AddMember("params", params, allocator);

    rapidjson::StringBuffer buffer(0, 512);
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);

    const size_t size = buffer.GetSize();
    if (size > (sizeof(m_buf) - 2)) {
        return;
    }

    memcpy(m_sendBuf, buffer.GetString(), size);
    m_sendBuf[size]     = '\n';
    m_sendBuf[size + 1] = '\0';

    send(size + 1);
}


void Client::parse(char *line, size_t len)
{
    LOG_DEBUG("Client::parse");

    startTimeout();

    line[len - 1] = '\0';

    LOG_DEBUG("[%s:%u] received (%d bytes): \"%s\"", m_url.host(), m_url.port(), len, line);

    rapidjson::Document doc;
    if (doc.ParseInsitu(line).HasParseError()) {
        if (!m_quiet) {
            LOG_ERR("[%s:%u] JSON decode failed: \"%s\"", m_url.host(), m_url.port(), rapidjson::GetParseError_En(doc.GetParseError()));
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


void Client::parseExtensions(const rapidjson::Value &value)
{
    if (!value.IsArray()) {
        return;
    }

    for (const rapidjson::Value &ext : value.GetArray()) {
        if (!ext.IsString()) {
            continue;
        }

        if (strcmp(ext.GetString(), "nicehash") == 0) {
            m_nicehash = true;
        }
    }
}

void Client::parseNotification(const char *method, const rapidjson::Value &params, const rapidjson::Value &error)
{
    if (error.IsObject()) {
        if (!m_quiet) {
            LOG_ERR("[%s:%u] error: \"%s\", code: %d", m_url.host(), m_url.port(), error["message"].GetString(), error["code"].GetInt());
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


void Client::parseResponse(int64_t id, const rapidjson::Value &result, const rapidjson::Value &error)
{
    if (error.IsObject()) {
        const char *message = error["message"].GetString();

        auto it = m_results.find(id);
        if (it != m_results.end()) {
            it->second.done();
            m_listener->onResultAccepted(this, it->second, message);
            m_results.erase(it);
        }
        else if (!m_quiet) {
            LOG_ERR("[%s:%u] error: \"%s\", code: %d", m_url.host(), m_url.port(), message, error["code"].GetInt());
        }

        if (id == 1 || isCriticalError(message)) {
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
        it->second.done();
        m_listener->onResultAccepted(this, it->second, nullptr);
        m_results.erase(it);
    }
}


void Client::ping()
{
    LOG_DEBUG("Client::ping");
    send(snprintf(m_sendBuf, sizeof(m_sendBuf), "{\"id\":%" PRId64 ",\"jsonrpc\":\"2.0\",\"method\":\"keepalived\",\"params\":{\"id\":\"%s\"}}\n", m_sequence, m_rpcId));
}


void Client::reconnect() {

    LOG_DEBUG("Client::reconnect");

#   ifndef XMRIG_PROXY_PROJECT
    if (m_url.isKeepAlive()) {
        uv_timer_stop(&m_keepAliveTimer);
    }
#   endif

    if (m_failures == -1) {
        LOG_DEBUG("Client::onConnect -> m_failures == -1");
        return m_listener->onClose(this, -1);
    }

    m_failures++;
    m_listener->onClose(this, (int) m_failures);

    m_expire = uv_now(uv_default_loop()) + m_retryPause;
}

void Client::startTimeout()
{
    LOG_DEBUG("Client::startTimeout");

    m_expire = 0;

#   ifndef XMRIG_PROXY_PROJECT
    if (!m_url.isKeepAlive()) {
        return;
    }

    uv_timer_start(&m_keepAliveTimer, [](uv_timer_t *handle) { getClient(handle->data)->ping(); }, kKeepAliveTimeout, 0);
#   endif
}
