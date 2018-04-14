/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
 * Copyright 2018      Sebastian Stolzenberg <https://github.com/sebastianstolzenberg>
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
#include "net/JobResult.h"
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

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

    m_keepAliveTimer.data = this;

    uv_timer_init(uv_default_loop(), &m_keepAliveTimer);

    uv_mutex_init(&m_mutex);

    uv_async_init(uv_default_loop(), &onConnectedAsync, Client::onConnected);
    uv_async_init(uv_default_loop(), &onReceivedAsync, Client::onReceived);
    uv_async_init(uv_default_loop(), &onErrorAsync, Client::onError);
}


Client::~Client()
{
    uv_close((uv_handle_t*) &onConnectedAsync, NULL);
    uv_close((uv_handle_t*) &onReceivedAsync, NULL);
    uv_close((uv_handle_t*) &onErrorAsync, NULL);

    uv_mutex_destroy(&m_mutex);
}

void Client::connect(const Url *url)
{
    LOG_DEBUG("connect %s", url);

    setUrl(url);
    connect();
}


void Client::connect()
{
    LOG_DEBUG("connect");

    m_connection = establishConnection(shared_from_this(),
                                       m_url.useTls() ? CONNECTION_TYPE_TLS : CONNECTION_TYPE_TCP,
                                       m_url.host(), m_url.port());
}


void Client::disconnect()
{
    LOG_DEBUG("disconnect");

    uv_timer_stop(&m_keepAliveTimer);

    m_expire   = 0;
    m_failures = -1;

    close();
}


void Client::setUrl(const Url *url)
{
    LOG_DEBUG("setUrl");

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

    LOG_DEBUG("tick expired");

    if (m_connection) {
        LOG_WARN("[%s:%u] timeout", m_url.host(), m_url.port());
        LOG_DEBUG("tick -> reconnect");
        reconnect();
    }
    else {
        LOG_DEBUG("tick -> connect");
        connect();
    }
}


int64_t Client::submit(const JobResult &result)
{
    char nonce[9];
    char data[65];

    Job::toHex(reinterpret_cast<const unsigned char*>(&result.nonce), 4, nonce);
    nonce[8] = '\0';

    Job::toHex(result.result, 32, data);
    data[64] = '\0';

    const int size = snprintf(m_sendBuf, sizeof(m_sendBuf),
                              "{"
                                      "\"id\":%" PRIu64 ","
                                      "\"jsonrpc\":\"2.0\","
                                      "\"method\":\"submit\","
                                      "\"params\":"
                                      "{"
                                      "\"id\":\"%s\","
                                      "\"job_id\":\"%s\","
                                      "\"nonce\":\"%s\","
                                      "\"result\":\"%s\""
                                      "}"
                                      "}\n",
                              m_sequence, m_rpcId, result.jobId.data(), nonce, data);

    m_results[m_sequence] = SubmitResult(m_sequence, result.diff, result.actualDiff());
    return send(m_sendBuf, size);
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

        switch (variantFromProxy) {
            case -1:
                job.setPowVersion(Options::POW_AUTODETECT);
                break;
            case 0:
                job.setPowVersion(Options::POW_V1);
                break;
            case 1:
                job.setPowVersion(Options::POW_V2);
                break;
            default:
                break;
        }
    } else {
        job.setPowVersion(Options::i()->forcePowVersion());
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

    reconnect();
    return false;
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


bool Client::parseLogin(const rapidjson::Value &result, int *code)
{
    const char *id = result["id"].GetString();
    if (!id || strlen(id) >= sizeof(m_rpcId)) {
        *code = 1;
        return false;
    }

    m_nicehash = m_url.isNicehash();

    if (result.HasMember("extensions")) {
        parseExtensions(result["extensions"]);
    }

    memset(m_rpcId, 0, sizeof(m_rpcId));
    memcpy(m_rpcId, id, strlen(id));

    const bool rc = parseJob(result["job"], code);
    m_jobs = 0;

    return rc;
}


int64_t Client::send(char* buf, size_t size)
{
    if (m_connection)
    {
      m_connection->send(buf, size);
      m_expire = uv_now(uv_default_loop()) + kResponseTimeout;
      m_sequence++;
    }

    return m_sequence;
}


void Client::close()
{
    LOG_DEBUG("close");

    m_connection.reset();
}


void Client::login()
{
    LOG_DEBUG("login");

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

    send(m_sendBuf, size + 1);
}

void Client::processReceivedData(char* data, size_t size)
{
    LOG_DEBUG("processReceivedData");

    if ((size_t) size > (sizeof(m_buf) - 8 - m_recvBufPos)) {
        reconnect();
        return;
    }

    m_recvBufPos += size;

    char* end;
    char* start = data;
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

    if (start == data) {
        return;
    }

    memcpy(data, start, remaining);
    m_recvBufPos = remaining;
}

void Client::parse(char *line, size_t len)
{
    LOG_DEBUG("parse");

    startTimeout();

    line[len - 1] = '\0';

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


void Client::parseNotification(const char *method, const rapidjson::Value &params, const rapidjson::Value &error)
{
    if (error.IsObject()) {
        if (!m_quiet) {
            LOG_ERR("[%s:%u] Parse notification failed: \"%s\", code: %d", m_url.host(), m_url.port(), error["message"].GetString(), error["code"].GetInt());
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

    LOG_WARN("[%s:%u] Unsupported method: \"%s\"", m_url.host(), m_url.port(), method);
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
            LOG_ERR("[%s:%u] Parse response failed: \"%s\", code: %d", m_url.host(), m_url.port(), message, error["code"].GetInt());
        }

        if (id == 1 || isCriticalError(message)) {
            reconnect();
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
                LOG_ERR("[%s:%u] Login error code: %d", m_url.host(), m_url.port(), code);
            }

            return reconnect();
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
    LOG_DEBUG("ping");

    const int size = snprintf(m_sendBuf, sizeof(m_sendBuf),
                              "{"
                                      "\"id\":%" PRId64 ","
                                      "\"jsonrpc\":\"2.0\","
                                      "\"method\":\"keepalived\","
                                      "\"params\":"
                                      "{"
                                      "\"id\":\"%s\""
                                      "}"
                                      "}\n",
                              m_sequence, m_rpcId);
    send(m_sendBuf, size);
}


void Client::reconnect()
{
    LOG_DEBUG("reconnect");

    close();

    if (m_url.isKeepAlive()) {
        uv_timer_stop(&m_keepAliveTimer);
    }

    if (m_failures == -1) {
        LOG_DEBUG("reconnect -> m_failures == -1");
        return m_listener->onClose(this, -1);
    }

    m_failures++;
    m_listener->onClose(this, (int) m_failures);

    m_expire = uv_now(uv_default_loop()) + m_retryPause;
}


void Client::startTimeout()
{
    LOG_DEBUG("startTimeout");

    m_expire = 0;

    if (!m_url.isKeepAlive()) {
        return;
    }

    uv_timer_start(&m_keepAliveTimer, [](uv_timer_t *handle) { getClient(handle->data)->ping(); }, kKeepAliveTimeout, 0);
}

void Client::onConnected(uv_async_t *handle)
{
    LOG_DEBUG("onConnected");

    auto client = getClient(handle->data);
    if (client) {
        client->login();
    }
}

void Client::scheduleOnConnected()
{
    LOG_DEBUG("scheduleOnConnected");
    onConnectedAsync.data = this;

    uv_async_send(&onConnectedAsync);
}

void Client::onReceived(uv_async_t *handle)
{
    LOG_DEBUG("onReceived");

    auto client = getClient(handle->data);
    if (client) {
        uv_mutex_lock(&client->m_mutex);

        while (!client->m_readQueue.empty()) {
            std::string data = client->m_readQueue.front();
            client->processReceivedData(const_cast<char *>(data.c_str()), data.size());
            client->m_readQueue.pop_front();
        }

        uv_mutex_unlock(&client->m_mutex);
    }
}

void Client::scheduleOnReceived(char* data, std::size_t size)
{
    LOG_DEBUG("scheduleOnReceived");

    uv_mutex_lock(&m_mutex);
    m_readQueue.emplace_back(data, size);
    uv_mutex_unlock(&m_mutex);

    onReceivedAsync.data = this;
    uv_async_send(&onReceivedAsync);
}

void Client::onError(uv_async_t *handle)
{
    LOG_DEBUG("onError");

    auto client = getClient(handle->data);
    if (client) {
        client->reconnect();
    }
}

void Client::scheduleOnError(const std::string &error)
{
    LOG_DEBUG("scheduleOnError");

    if (!m_quiet) {
        LOG_ERR("[%s:%u] Error: \"%s\"", m_url.host(), m_url.port(), error.c_str());
    }

    onErrorAsync.data = this;
    uv_async_send(&onErrorAsync);
}
