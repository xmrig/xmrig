/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2019      Howard Chu  <https://github.com/hyc>
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


#include <algorithm>
#include <assert.h>


#include "3rdparty/http-parser/http_parser.h"
#include "base/io/Json.h"
#include "base/io/log/Log.h"
#include "base/kernel/interfaces/IClientListener.h"
#include "base/net/http/HttpClient.h"
#include "base/net/stratum/DaemonClient.h"
#include "base/net/stratum/SubmitResult.h"
#include "base/tools/Buffer.h"
#include "base/tools/Timer.h"
#include "net/JobResult.h"
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"


#ifdef XMRIG_FEATURE_TLS
#   include "base/net/http/HttpsClient.h"
#endif


xmrig::DaemonClient::DaemonClient(int id, IClientListener *listener) :
    BaseClient(id, listener)
{
    m_timer = new Timer(this);
}


xmrig::DaemonClient::~DaemonClient()
{
    delete m_timer;
}


bool xmrig::DaemonClient::disconnect()
{
    setState(UnconnectedState);

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
    if (result.jobId != (m_blocktemplate.data() + m_blocktemplate.size() - 48)) {
        return -1;
    }

    Buffer::toHex(reinterpret_cast<const uint8_t *>(&result.nonce), 4, m_blocktemplate.data() + 78);

    using namespace rapidjson;
    Document doc(kObjectType);
    auto &allocator = doc.GetAllocator();

    doc.AddMember("id",      m_sequence,    allocator);
    doc.AddMember("jsonrpc", "2.0",         allocator);
    doc.AddMember("method",  "submitblock", allocator);

    Value params(kArrayType);
    params.PushBack(m_blocktemplate.toJSON(), allocator);

    doc.AddMember("params", params, allocator);

#   ifdef XMRIG_PROXY_PROJECT
    m_results[m_sequence] = SubmitResult(m_sequence, result.diff, result.actualDiff(), result.id);
#   else
    m_results[m_sequence] = SubmitResult(m_sequence, result.diff, result.actualDiff());
#   endif

    send(HTTP_POST, "/json_rpc", doc);

    return m_sequence++;
}


void xmrig::DaemonClient::connect()
{
    setState(ConnectingState);
    getBlockTemplate();
}


void xmrig::DaemonClient::connect(const Pool &pool)
{
    setPool(pool);
    connect();
}


void xmrig::DaemonClient::onHttpData(const HttpData &data)
{
    if (data.status != HTTP_STATUS_OK) {
        return retry();
    }

    LOG_DEBUG("[%s:%d] received (%d bytes): \"%.*s\"", m_pool.host().data(), m_pool.port(), static_cast<int>(data.body.size()), static_cast<int>(data.body.size()), data.body.c_str());

    m_ip = static_cast<const HttpContext &>(data).ip().c_str();

#   ifdef XMRIG_FEATURE_TLS
    if (isTLS()) {
        m_tlsVersion     = static_cast<const HttpsClient &>(data).version();
        m_tlsFingerprint = static_cast<const HttpsClient &>(data).fingerprint();
    }
#   endif

    rapidjson::Document doc;
    if (doc.Parse(data.body.c_str()).HasParseError()) {
        if (!isQuiet()) {
            LOG_ERR("[%s:%d] JSON decode failed: \"%s\"", m_pool.host().data(), m_pool.port(), rapidjson::GetParseError_En(doc.GetParseError()));
        }

        return retry();
    }

    if (data.method == HTTP_GET && data.url == "/getheight") {
        if (m_job.height() != Json::getUint64(doc, "height")) {
            getBlockTemplate();
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
        getBlockTemplate();
    }
    else if (m_state == ConnectedState) {
        send(HTTP_GET, "/getheight");
    }
}


bool xmrig::DaemonClient::parseJob(const rapidjson::Value &params, int *code)
{
    Job job(m_id, false, m_pool.algorithm(), String());

    String blocktemplate = Json::getString(params, "blocktemplate_blob");
    if (blocktemplate.isNull() || !job.setBlob(Json::getString(params, "blockhashing_blob"))) {
        *code = 4;
        return false;
    }

    job.setHeight(Json::getUint64(params, "height"));
    job.setDiff(Json::getUint64(params, "difficulty"));
    job.setId(blocktemplate.data() + blocktemplate.size() - 48);

    m_job           = std::move(job);
    m_blocktemplate = std::move(blocktemplate);

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

    int code = -1;
    if (result.HasMember("blocktemplate_blob") && parseJob(result, &code)) {
        return true;
    }

    if (handleSubmitResponse(id)) {
        getBlockTemplate();
        return true;
    }


    return false;
}


int64_t xmrig::DaemonClient::getBlockTemplate()
{
    using namespace rapidjson;
    Document doc(kObjectType);
    auto &allocator = doc.GetAllocator();

    doc.AddMember("id",      m_sequence,         allocator);
    doc.AddMember("jsonrpc", "2.0",              allocator);
    doc.AddMember("method",  "getblocktemplate", allocator);

    Value params(kObjectType);
    params.AddMember("wallet_address", m_pool.user().toJSON(), allocator);
    params.AddMember("reserve_size",   8,                      allocator);

    doc.AddMember("params", params, allocator);

    send(HTTP_POST, "/json_rpc", doc);

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

    m_timer->stop();
    m_timer->start(m_retryPause, 0);
}


void xmrig::DaemonClient::send(int method, const char *url, const char *data, size_t size)
{
    LOG_DEBUG("[%s:%d] " MAGENTA_BOLD("\"%s %s\"") BLACK_BOLD_S " send (%zu bytes): \"%.*s\"",
              m_pool.host().data(),
              m_pool.port(),
              http_method_str(static_cast<http_method>(method)),
              url,
              size,
              static_cast<int>(size),
              data);

    HttpClient *client;
#   ifdef XMRIG_FEATURE_TLS
    if (m_pool.isTLS()) {
        client = new HttpsClient(method, url, this, data, size, m_pool.fingerprint());
    }
    else
#   endif
    {
        client = new HttpClient(method, url, this, data, size);
    }

    client->setQuiet(isQuiet());
    client->connect(m_pool.host(), m_pool.port());
}


void xmrig::DaemonClient::send(int method, const char *url, const rapidjson::Document &doc)
{
    using namespace rapidjson;

    StringBuffer buffer(nullptr, 512);
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);

    send(method, url, buffer.GetString(), buffer.GetSize());
}


void xmrig::DaemonClient::setState(SocketState state)
{
    assert(m_state != state);
    if (m_state == state) {
        return;
    }

    m_state = state;

    switch (state) {
    case ConnectedState:
        {
            m_failures = 0;
            m_listener->onLoginSuccess(this);

            const uint64_t interval = std::max<uint64_t>(20, m_pool.pollInterval());
            m_timer->start(interval, interval);
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
