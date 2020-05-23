/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2019      jtgrassie   <https://github.com/jtgrassie>
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


#include "base/net/stratum/SelfSelectClient.h"
#include "3rdparty/http-parser/http_parser.h"
#include "3rdparty/rapidjson/document.h"
#include "3rdparty/rapidjson/error/en.h"
#include "base/io/json/Json.h"
#include "base/io/json/JsonRequest.h"
#include "base/io/log/Log.h"
#include "base/net/http/Fetch.h"
#include "base/net/http/HttpData.h"
#include "base/net/stratum/Client.h"


namespace xmrig {

static const char *kBlob                = "blob";
static const char *kBlockhashingBlob    = "blockhashing_blob";
static const char *kBlocktemplateBlob   = "blocktemplate_blob";
static const char *kDifficulty          = "difficulty";
static const char *kHeight              = "height";
static const char *kId                  = "id";
static const char *kJobId               = "job_id";
static const char *kNextSeedHash        = "next_seed_hash";
static const char *kPrevHash            = "prev_hash";
static const char *kSeedHash            = "seed_hash";

static const char * const required_fields[] = { kBlocktemplateBlob, kBlockhashingBlob, kHeight, kDifficulty, kPrevHash };

} /* namespace xmrig */


xmrig::SelfSelectClient::SelfSelectClient(int id, const char *agent, IClientListener *listener) :
    m_listener(listener)
{
    m_httpListener  = std::make_shared<HttpListener>(this);
    m_client        = new Client(id, agent, this);
}


xmrig::SelfSelectClient::~SelfSelectClient()
{
    delete m_client;
}


void xmrig::SelfSelectClient::tick(uint64_t now)
{
    m_client->tick(now);

    if (m_state == RetryState) {
        if (Chrono::steadyMSecs() - m_timestamp < m_retryPause) {
            return;
        }

        getBlockTemplate();
    }
}


void xmrig::SelfSelectClient::onJobReceived(IClient *, const Job &job, const rapidjson::Value &)
{
    m_job = job;

    getBlockTemplate();
}


void xmrig::SelfSelectClient::onLogin(IClient *, rapidjson::Document &doc, rapidjson::Value &params)
{
    params.AddMember("mode", "self-select", doc.GetAllocator());

    m_listener->onLogin(this, doc, params);
}


bool xmrig::SelfSelectClient::parseResponse(int64_t id, rapidjson::Value &result, const rapidjson::Value &error)
{
    if (id == -1) {
        return false;
    }

    if (error.IsObject()) {
        LOG_ERR("[%s] error: " RED_BOLD("\"%s\"") RED_S ", code: %d", pool().daemon().url().data(), Json::getString(error, "message"), Json::getInt(error, "code"));

        return false;
    }

    if (!result.IsObject()) {
        return false;
    }

    for (auto field : required_fields) {
        if (!result.HasMember(field)) {
            LOG_ERR("[%s] required field " RED_BOLD("\"%s\"") RED_S " not found", pool().daemon().url().data(), field);

            return false;
        }
    }

    if (!m_job.setBlob(result[kBlockhashingBlob].GetString())) {
        return false;
    }

    if (pool().coin().isValid()) {
        m_job.setAlgorithm(pool().coin().algorithm(m_job.blob()[0]));
    }

    m_job.setHeight(Json::getUint64(result, kHeight));
    m_job.setSeedHash(Json::getString(result, kSeedHash));

    submitBlockTemplate(result);

    return true;
}


void xmrig::SelfSelectClient::getBlockTemplate()
{
    setState(WaitState);

    using namespace rapidjson;
    Document doc(kObjectType);
    auto &allocator = doc.GetAllocator();

    Value params(kObjectType);
    params.AddMember("wallet_address",  m_job.poolWallet().toJSON(), allocator);
    params.AddMember("extra_nonce",     m_job.extraNonce().toJSON(), allocator);

    JsonRequest::create(doc, m_sequence++, "getblocktemplate", params);

    FetchRequest req(HTTP_POST, pool().daemon().host(), pool().daemon().port(), "/json_rpc", doc, pool().daemon().isTLS(), isQuiet());
    fetch(std::move(req), m_httpListener);
}


void xmrig::SelfSelectClient::retry()
{
    setState(RetryState);
}


void xmrig::SelfSelectClient::setState(State state)
{
    if (m_state == state) {
        return;
    }

    switch (state) {
    case IdleState:
        m_timestamp = 0;
        m_failures  = 0;
        break;

    case WaitState:
        m_timestamp = Chrono::steadyMSecs();
        break;

    case RetryState:
        m_timestamp = Chrono::steadyMSecs();

        if (m_failures > m_retries) {
            m_listener->onClose(this, static_cast<int>(m_failures));
        }

        m_failures++;
        break;
    }

    m_state = state;
}


void xmrig::SelfSelectClient::submitBlockTemplate(rapidjson::Value &result)
{
    using namespace rapidjson;
    Document doc(kObjectType);
    auto &allocator = doc.GetAllocator();

    Value params(kObjectType);
    params.AddMember(StringRef(kId),            m_job.clientId().toJSON(), allocator);
    params.AddMember(StringRef(kJobId),         m_job.id().toJSON(), allocator);
    params.AddMember(StringRef(kBlob),          result[kBlocktemplateBlob], allocator);
    params.AddMember(StringRef(kHeight),        m_job.height(), allocator);
    params.AddMember(StringRef(kDifficulty),    result[kDifficulty], allocator);
    params.AddMember(StringRef(kPrevHash),      result[kPrevHash], allocator);
    params.AddMember(StringRef(kSeedHash),      result[kSeedHash], allocator);
    params.AddMember(StringRef(kNextSeedHash),  result[kNextSeedHash], allocator);

    JsonRequest::create(doc, sequence(), "block_template", params);

    send(doc, [this](const rapidjson::Value &result, bool success, uint64_t) {
        if (!success) {
            if (!isQuiet()) {
                LOG_ERR("[%s] error: " RED_BOLD("\"%s\"") RED_S ", code: %d", pool().daemon().url().data(), Json::getString(result, "message"), Json::getInt(result, "code"));
            }

            return retry();
        }

        if (!m_active) {
            return;
        }

        if (m_failures > m_retries) {
            m_listener->onLoginSuccess(this);
        }

        setState(IdleState);
        m_listener->onJobReceived(this, m_job, rapidjson::Value{});
    });
}


void xmrig::SelfSelectClient::onHttpData(const HttpData &data)
{
    if (data.status != HTTP_STATUS_OK) {
        return retry();
    }

    rapidjson::Document doc;
    if (doc.Parse(data.body.c_str()).HasParseError()) {
        if (!isQuiet()) {
            LOG_ERR("[%s] JSON decode failed: \"%s\"",  pool().daemon().url().data(), rapidjson::GetParseError_En(doc.GetParseError()));
        }

        return retry();
    }

    const int64_t id = Json::getInt64(doc, "id", -1);
    if (id > 0 && m_sequence - id != 1) {
        return;
    }

    if (!parseResponse(id, doc["result"], Json::getObject(doc, "error"))) {
        retry();
    }
}
