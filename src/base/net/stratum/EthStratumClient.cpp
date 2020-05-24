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

#include <cinttypes>
#include <sstream>
#include <iomanip>

#include "3rdparty/libethash/endian.h"
#include "3rdparty/rapidjson/document.h"
#include "3rdparty/rapidjson/error/en.h"
#include "3rdparty/rapidjson/stringbuffer.h"
#include "3rdparty/rapidjson/writer.h"

#include "base/io/json/Json.h"
#include "base/io/json/JsonRequest.h"
#include "base/io/log/Log.h"
#include "base/kernel/interfaces/IClientListener.h"
#include "base/net/stratum/EthStratumClient.h"

#include "net/JobResult.h"


namespace xmrig {


EthStratumClient::EthStratumClient(int id, const char *agent, IClientListener *listener) :
    Client(id, agent, listener)
{
}


void EthStratumClient::login()
{
    using namespace rapidjson;
    m_results.clear();

    Document doc(kObjectType);
    auto& allocator = doc.GetAllocator();

    Value params(kArrayType);
    params.PushBack(StringRef(agent()), allocator);

    JsonRequest::create(doc, m_sequence, "mining.subscribe", params);

    send(doc, [this](const rapidjson::Value& result, bool success, uint64_t elapsed) { OnSubscribeResponse(result, success, elapsed); });
}


void EthStratumClient::onClose()
{
    m_authorized = false;
    Client::onClose();
}


int64_t EthStratumClient::submit(const JobResult& result)
{
#   ifndef XMRIG_PROXY_PROJECT
    if ((m_state != ConnectedState) || !m_authorized) {
        return -1;
    }
#   endif

    if (result.diff == 0) {
        disconnect();
        return -1;
    }

    using namespace rapidjson;

    Document doc(kObjectType);
    auto& allocator = doc.GetAllocator();

    Value params(kArrayType);
    params.PushBack(StringRef(m_pool.user().data()), allocator);
    params.PushBack(StringRef(result.jobId.data()), allocator);

    std::stringstream s;
    s << "0x" << std::hex << std::setw(16) << std::setfill('0') << result.nonce;
    params.PushBack(rapidjson::Value(s.str().c_str(), allocator), allocator);

    s.str(std::string());
    s << "0x";
    for (size_t i = 0; i < 32; ++i) {
        const uint32_t k = result.headerHash()[i];
        s << std::hex << std::setw(2) << std::setfill('0') << k;
    }
    params.PushBack(rapidjson::Value(s.str().c_str(), allocator), allocator);

    s.str(std::string());
    s << "0x";
    for (size_t i = 0; i < 32; ++i) {
        const uint32_t k = result.mixHash()[i];
        s << std::hex << std::setw(2) << std::setfill('0') << k;
    }
    params.PushBack(rapidjson::Value(s.str().c_str(), allocator), allocator);

    JsonRequest::create(doc, m_sequence, "mining.submit", params);

    uint64_t actual_diff = ethash_swap_u64(*((uint64_t*)result.result()));
    actual_diff = actual_diff ? (uint64_t(-1) / actual_diff) : 0;

#   ifdef XMRIG_PROXY_PROJECT
    m_results[m_sequence] = SubmitResult(m_sequence, result.diff, actual_diff, result.id, 0);
#   else
    m_results[m_sequence] = SubmitResult(m_sequence, result.diff, actual_diff, 0, result.backend);
#   endif

    return send(doc);
}


bool EthStratumClient::handleResponse(int64_t id, const rapidjson::Value& result, const rapidjson::Value& error)
{
    auto it = m_callbacks.find(id);
    if (it != m_callbacks.end()) {
        const uint64_t elapsed = Chrono::steadyMSecs() - it->second.ts;

        if (error.IsObject()) {
            it->second.callback(error, false, elapsed);
        }
        else {
            it->second.callback(result, true, elapsed);
        }

        m_callbacks.erase(it);

        return true;
    }

    const char* err = nullptr;
    if (error.IsArray() && error.GetArray().Size() > 1) {
        auto& value = error.GetArray()[1];
        if (value.IsString()) {
            err = value.GetString();
        }
    }

    handleSubmitResponse(id, err);
    return false;
}


void EthStratumClient::parseNotification(const char* method, const rapidjson::Value& params, const rapidjson::Value& error)
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

    if (strcmp(method, "mining.set_target") == 0) {
        if (!params.IsArray()) {
            LOG_ERR("Invalid mining.set_target notification: params is not an array");
            return;
        }

        if (params.GetArray().Size() != 1) {
            LOG_ERR("Invalid mining.set_target notification: params array has wrong size");
            return;
        }
 
        auto& new_target = params.GetArray()[0];
        if (!new_target.IsString()) {
            LOG_ERR("Invalid mining.set_target notification: target is not a string");
            return;
        }

        std::string new_target_str = new_target.GetString();
        new_target_str.resize(16, '0');

        m_target = std::stoull(new_target_str, nullptr, 16);
        LOG_DEBUG("Target set to %016" PRIX64, m_target);

        return;
    }

    if (strcmp(method, "mining.notify") == 0) {
        if (!params.IsArray()) {
            LOG_ERR("Invalid mining.notify notification: params is not an array");
            return;
        }

        auto arr = params.GetArray();

        if (arr.Size() < 6) {
            LOG_ERR("Invalid mining.notify notification: params array has wrong size");
            return;
        }

        Job job;
        job.setId(arr[0].GetString());

        Algorithm algo = m_pool.algorithm();
        if (algo == Algorithm::INVALID) {
            algo = m_pool.coin().algorithm(0);
        }
        job.setAlgorithm(algo);

        std::stringstream s;

        // header hash (32 bytes)
        s << arr[1].GetString();

        // nonce template (8 bytes)
        for (uint64_t i = 0, k = m_extraNonce; i < sizeof(m_extraNonce); ++i, k >>= 8) {
            s << std::hex << std::setw(2) << std::setfill('0') << (k & 0xFF);
        }

        std::string blob = s.str();

        // zeros up to 76 bytes
        blob.resize(76 * 2, '0');
        job.setBlob(blob.c_str());

        std::string target_str = arr[3].GetString();
        target_str.resize(16, '0');
        const uint64_t target = strtoull(target_str.c_str(), nullptr, 16);
        job.setDiff(job.toDiff(target));

        job.setHeight(arr[5].GetUint64());

        m_listener->onJobReceived(this, job, params);
    }
}


bool EthStratumClient::disconnect()
{
    m_authorized = false;

    return Client::disconnect();
}


void EthStratumClient::OnSubscribeResponse(const rapidjson::Value& result, bool success, uint64_t elapsed)
{
    if (!success) {
        return;
    }

    if (!result.IsArray()) {
        LOG_ERR("Invalid mining.subscribe response: result is not an array");
        return;
    }

    if (result.GetArray().Size() <= 1) {
        LOG_ERR("Invalid mining.subscribe response: result array is too short");
        return;
    }

    auto& extra_nonce = result.GetArray()[1];
    if (!extra_nonce.IsString()) {
        LOG_ERR("Invalid mining.subscribe response: extra nonce is not a string");
        return;
    }

    const char* s = extra_nonce.GetString();
    size_t len = extra_nonce.GetStringLength();

    // Skip "0x"
    if ((len >= 2) && (s[0] == '0') && (s[1] == 'x')) {
        s += 2;
        len -= 2;
    }

    if (len & 1) {
        LOG_ERR("Invalid mining.subscribe response: extra nonce has an odd number of hex chars");
        return;
    }

    if (len > 8) {
        LOG_ERR("Invalid mining.subscribe response: extra nonce is too long");
        return;
    }

    std::string extra_nonce_str(s);
    extra_nonce_str.resize(16, '0');

    m_extraNonce = std::stoull(extra_nonce_str, nullptr, 16);
    LOG_DEBUG("Extra nonce set to %s", s);

    using namespace rapidjson;

    Document doc(kObjectType);
    auto& allocator = doc.GetAllocator();

    Value params(kArrayType);

    const char* user = m_pool.user().data();
    const char* pass = m_pool.password().data();

    params.PushBack(StringRef(user), allocator);
    params.PushBack(StringRef(pass), allocator);

    JsonRequest::create(doc, m_sequence, "mining.authorize", params);

    send(doc, [this](const rapidjson::Value& result, bool success, uint64_t elapsed) { OnAuthorizeResponse(result, success, elapsed); });
}

void EthStratumClient::OnAuthorizeResponse(const rapidjson::Value& result, bool success, uint64_t elapsed)
{
    if (!success) {
        disconnect();
        return;
    }

    if (!result.IsBool()) {
        LOG_ERR("Invalid mining.authorize response: result is not a boolean");

        disconnect();
        return;
    }

    if (!result.GetBool()) {
        LOG_ERR("Login failed");
        disconnect();
        return;
    }

    LOG_DEBUG("Login succeeded");

    m_authorized = true;
}

}
