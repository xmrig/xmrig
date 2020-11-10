/* XMRig
 * Copyright (c) 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#include "base/net/stratum/benchmark/BenchClient.h"
#include "3rdparty/fmt/core.h"
#include "3rdparty/rapidjson/document.h"
#include "backend/cpu/Cpu.h"
#include "base/io/json/Json.h"
#include "base/io/log/Log.h"
#include "base/io/log/Tags.h"
#include "base/kernel/interfaces/IClientListener.h"
#include "base/net/http/Fetch.h"
#include "base/net/http/HttpData.h"
#include "base/net/http/HttpListener.h"
#include "base/net/stratum/benchmark/BenchConfig.h"
#include "version.h"


xmrig::BenchClient::BenchClient(const std::shared_ptr<BenchConfig> &benchmark, IClientListener* listener) :
    m_listener(listener),
    m_benchmark(benchmark)
{
    std::vector<char> blob(112 * 2 + 1, '0');
    blob.back() = '\0';

    m_job.setBlob(blob.data());
    m_job.setAlgorithm(m_benchmark->algorithm());
    m_job.setDiff(std::numeric_limits<uint64_t>::max());
    m_job.setHeight(1);
    m_job.setBenchSize(m_benchmark->size());
    m_job.setBenchHash(m_benchmark->hash());

#   ifdef XMRIG_FEATURE_HTTP
    if (m_benchmark->isSubmit()) {
        m_mode = ONLINE_BENCH;

        return;
    }

    if (!m_benchmark->id().isEmpty()) {
        m_job.setId(m_benchmark->id());
        m_job.setBenchToken(m_benchmark->token());
        m_mode = ONLINE_VERIFY;

        return;
    }
#   endif

    m_job.setId("00000000");

    if (m_job.benchHash() && m_job.setSeedHash(m_benchmark->seed())) {
        m_mode = STATIC_VERIFY;

        return;
    }

    blob[Job::kMaxSeedSize * 2] = '\0';
    m_job.setSeedHash(blob.data());
}


void xmrig::BenchClient::connect()
{
#   ifdef XMRIG_FEATURE_HTTP
    switch (m_mode) {
    case STATIC_BENCH:
    case STATIC_VERIFY:
        return start();

    case ONLINE_BENCH:
        return createBench();

    case ONLINE_VERIFY:
        return getBench();
    }
#   else
    start();
#   endif
}


void xmrig::BenchClient::setPool(const Pool &pool)
{
    m_pool = pool;
}


void xmrig::BenchClient::onHttpData(const HttpData &data)
{
#   ifdef XMRIG_FEATURE_HTTP
    rapidjson::Document doc;

    try {
        doc = data.json();
    } catch (const std::exception &ex) {
        return setError(ex.what());
    }

    if (data.status != 200) {
        return setError(data.statusName());
    }

    if (m_mode == ONLINE_BENCH) {
        startBench(doc);
    }
    else {
        startVerify(doc);
    }
#   endif
}


void xmrig::BenchClient::start()
{
    m_listener->onLoginSuccess(this);
    m_listener->onJobReceived(this, m_job, rapidjson::Value());
}



#ifdef XMRIG_FEATURE_HTTP
void xmrig::BenchClient::createBench()
{
    createHttpListener();

    using namespace rapidjson;

    Document doc(kObjectType);
    auto &allocator = doc.GetAllocator();

    doc.AddMember(StringRef(BenchConfig::kSize), m_benchmark->size(), allocator);
    doc.AddMember(StringRef(BenchConfig::kAlgo), m_benchmark->algorithm().toJSON(), allocator);
    doc.AddMember("version",                     APP_VERSION, allocator);
    doc.AddMember("cpu",                         Cpu::toJSON(doc), allocator);

    FetchRequest req(HTTP_POST, BenchConfig::kApiHost, BenchConfig::kApiPort, "/1/benchmark", doc, BenchConfig::kApiTLS, true);
    fetch(std::move(req), m_httpListener);
}


void xmrig::BenchClient::createHttpListener()
{
    if (!m_httpListener) {
        m_httpListener = std::make_shared<HttpListener>(this, Tags::bench());
    }
}


void xmrig::BenchClient::getBench()
{
    createHttpListener();

    FetchRequest req(HTTP_GET, BenchConfig::kApiHost, BenchConfig::kApiPort, fmt::format("/1/benchmark/{}", m_job.id()).c_str(), BenchConfig::kApiTLS, true);
    fetch(std::move(req), m_httpListener);
}


void xmrig::BenchClient::setError(const char *message)
{
    LOG_ERR("%s " RED("benchmark failed ") RED_BOLD("\"%s\""), Tags::bench(), message);
}


void xmrig::BenchClient::startBench(const rapidjson::Value &value)
{
    m_job.setId(Json::getString(value, BenchConfig::kId));
    m_job.setSeedHash(Json::getString(value, BenchConfig::kSeed));
    m_job.setBenchToken(Json::getString(value, BenchConfig::kToken));

    start();
}


void xmrig::BenchClient::startVerify(const rapidjson::Value &value)
{
    const char *hash = Json::getString(value, BenchConfig::kHash);
    if (hash) {
        m_job.setBenchHash(strtoull(hash, nullptr, 16));
    }

    m_job.setAlgorithm(Json::getString(value, BenchConfig::kAlgo));
    m_job.setSeedHash(Json::getString(value, BenchConfig::kSeed));
    m_job.setBenchSize(Json::getUint(value, BenchConfig::kSize));

    start();
}
#endif
