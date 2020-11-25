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
#include "backend/common/benchmark/BenchState.h"
#include "backend/common/interfaces/IBackend.h"
#include "backend/cpu/Cpu.h"
#include "base/io/json/Json.h"
#include "base/io/log/Log.h"
#include "base/io/log/Tags.h"
#include "base/kernel/interfaces/IClientListener.h"
#include "base/net/dns/Dns.h"
#include "base/net/http/Fetch.h"
#include "base/net/http/HttpData.h"
#include "base/net/http/HttpListener.h"
#include "base/net/stratum/benchmark/BenchConfig.h"
#include "version.h"


xmrig::BenchClient::BenchClient(const std::shared_ptr<BenchConfig> &benchmark, IClientListener* listener) :
    m_listener(listener),
    m_benchmark(benchmark),
    m_hash(benchmark->hash())
{
    std::vector<char> blob(112 * 2 + 1, '0');
    blob.back() = '\0';

    m_job.setBlob(blob.data());
    m_job.setAlgorithm(m_benchmark->algorithm());
    m_job.setDiff(std::numeric_limits<uint64_t>::max());
    m_job.setHeight(1);
    m_job.setId("00000000");

    blob[Job::kMaxSeedSize * 2] = '\0';
    m_job.setSeedHash(blob.data());

    BenchState::init(this, m_benchmark->size());

#   ifdef XMRIG_FEATURE_HTTP
    if (m_benchmark->isSubmit()) {
        m_mode = ONLINE_BENCH;

        return;
    }

    if (!m_benchmark->id().isEmpty()) {
        m_job.setId(m_benchmark->id());
        m_token = m_benchmark->token();
        m_mode  = ONLINE_VERIFY;

        return;
    }
#   endif

    if (m_hash && setSeed(m_benchmark->seed())) {
        m_mode = STATIC_VERIFY;

        return;
    }

    m_job.setBenchSize(m_benchmark->size());

}


xmrig::BenchClient::~BenchClient()
{
    BenchState::destroy();
}


const char *xmrig::BenchClient::tag() const
{
    return Tags::bench();
}


void xmrig::BenchClient::connect()
{
#   ifdef XMRIG_FEATURE_HTTP
    if (m_mode == ONLINE_BENCH || m_mode == ONLINE_VERIFY) {
        return resolve();
    }
#   endif

    start();
}


void xmrig::BenchClient::setPool(const Pool &pool)
{
    m_pool = pool;
}


void xmrig::BenchClient::onBenchDone(uint64_t result, uint64_t diff, uint64_t ts)
{
    m_result    = result;
    m_diff      = diff;
    m_doneTime  = ts;

#   ifdef XMRIG_FEATURE_HTTP
    if (!m_token.isEmpty()) {
        send(DONE_BENCH);
    }
#   endif

    const uint64_t ref = referenceHash();
    const char *color  = ref ? ((result == ref) ? GREEN_BOLD_S : RED_BOLD_S) : BLACK_BOLD_S;

    LOG_NOTICE("%s " WHITE_BOLD("benchmark finished in ") CYAN_BOLD("%.3f seconds") WHITE_BOLD_S " hash sum = " CLEAR "%s%016" PRIX64 CLEAR, tag(), static_cast<double>(ts - m_readyTime) / 1000.0, color, result);

    if (m_token.isEmpty()) {
        printExit();
    }
}


void xmrig::BenchClient::onBenchReady(uint64_t ts, uint32_t threads, const IBackend *backend)
{
    m_readyTime = ts;
    m_threads   = threads;
    m_backend   = backend;

#   ifdef XMRIG_FEATURE_HTTP
    if (m_mode == ONLINE_BENCH) {
        send(CREATE_BENCH);
    }
#   endif
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

    switch (m_request) {
    case GET_BENCH:
        return onGetReply(doc);

    case CREATE_BENCH:
        return onCreateReply(doc);

    case DONE_BENCH:
        return onDoneReply(doc);

    default:
        break;
    }
#   endif
}


void xmrig::BenchClient::onResolved(const Dns &dns, int status)
{
#   ifdef XMRIG_FEATURE_HTTP
    assert(!m_httpListener);

    if (status < 0) {
        return setError(dns.error(), "DNS error");
    }

    m_ip            = dns.get().ip();
    m_httpListener  = std::make_shared<HttpListener>(this, tag());

    if (m_mode == ONLINE_BENCH) {
        start();
    }
    else {
        send(GET_BENCH);
    }
#   endif
}


bool xmrig::BenchClient::setSeed(const char *seed)
{
    if (!seed) {
        return false;
    }

    size_t size = strlen(seed);
    if (size % 2 != 0) {
        return false;
    }

    size /= 2;
    if (size < 4 || size >= m_job.size()) {
        return false;
    }

    if (!Buffer::fromHex(seed, size * 2, m_job.blob())) {
        return false;
    }

    m_job.setBenchSize(BenchState::size());

    LOG_NOTICE("%s " WHITE_BOLD("seed ") BLACK_BOLD("%s"), tag(), seed);

    return true;
}


uint64_t xmrig::BenchClient::referenceHash() const
{
    if (m_hash || m_mode == ONLINE_BENCH) {
        return m_hash;
    }

    return BenchState::referenceHash(m_job.algorithm(), BenchState::size(), m_threads);
}


void xmrig::BenchClient::printExit()
{
    LOG_INFO("%s " WHITE_BOLD("press ") MAGENTA_BOLD("Ctrl+C") WHITE_BOLD(" to exit"), tag());
}


void xmrig::BenchClient::start()
{
    const uint32_t size = BenchState::size();

    LOG_NOTICE("%s " MAGENTA_BOLD("start benchmark ") "hashes " CYAN_BOLD("%u%s") " algo " WHITE_BOLD("%s"),
               tag(),
               size < 1000000 ? size / 1000 : size / 1000000,
               size < 1000000 ? "K" : "M",
               m_job.algorithm().shortName());

    m_listener->onLoginSuccess(this);
    m_listener->onJobReceived(this, m_job, rapidjson::Value());
}



#ifdef XMRIG_FEATURE_HTTP
void xmrig::BenchClient::onCreateReply(const rapidjson::Value &value)
{
    m_startTime = Chrono::steadyMSecs();
    m_token     = Json::getString(value, BenchConfig::kToken);

    m_job.setId(Json::getString(value, BenchConfig::kId));
    setSeed(Json::getString(value, BenchConfig::kSeed));

    m_listener->onJobReceived(this, m_job, rapidjson::Value());

    send(START_BENCH);
}


void xmrig::BenchClient::onDoneReply(const rapidjson::Value &)
{
    LOG_NOTICE("%s " WHITE_BOLD("benchmark submitted ") CYAN_BOLD("https://xmrig.com/benchmark/%s"), tag(), m_job.id().data());
    printExit();
}


void xmrig::BenchClient::onGetReply(const rapidjson::Value &value)
{
    const char *hash = Json::getString(value, BenchConfig::kHash);
    if (hash) {
        m_hash = strtoull(hash, nullptr, 16);
    }

    BenchState::setSize(Json::getUint(value, BenchConfig::kSize));

    m_job.setAlgorithm(Json::getString(value, BenchConfig::kAlgo));
    setSeed(Json::getString(value, BenchConfig::kSeed));

    start();
}


void xmrig::BenchClient::resolve()
{
    m_dns = std::make_shared<Dns>(this);

    if (!m_dns->resolve(BenchConfig::kApiHost)) {
        setError(m_dns->error(), "getaddrinfo error");
    }
}


void xmrig::BenchClient::send(Request request)
{
    using namespace rapidjson;

    Document doc(kObjectType);
    auto &allocator = doc.GetAllocator();
    m_request       = request;

    switch (m_request) {
    case GET_BENCH:
        {
            FetchRequest req(HTTP_GET, m_ip, BenchConfig::kApiPort, fmt::format("/1/benchmark/{}", m_job.id()).c_str(), BenchConfig::kApiTLS, true);
            fetch(std::move(req), m_httpListener);
        }
        break;

    case CREATE_BENCH:
        {
            doc.AddMember(StringRef(BenchConfig::kSize),    m_benchmark->size(), allocator);
            doc.AddMember(StringRef(BenchConfig::kAlgo),    m_benchmark->algorithm().toJSON(), allocator);
            doc.AddMember("version",                        APP_VERSION, allocator);
            doc.AddMember("threads",                        m_threads, allocator);
            doc.AddMember("steady_ready_ts",                m_readyTime, allocator);
            doc.AddMember("cpu",                            Cpu::toJSON(doc), allocator);

            FetchRequest req(HTTP_POST, m_ip, BenchConfig::kApiPort, "/1/benchmark", doc, BenchConfig::kApiTLS, true);
            fetch(std::move(req), m_httpListener);
        }
        break;

    case START_BENCH:
        doc.AddMember("steady_start_ts",    m_startTime, allocator);
        update(doc);
        break;

    case DONE_BENCH:
        doc.AddMember("steady_done_ts",     m_doneTime, allocator);
        doc.AddMember("hash",               Value(fmt::format("{:016X}", m_result).c_str(), allocator), allocator);
        doc.AddMember("diff",               m_diff, allocator);
        doc.AddMember("backend",            m_backend->toJSON(doc), allocator);
        update(doc);
        break;

    case NO_REQUEST:
        break;
    }
}


void xmrig::BenchClient::setError(const char *message, const char *label)
{
    LOG_ERR("%s " RED("%s: ") RED_BOLD("\"%s\""), tag(), label ? label : "benchmark failed", message);
    printExit();

    BenchState::destroy();
}


void xmrig::BenchClient::update(const rapidjson::Value &body)
{
    assert(!m_token.isEmpty());

    FetchRequest req(HTTP_PATCH, m_ip, BenchConfig::kApiPort, fmt::format("/1/benchmark/{}", m_job.id()).c_str(), body, BenchConfig::kApiTLS, true);
    req.headers.insert({ "Authorization", fmt::format("Bearer {}", m_token)});

    fetch(std::move(req), m_httpListener);
}
#endif
