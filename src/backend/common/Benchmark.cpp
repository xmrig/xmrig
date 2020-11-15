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


#include "backend/common/Benchmark.h"
#include "3rdparty/fmt/core.h"
#include "backend/common/interfaces/IBackend.h"
#include "backend/common/interfaces/IWorker.h"
#include "base/io/log/Log.h"
#include "base/io/log/Tags.h"
#include "base/net/http/Fetch.h"
#include "base/net/http/HttpData.h"
#include "base/net/http/HttpListener.h"
#include "base/net/stratum/benchmark/BenchConfig.h"
#include "base/net/stratum/Job.h"
#include "base/tools/Chrono.h"


#include <algorithm>


namespace xmrig {


static const std::map<int, std::map<uint32_t, uint64_t> > hashCheck = {
    { Algorithm::RX_0, {
        {   250000U, 0x7D6054757BB08A63ULL },
        {   500000U, 0x96607546DE1F5ECCULL },
        {  1000000U, 0x898B6E0431C28A6BULL },
        {  2000000U, 0xEE9468F8B40926BCULL },
        {  3000000U, 0xC2BC5D11724813C0ULL },
        {  4000000U, 0x3A2C7B285B87F941ULL },
        {  5000000U, 0x3B5BD2C3A16B450EULL },
        {  6000000U, 0x5CD0602F20C5C7C4ULL },
        {  7000000U, 0x101DE939474B6812ULL },
        {  8000000U, 0x52B765A1B156C6ECULL },
        {  9000000U, 0x323935102AB6B45CULL },
        { 10000000U, 0xB5231262E2792B26ULL }
    }},
    { Algorithm::RX_WOW, {
        {   250000U, 0xC7F712C9603E2603ULL },
        {   500000U, 0x21A0E5AAE6DA7D8DULL },
        {  1000000U, 0x0F3E5400B39EA96AULL },
        {  2000000U, 0x85944CCFA2752D1FULL },
        {  3000000U, 0x64AFFCAE991811BAULL },
        {  4000000U, 0x3E4D0B836D3B13BAULL },
        {  5000000U, 0xEB7417D621271166ULL },
        {  6000000U, 0x97FFE10C0949FFA5ULL },
        {  7000000U, 0x84CAC0F8879A4BA1ULL },
        {  8000000U, 0xA1B79F031DA2459FULL },
        {  9000000U, 0x9B65226DA873E65DULL },
        { 10000000U, 0x0F9E00C5A511C200ULL }
    }}
};


static const std::map<int, std::map<uint32_t, uint64_t> > hashCheck1T = {
    { Algorithm::RX_0, {
        {   250000U, 0x90A15B799486F3EBULL },
        {   500000U, 0xAA83118FEE570F9AULL },
        {  1000000U, 0x3DF47B0A427C93D9ULL },
        {  2000000U, 0xED4D639B0AEB85C6ULL },
        {  3000000U, 0x2D4F9B4275A713C3ULL },
        {  4000000U, 0xA9EBE4888377F8D3ULL },
        {  5000000U, 0xB92F81851E180454ULL },
        {  6000000U, 0xFB9F98F63C2F1B7DULL },
        {  7000000U, 0x2CC3D7A779D5AB35ULL },
        {  8000000U, 0x2EEF833EA462F4B1ULL },
        {  9000000U, 0xC6D39EF59213A07CULL },
        { 10000000U, 0x95E6BAE68DD779CDULL }
    }},
    { Algorithm::RX_WOW, {
        {   250000U, 0x7B409F096C863207ULL },
        {   500000U, 0x70B7B80D15654216ULL },
        {  1000000U, 0x31301CC550306A59ULL },
        {  2000000U, 0x92F65E9E31116361ULL },
        {  3000000U, 0x7FE8DF6F43BA5285ULL },
        {  4000000U, 0xD6CDA54FE4D9BBF7ULL },
        {  5000000U, 0x73AF673E1A38E2B4ULL },
        {  6000000U, 0x81FDC5C4B45D84E4ULL },
        {  7000000U, 0xAA08CA57666DC874ULL },
        {  8000000U, 0x9DCEFB833FC875BCULL },
        {  9000000U, 0x862F051352CFCA1FULL },
        { 10000000U, 0xC403F220189E8430ULL }
    }}
};


} // namespace xmrig


xmrig::Benchmark::Benchmark(const Job &job, size_t workers, const IBackend *backend) :
    m_algo(job.algorithm()),
    m_backend(backend),
    m_workers(workers),
    m_id(job.id()),
    m_token(job.benchToken()),
    m_end(job.benchSize()),
    m_hash(job.benchHash())
{
#   ifdef XMRIG_FEATURE_HTTP
    if (!m_token.isEmpty()) {
        m_httpListener = std::make_shared<HttpListener>(this, Tags::bench());
    }
#   endif
}


bool xmrig::Benchmark::finish(uint64_t totalHashCount)
{
    m_reset   = true;
    m_current = totalHashCount;

    if (m_done < m_workers) {
        return false;
    }

    const double dt    = static_cast<double>(m_doneTime - m_startTime) / 1000.0;
    uint64_t checkData = referenceHash();
    const char *color  = checkData ? ((m_data == checkData) ? GREEN_BOLD_S : RED_BOLD_S) : BLACK_BOLD_S;

    LOG_NOTICE("%s " WHITE_BOLD("benchmark finished in ") CYAN_BOLD("%.3f seconds") WHITE_BOLD_S " hash sum = " CLEAR "%s%016" PRIX64 CLEAR, Tags::bench(), dt, color, m_data);

#   ifdef XMRIG_FEATURE_HTTP
    if (!m_token.isEmpty()) {
        using namespace rapidjson;

        Document doc(kObjectType);
        auto &allocator = doc.GetAllocator();

        doc.AddMember("steady_done_ts",                 m_doneTime, allocator);
        doc.AddMember(StringRef(BenchConfig::kHash),    Value(fmt::format("{:016X}", m_data).c_str(), allocator), allocator);
        doc.AddMember("backend",                        m_backend->toJSON(doc), allocator);

        send(doc);
    }
    else
#   endif
    {
        printExit();
    }

    return true;
}


void xmrig::Benchmark::start()
{
    m_startTime = Chrono::steadyMSecs();

#   ifdef XMRIG_FEATURE_HTTP
    if (!m_token.isEmpty()) {
        using namespace rapidjson;

        Document doc(kObjectType);
        doc.AddMember("steady_start_ts", m_startTime, doc.GetAllocator());

        send(doc);
    }
#   endif
}


void xmrig::Benchmark::printProgress() const
{
    if (!m_startTime || !m_current) {
        return;
    }

    const double dt      = static_cast<double>(Chrono::steadyMSecs() - m_startTime) / 1000.0;
    const double percent = static_cast<double>(m_current) / m_end * 100.0;

    LOG_NOTICE("%s " MAGENTA_BOLD("%5.2f%% ") CYAN_BOLD("%" PRIu64) CYAN("/%" PRIu64) BLACK_BOLD(" (%.3fs)"), Tags::bench(), percent, m_current, m_end, dt);
}


void xmrig::Benchmark::tick(IWorker *worker)
{
    if (m_reset) {
        m_data  = 0;
        m_done  = 0;
        m_reset = false;
    }

    const uint64_t doneTime = worker->benchDoneTime();
    if (!doneTime) {
        return;
    }

    ++m_done;
    m_data ^= worker->benchData();
    m_doneTime = std::max(doneTime, m_doneTime);
}


void xmrig::Benchmark::onHttpData(const HttpData &data)
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

    if (m_doneTime) {
        LOG_NOTICE("%s " WHITE_BOLD("benchmark submitted ") CYAN_BOLD("https://xmrig.com/benchmark/%s"), Tags::bench(), m_id.data());
        printExit();
    }
#   endif
}


uint64_t xmrig::Benchmark::referenceHash() const
{
    if (m_hash) {
        return m_hash;
    }

#   ifdef XMRIG_FEATURE_HTTP
    if (!m_token.isEmpty()) {
        return 0;
    }
#   endif

    uint64_t hash = 0;

    try {
        const auto& h = (m_workers == 1) ? hashCheck1T : hashCheck;
        hash = h.at(m_algo).at(m_end);
    } catch (const std::exception &ex) {}

    return hash;
}


void xmrig::Benchmark::printExit()
{
    LOG_INFO("%s " WHITE_BOLD("press ") MAGENTA_BOLD("Ctrl+C") WHITE_BOLD(" to exit"), Tags::bench());
}


#ifdef XMRIG_FEATURE_HTTP
void xmrig::Benchmark::send(const rapidjson::Value &body)
{
    FetchRequest req(HTTP_PATCH, BenchConfig::kApiHost, BenchConfig::kApiPort, fmt::format("/1/benchmark/{}", m_id).c_str(), body, BenchConfig::kApiTLS, true);
    req.headers.insert({ "Authorization", fmt::format("Bearer {}", m_token)});

    fetch(std::move(req), m_httpListener);
}


void xmrig::Benchmark::setError(const char *message)
{
    LOG_ERR("%s " RED("benchmark failed ") RED_BOLD("\"%s\""), Tags::bench(), message);
}
#endif
