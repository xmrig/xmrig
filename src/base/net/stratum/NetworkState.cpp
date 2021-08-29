/* XMRig
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#include "base/net/stratum/NetworkState.h"
#include "3rdparty/rapidjson/document.h"
#include "base/io/log/Log.h"
#include "base/kernel/interfaces/IClient.h"
#include "base/kernel/interfaces/IStrategy.h"
#include "base/net/stratum/Job.h"
#include "base/net/stratum/Pool.h"
#include "base/net/stratum/SubmitResult.h"
#include "base/tools/Chrono.h"


#include <algorithm>
#include <cstdio>
#include <cstring>
#include <uv.h>



namespace xmrig {


inline static void printCount(uint64_t accepted, uint64_t rejected)
{
    float percent   = 100.0;
    int color       = 2;

    if (!accepted) {
        percent     = 0.0;
        color       = 1;
    }
    else if (rejected) {
        percent     = static_cast<float>(accepted) / (accepted + rejected) * 100.0;
        color       = 3;
    }

    Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-17s") CSI "1;3%dm%" PRIu64 CLEAR CSI "0;3%dm (%1.1f%%)", "accepted", color, accepted, color, percent);

    if (rejected) {
        Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-17s") RED_BOLD("%" PRIu64), "rejected", rejected);
    }
}


inline static void printHashes(uint64_t accepted, uint64_t hashes)
{
    Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-17s") CYAN_BOLD("%" PRIu64) " avg " CYAN("%1.0f"),
               "pool-side hashes", hashes, static_cast<double>(hashes) / accepted);
}


inline static void printAvgTime(uint64_t time)
{
    Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-17s") CSI "1;3%dm%1.1fs", "avg result time", (time < 10000 ? 3 : 2), time / 1000.0);
}


static void printDiff(uint64_t diff)
{
    Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-17s") CYAN_BOLD("%s"), "difficulty", NetworkState::humanDiff(diff).c_str());
}


inline static void printDiff(size_t i, uint64_t diff, uint64_t hashes)
{
    if (!diff) {
        return;
    }

    const double effort = static_cast<double>(hashes) / diff * 100.0;
    const double target = (i + 1) * 100.0;
    const int color     = effort > (target + 100.0) ? 1 : (effort > target ? 3 : 2);

    Log::print("%3zu | %10s | " CSI "0;3%dm%8.2f" CLEAR " |", i + 1, NetworkState::humanDiff(diff).c_str(), color, effort);
}


inline static void printLatency(uint32_t latency)
{
    if (!latency) {
        return;
    }

    const int color = latency < 100 ? 2 : (latency > 500 ? 1 : 3);

    Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-17s") CSI "1;3%dm%ums", "ping time", color, latency);
}


} // namespace xmrig



xmrig::NetworkState::NetworkState(IStrategyListener *listener) : StrategyProxy(listener)
{
}


#ifdef XMRIG_FEATURE_API
rapidjson::Value xmrig::NetworkState::getConnection(rapidjson::Document &doc, int version) const
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    Value connection(kObjectType);
    connection.AddMember("pool",            StringRef(m_pool), allocator);
    connection.AddMember("ip",              m_ip.toJSON(), allocator);
    connection.AddMember("uptime",          connectionTime() / 1000, allocator);
    connection.AddMember("uptime_ms",       connectionTime(), allocator);
    connection.AddMember("ping",            latency(), allocator);
    connection.AddMember("failures",        m_failures, allocator);
    connection.AddMember("tls",             m_tls.toJSON(), allocator);
    connection.AddMember("tls-fingerprint", m_fingerprint.toJSON(), allocator);

    connection.AddMember("algo",            m_algorithm.toJSON(), allocator);
    connection.AddMember("diff",            m_diff, allocator);
    connection.AddMember("accepted",        m_accepted, allocator);
    connection.AddMember("rejected",        m_rejected, allocator);
    connection.AddMember("avg_time",        avgTime() / 1000, allocator);
    connection.AddMember("avg_time_ms",     avgTime(), allocator);
    connection.AddMember("hashes_total",    m_hashes, allocator);

    if (version == 1) {
        connection.AddMember("error_log", Value(kArrayType), allocator);
    }

    return connection;
}


rapidjson::Value xmrig::NetworkState::getResults(rapidjson::Document &doc, int version) const
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    Value results(kObjectType);

    results.AddMember("diff_current",  m_diff, allocator);
    results.AddMember("shares_good",   m_accepted, allocator);
    results.AddMember("shares_total",  m_accepted + m_rejected, allocator);
    results.AddMember("avg_time",      avgTime() / 1000, allocator);
    results.AddMember("avg_time_ms",   avgTime(), allocator);
    results.AddMember("hashes_total",  m_hashes, allocator);

    Value best(kArrayType);
    best.Reserve(m_topDiff.size(), allocator);

    for (uint64_t i : m_topDiff) {
        best.PushBack(i, allocator);
    }

    results.AddMember("best", best, allocator);

    if (version == 1) {
        results.AddMember("error_log", Value(kArrayType), allocator);
    }

    return results;
}
#endif


void xmrig::NetworkState::printConnection() const
{
    if (!m_active) {
        LOG_NOTICE(YELLOW_BOLD_S "no active connection");

        return;
    }

    Log::print(MAGENTA_BOLD_S " - CONNECTION");
    Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-17s") CYAN_BOLD("%s ") BLACK_BOLD("(%s) ") GREEN_BOLD("%s"),
               "pool address", m_pool, m_ip.data(), m_tls.isNull() ? "" : m_tls.data());

    Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-17s") WHITE_BOLD("%s"), "algorithm", m_algorithm.name());
    printDiff(m_diff);
    printLatency(latency());
    Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-17s") CYAN_BOLD("%" PRIu64 "s"), "connection time", connectionTime() / 1000);
}


void xmrig::NetworkState::printResults() const
{
    if (!m_hashes) {
        LOG_NOTICE(YELLOW_BOLD_S "no results yet");

        return;
    }

    Log::print(MAGENTA_BOLD_S " - RESULTS");

    printCount(m_accepted, m_rejected);
    printHashes(m_accepted, m_hashes);
    printDiff(m_diff);

    if (m_active && !m_latency.empty()) {
        printAvgTime(avgTime());
    }

    Log::print(MAGENTA_BOLD_S " - TOP 10");
    Log::print(WHITE_BOLD_S "  # | DIFFICULTY | EFFORT %% |");

    for (size_t i = 0; i < m_topDiff.size(); ++i) {
        printDiff(i, m_topDiff[i], m_hashes);
    }
}


const char *xmrig::NetworkState::scaleDiff(uint64_t &diff)
{
    if (diff >= 100000000000) {
        diff /= 1000000000;

        return "G";
    }

    if (diff >= 100000000) {
        diff /= 1000000;

        return "M";
    }

    if (diff >= 1000000) {
        diff /= 1000;

        return "K";
    }

    return "";
}


std::string xmrig::NetworkState::humanDiff(uint64_t diff)
{
    const char *scale = scaleDiff(diff);

    return std::to_string(diff) + scale;
}


void xmrig::NetworkState::onActive(IStrategy *strategy, IClient *client)
{
    snprintf(m_pool, sizeof(m_pool) - 1, "%s:%d", client->pool().host().data(), client->pool().port());

    m_ip             = client->ip();
    m_tls            = client->tlsVersion();
    m_fingerprint    = client->tlsFingerprint();
    m_active         = true;
    m_connectionTime = Chrono::steadyMSecs();

    StrategyProxy::onActive(strategy, client);
}


void xmrig::NetworkState::onJob(IStrategy *strategy, IClient *client, const Job &job, const rapidjson::Value &params)
{
    m_algorithm = job.algorithm();
    m_diff      = job.diff();

    StrategyProxy::onJob(strategy, client, job, params);
}


void xmrig::NetworkState::onPause(IStrategy *strategy)
{
    if (!strategy->isActive()) {
        stop();
    }

    StrategyProxy::onPause(strategy);
}


void xmrig::NetworkState::onResultAccepted(IStrategy *strategy, IClient *client, const SubmitResult &result, const char *error)
{
    add(result, error);

    StrategyProxy::onResultAccepted(strategy, client, result, error);
}


uint32_t xmrig::NetworkState::latency() const
{
    const size_t calls = m_latency.size();
    if (calls == 0) {
        return 0;
    }

    auto v = m_latency;
    std::nth_element(v.begin(), v.begin() + calls / 2, v.end());

    return v[calls / 2];
}


uint64_t xmrig::NetworkState::avgTime() const
{
    if (m_latency.empty()) {
        return 0;
    }

    return connectionTime() / m_latency.size();
}


uint64_t xmrig::NetworkState::connectionTime() const
{
    return m_active ? ((Chrono::steadyMSecs() - m_connectionTime)) : 0;
}


void xmrig::NetworkState::add(const SubmitResult &result, const char *error)
{
    if (error) {
        m_rejected++;
        return;
    }

    m_accepted++;
    m_hashes += result.diff;

    const size_t ln = m_topDiff.size() - 1;
    if (result.actualDiff > m_topDiff[ln]) {
        m_topDiff[ln] = result.actualDiff;
        std::sort(m_topDiff.rbegin(), m_topDiff.rend());
    }

    m_latency.push_back(result.elapsed > 0xFFFF ? 0xFFFF : static_cast<uint16_t>(result.elapsed));
}


void xmrig::NetworkState::stop()
{
    m_active      = false;
    m_diff        = 0;
    m_ip          = nullptr;
    m_tls         = nullptr;
    m_fingerprint = nullptr;

    m_failures++;
    m_latency.clear();
}
