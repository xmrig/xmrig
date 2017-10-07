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

#include <cmath>
#include <string.h>
#include <uv.h>

#if _WIN32
#   include "winsock2.h"
#else
#   include "unistd.h"
#endif


#include "api/ApiState.h"
#include "Cpu.h"
#include "Mem.h"
#include "net/Job.h"
#include "Options.h"
#include "Platform.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/prettywriter.h"
#include "version.h"
#include "workers/Hashrate.h"


extern "C"
{
#include "crypto/c_keccak.h"
}


static inline double normalize(double d)
{
    if (!std::isnormal(d)) {
        return 0.0;
    }

    return std::floor(d * 100.0) / 100.0;
}


ApiState::ApiState()
{
    m_threads  = Options::i()->threads();
    m_hashrate = new double[m_threads * 3]();

    memset(m_totalHashrate, 0, sizeof(m_totalHashrate));
    memset(m_workerId, 0, sizeof(m_workerId));

    if (Options::i()->apiWorkerId()) {
        strncpy(m_workerId, Options::i()->apiWorkerId(), sizeof(m_workerId) - 1);
    }
    else {
        gethostname(m_workerId, sizeof(m_workerId) - 1);
    }

    genId();
}


ApiState::~ApiState()
{
    delete [] m_hashrate;
}


char *ApiState::get(const char *url, int *status) const
{
    rapidjson::Document doc;
    doc.SetObject();

    getIdentify(doc);
    getMiner(doc);
    getHashrate(doc);
    getResults(doc);
    getConnection(doc);

    return finalize(doc);
}


void ApiState::tick(const Hashrate *hashrate)
{
    for (int i = 0; i < m_threads; ++i) {
        m_hashrate[i * 3]     = hashrate->calc((size_t) i, Hashrate::ShortInterval);
        m_hashrate[i * 3 + 1] = hashrate->calc((size_t) i, Hashrate::MediumInterval);
        m_hashrate[i * 3 + 2] = hashrate->calc((size_t) i, Hashrate::LargeInterval);
    }

    m_totalHashrate[0] = hashrate->calc(Hashrate::ShortInterval);
    m_totalHashrate[1] = hashrate->calc(Hashrate::MediumInterval);
    m_totalHashrate[2] = hashrate->calc(Hashrate::LargeInterval);
    m_highestHashrate  = hashrate->highest();
}


void ApiState::tick(const NetworkState &network)
{
    m_network = network;
}


char *ApiState::finalize(rapidjson::Document &doc) const
{
    rapidjson::StringBuffer buffer(0, 4096);
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
    writer.SetMaxDecimalPlaces(10);
    doc.Accept(writer);

    return strdup(buffer.GetString());
}


void ApiState::genId()
{
    memset(m_id, 0, sizeof(m_id));

    uv_interface_address_t *interfaces;
    int count = 0;

    if (uv_interface_addresses(&interfaces, &count) < 0) {
        return;
    }

    for (int i = 0; i < count; i++) {
        if (!interfaces[i].is_internal && interfaces[i].address.address4.sin_family == AF_INET) {
            uint8_t hash[200];
            const size_t addrSize = sizeof(interfaces[i].phys_addr);
            const size_t inSize   = strlen(APP_KIND) + addrSize;

            uint8_t *input = new uint8_t[inSize]();
            memcpy(input, interfaces[i].phys_addr, addrSize);
            memcpy(input + addrSize, APP_KIND, strlen(APP_KIND));

            keccak(input, static_cast<int>(inSize), hash, sizeof(hash));
            Job::toHex(hash, 8, m_id);

            delete [] input;
            break;
        }
    }

    uv_free_interface_addresses(interfaces, count);
}


void ApiState::getConnection(rapidjson::Document &doc) const
{
    auto &allocator = doc.GetAllocator();

    rapidjson::Value connection(rapidjson::kObjectType);
    connection.AddMember("pool",      rapidjson::StringRef(m_network.pool), allocator);
    connection.AddMember("uptime",    m_network.connectionTime(), allocator);
    connection.AddMember("ping",      m_network.latency(), allocator);
    connection.AddMember("failures",  m_network.failures, allocator);
    connection.AddMember("error_log", rapidjson::Value(rapidjson::kArrayType), allocator);

    doc.AddMember("connection", connection, allocator);
}


void ApiState::getHashrate(rapidjson::Document &doc) const
{
    auto &allocator = doc.GetAllocator();

    rapidjson::Value hashrate(rapidjson::kObjectType);
    rapidjson::Value total(rapidjson::kArrayType);
    rapidjson::Value threads(rapidjson::kArrayType);

    for (int i = 0; i < 3; ++i) {
        total.PushBack(normalize(m_totalHashrate[i]), allocator);
    }

    for (int i = 0; i < m_threads * 3; i += 3) {
        rapidjson::Value thread(rapidjson::kArrayType);
        thread.PushBack(normalize(m_hashrate[i]),     allocator);
        thread.PushBack(normalize(m_hashrate[i + 1]), allocator);
        thread.PushBack(normalize(m_hashrate[i + 2]), allocator);

        threads.PushBack(thread, allocator);
    }

    hashrate.AddMember("total", total, allocator);
    hashrate.AddMember("highest", normalize(m_highestHashrate), allocator);
    hashrate.AddMember("threads", threads, allocator);
    doc.AddMember("hashrate", hashrate, allocator);
}


void ApiState::getIdentify(rapidjson::Document &doc) const
{
    doc.AddMember("id",        rapidjson::StringRef(m_id),       doc.GetAllocator());
    doc.AddMember("worker_id", rapidjson::StringRef(m_workerId), doc.GetAllocator());
}


void ApiState::getMiner(rapidjson::Document &doc) const
{
    auto &allocator = doc.GetAllocator();

    rapidjson::Value cpu(rapidjson::kObjectType);
    cpu.AddMember("brand",   rapidjson::StringRef(Cpu::brand()), allocator);
    cpu.AddMember("aes",     Cpu::hasAES(), allocator);
    cpu.AddMember("x64",     Cpu::isX64(), allocator);
    cpu.AddMember("sockets", Cpu::sockets(), allocator);

    doc.AddMember("version",      APP_VERSION, allocator);
    doc.AddMember("kind",         APP_KIND, allocator);
    doc.AddMember("ua",           rapidjson::StringRef(Platform::userAgent()), allocator);
    doc.AddMember("cpu",          cpu, allocator);
    doc.AddMember("algo",         rapidjson::StringRef(Options::i()->algoName()), allocator);
    doc.AddMember("hugepages",    Mem::isHugepagesEnabled(), allocator);
    doc.AddMember("donate_level", Options::i()->donateLevel(), allocator);
}


void ApiState::getResults(rapidjson::Document &doc) const
{
    auto &allocator = doc.GetAllocator();

    rapidjson::Value results(rapidjson::kObjectType);

    results.AddMember("diff_current",  m_network.diff, allocator);
    results.AddMember("shares_good",   m_network.accepted, allocator);
    results.AddMember("shares_total",  m_network.accepted + m_network.rejected, allocator);
    results.AddMember("avg_time",      m_network.avgTime(), allocator);
    results.AddMember("hashes_total",  m_network.total, allocator);

    rapidjson::Value best(rapidjson::kArrayType);
    for (size_t i = 0; i < m_network.topDiff.size(); ++i) {
        best.PushBack(m_network.topDiff[i], allocator);
    }

    results.AddMember("best",      best, allocator);
    results.AddMember("error_log", rapidjson::Value(rapidjson::kArrayType), allocator);

    doc.AddMember("results", results, allocator);
}
