/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2016-2018 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#include <math.h>
#include <string.h>
#include <uv.h>

#if _WIN32
#   include "winsock2.h"
#else
#   include "unistd.h"
#endif


#include "api/ApiRouter.h"
#include "common/api/HttpReply.h"
#include "common/api/HttpRequest.h"
#include "common/cpu/Cpu.h"
#include "common/crypto/keccak.h"
#include "common/net/Job.h"
#include "common/Platform.h"
#include "core/Config.h"
#include "core/Controller.h"
#include "interfaces/IThread.h"
#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"
#include "version.h"
#include "workers/Hashrate.h"
#include "workers/Workers.h"


static inline double normalize(double d)
{
    if (!isnormal(d)) {
        return 0.0;
    }

    return floor(d * 100.0) / 100.0;
}


ApiRouter::ApiRouter(xmrig::Controller *controller) :
    m_controller(controller)
{
    memset(m_workerId, 0, sizeof(m_workerId));

    setWorkerId(controller->config()->apiWorkerId());
    genId(controller->config()->apiId());
}


ApiRouter::~ApiRouter()
{
}


void ApiRouter::ApiRouter::get(const xmrig::HttpRequest &req, xmrig::HttpReply &reply) const
{
    rapidjson::Document doc;

    if (req.match("/1/config")) {
        if (req.isRestricted()) {
            reply.status = 403;
            return;
        }

        m_controller->config()->getJSON(doc);

        return finalize(reply, doc);
    }

    if (req.match("/1/threads")) {
        getThreads(doc);

        return finalize(reply, doc);
    }

    doc.SetObject();

    getIdentify(doc);
    getMiner(doc);
    getHashrate(doc);
    getResults(doc);
    getConnection(doc);

    return finalize(reply, doc);
}


void ApiRouter::exec(const xmrig::HttpRequest &req, xmrig::HttpReply &reply)
{
    if (req.method() == xmrig::HttpRequest::Put && req.match("/1/config")) {
        m_controller->config()->reload(req.body());
        return;
    }

    reply.status = 404;
}


void ApiRouter::tick(const NetworkState &network)
{
    m_network = network;
}


void ApiRouter::onConfigChanged(xmrig::Config *config, xmrig::Config *previousConfig)
{
    updateWorkerId(config->apiWorkerId(), previousConfig->apiWorkerId());
}


void ApiRouter::finalize(xmrig::HttpReply &reply, rapidjson::Document &doc) const
{
    rapidjson::StringBuffer buffer(0, 4096);
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
    writer.SetMaxDecimalPlaces(10);
    doc.Accept(writer);

    reply.status = 200;
    reply.buf    = strdup(buffer.GetString());
    reply.size   = buffer.GetSize();
}


void ApiRouter::genId(const char *id)
{
    memset(m_id, 0, sizeof(m_id));

    if (id && strlen(id) > 0) {
        strncpy(m_id, id, sizeof(m_id) - 1);
        return;
    }

    uv_interface_address_t *interfaces;
    int count = 0;

    if (uv_interface_addresses(&interfaces, &count) < 0) {
        return;
    }

    for (int i = 0; i < count; i++) {
        if (!interfaces[i].is_internal && interfaces[i].address.address4.sin_family == AF_INET) {
            uint8_t hash[200];
            const size_t addrSize = sizeof(interfaces[i].phys_addr);
            const size_t inSize   = strlen(APP_KIND) + addrSize + sizeof(uint16_t);
            const uint16_t port   = static_cast<uint16_t>(m_controller->config()->apiPort());

            uint8_t *input = new uint8_t[inSize]();
            memcpy(input, &port, sizeof(uint16_t));
            memcpy(input + sizeof(uint16_t), interfaces[i].phys_addr, addrSize);
            memcpy(input + sizeof(uint16_t) + addrSize, APP_KIND, strlen(APP_KIND));

            xmrig::keccak(input, inSize, hash);
            Job::toHex(hash, 8, m_id);

            delete [] input;
            break;
        }
    }

    uv_free_interface_addresses(interfaces, count);
}


void ApiRouter::getConnection(rapidjson::Document &doc) const
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


void ApiRouter::getHashrate(rapidjson::Document &doc) const
{
    auto &allocator = doc.GetAllocator();

    rapidjson::Value hashrate(rapidjson::kObjectType);
    rapidjson::Value total(rapidjson::kArrayType);
    rapidjson::Value threads(rapidjson::kArrayType);

    const Hashrate *hr = Workers::hashrate();

    total.PushBack(normalize(hr->calc(Hashrate::ShortInterval)),  allocator);
    total.PushBack(normalize(hr->calc(Hashrate::MediumInterval)), allocator);
    total.PushBack(normalize(hr->calc(Hashrate::LargeInterval)),  allocator);

    for (size_t i = 0; i < Workers::threads(); i++) {
        rapidjson::Value thread(rapidjson::kArrayType);
        thread.PushBack(normalize(hr->calc(i, Hashrate::ShortInterval)),  allocator);
        thread.PushBack(normalize(hr->calc(i, Hashrate::MediumInterval)), allocator);
        thread.PushBack(normalize(hr->calc(i, Hashrate::LargeInterval)),  allocator);

        threads.PushBack(thread, allocator);
    }

    hashrate.AddMember("total",   total, allocator);
    hashrate.AddMember("highest", normalize(hr->highest()), allocator);
    hashrate.AddMember("threads", threads, allocator);
    doc.AddMember("hashrate", hashrate, allocator);
}


void ApiRouter::getIdentify(rapidjson::Document &doc) const
{
    doc.AddMember("id",        rapidjson::StringRef(m_id),       doc.GetAllocator());
    doc.AddMember("worker_id", rapidjson::StringRef(m_workerId), doc.GetAllocator());
}


void ApiRouter::getMiner(rapidjson::Document &doc) const
{
    using namespace xmrig;
    auto &allocator = doc.GetAllocator();

    rapidjson::Value cpu(rapidjson::kObjectType);
    cpu.AddMember("brand",   rapidjson::StringRef(Cpu::info()->brand()), allocator);
    cpu.AddMember("aes",     Cpu::info()->hasAES(), allocator);
    cpu.AddMember("x64",     Cpu::info()->isX64(), allocator);
    cpu.AddMember("sockets", Cpu::info()->sockets(), allocator);

    doc.AddMember("version",      APP_VERSION, allocator);
    doc.AddMember("kind",         APP_KIND, allocator);
    doc.AddMember("ua",           rapidjson::StringRef(Platform::userAgent()), allocator);
    doc.AddMember("cpu",          cpu, allocator);
    doc.AddMember("algo",         rapidjson::StringRef(m_controller->config()->algorithm().name()), allocator);
    doc.AddMember("hugepages",    Workers::hugePages() > 0, allocator);
    doc.AddMember("donate_level", m_controller->config()->donateLevel(), allocator);
}


void ApiRouter::getResults(rapidjson::Document &doc) const
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


void ApiRouter::getThreads(rapidjson::Document &doc) const
{
    doc.SetObject();
    auto &allocator = doc.GetAllocator();
    const Hashrate *hr = Workers::hashrate();

    Workers::threadsSummary(doc);

    const std::vector<xmrig::IThread *> &threads = m_controller->config()->threads();
    rapidjson::Value list(rapidjson::kArrayType);

    for (const xmrig::IThread *thread : threads) {
        rapidjson::Value value = thread->toAPI(doc);

        rapidjson::Value hashrate(rapidjson::kArrayType);
        hashrate.PushBack(normalize(hr->calc(thread->index(), Hashrate::ShortInterval)),  allocator);
        hashrate.PushBack(normalize(hr->calc(thread->index(), Hashrate::MediumInterval)), allocator);
        hashrate.PushBack(normalize(hr->calc(thread->index(), Hashrate::LargeInterval)),  allocator);

        value.AddMember("hashrate", hashrate, allocator);
        list.PushBack(value, allocator);
    }

    doc.AddMember("threads", list, allocator);
}


void ApiRouter::setWorkerId(const char *id)
{
    memset(m_workerId, 0, sizeof(m_workerId));

    if (id && strlen(id) > 0) {
        strncpy(m_workerId, id, sizeof(m_workerId) - 1);
    }
    else {
        gethostname(m_workerId, sizeof(m_workerId) - 1);
    }
}


void ApiRouter::updateWorkerId(const char *id, const char *previousId)
{
    if (id == previousId) {
        return;
    }

    if (id != nullptr && previousId != nullptr && strcmp(id, previousId) == 0) {
        return;
    }

    setWorkerId(id);
}
