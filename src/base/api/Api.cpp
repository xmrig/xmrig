/* XMRig
 * Copyright (c) 2018-2024 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2024 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#include <uv.h>


#include "base/api/Api.h"
#include "base/api/interfaces/IApiListener.h"
#include "base/api/requests/HttpApiRequest.h"
#include "base/crypto/keccak.h"
#include "base/io/Env.h"
#include "base/io/json/Json.h"
#include "base/io/log/Log.h"
#include "base/io/log/Tags.h"
#include "base/kernel/Base.h"
#include "base/tools/Chrono.h"
#include "base/tools/Cvt.h"
#include "core/config/Config.h"
#include "core/Controller.h"
#include "version.h"


#ifdef XMRIG_FEATURE_HTTP
#   include "base/api/Httpd.h"
#endif


#include <thread>


namespace xmrig {


static rapidjson::Value getResources(rapidjson::Document &doc)
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    size_t rss = 0;
    uv_resident_set_memory(&rss);

    Value out(kObjectType);
    Value memory(kObjectType);
    Value load_average(kArrayType);

    memory.AddMember("free",                uv_get_free_memory(), allocator);
    memory.AddMember("total",               uv_get_total_memory(), allocator);
    memory.AddMember("resident_set_memory", static_cast<uint64_t>(rss), allocator);

    double loadavg[3] = { 0.0 };
    uv_loadavg(loadavg);

    for (double value : loadavg) {
        load_average.PushBack(Json::normalize(value, true), allocator);
    }

    out.AddMember("memory",               memory, allocator);
    out.AddMember("load_average",         load_average, allocator);
    out.AddMember("hardware_concurrency", std::thread::hardware_concurrency(), allocator);

    return out;
}


} // namespace xmrig


xmrig::Api::Api(Base *base) :
    m_base(base),
    m_timestamp(Chrono::currentMSecsSinceEpoch())
{
    base->addListener(this);

    genId(base->config()->apiId());
}


xmrig::Api::~Api()
{
#   ifdef XMRIG_FEATURE_HTTP
    if (m_httpd) {
        m_httpd->stop();
        delete m_httpd;
        m_httpd = nullptr; // Ensure the pointer is set to nullptr after deletion
    }
#   endif
}


void xmrig::Api::request(const HttpData &req)
{
    HttpApiRequest request(req, m_base->config()->http().isRestricted());

    exec(request);
}


void xmrig::Api::start()
{
    genWorkerId(m_base->config()->apiWorkerId());

#   ifdef XMRIG_FEATURE_HTTP
    if (!m_httpd) {
        m_httpd = new Httpd(m_base);
        if (!m_httpd->start()) {
            LOG_ERR("%s " RED_BOLD("HTTP API server failed to start."), Tags::network());

            delete m_httpd; // Properly handle failure to start
            m_httpd = nullptr;
        }
    }
#   endif
}


void xmrig::Api::stop()
{
#   ifdef XMRIG_FEATURE_HTTP
    if (m_httpd) {
        m_httpd->stop();
    }
#   endif
}


void xmrig::Api::tick()
{
#   ifdef XMRIG_FEATURE_HTTP
    if (!m_httpd || !m_base->config()->http().isEnabled() || m_httpd->isBound()) {
        return;
    }

    if (++m_ticks % 10 == 0) {
        m_ticks = 0;
        if (m_httpd) {
            m_httpd->start();
        }
    }
#   endif
}


void xmrig::Api::onConfigChanged(Config *config, Config *previousConfig)
{
    if (config->apiId() != previousConfig->apiId()) {
        genId(config->apiId());
    }

    if (config->apiWorkerId() != previousConfig->apiWorkerId()) {
        genWorkerId(config->apiWorkerId());
    }
}


void xmrig::Api::exec(IApiRequest &request)
{
    using namespace rapidjson;

    if (request.type() == IApiRequest::REQ_SUMMARY) {
        auto &allocator = request.doc().GetAllocator();

        request.accept();

        auto &reply = request.reply();
        reply.AddMember("id",         StringRef(m_id),       allocator);
        reply.AddMember("worker_id",  m_workerId.toJSON(), allocator);
        reply.AddMember("uptime",     (Chrono::currentMSecsSinceEpoch() - m_timestamp) / 1000, allocator);
        reply.AddMember("restricted", request.isRestricted(), allocator);
        reply.AddMember("resources",  getResources(request.doc()), allocator);

        Value features(kArrayType);
#       ifdef XMRIG_FEATURE_API
        features.PushBack("api", allocator);
#       endif
#       ifdef XMRIG_FEATURE_ASM
        features.PushBack("asm", allocator);
#       endif
#       ifdef XMRIG_FEATURE_HTTP
        features.PushBack("http", allocator);
#       endif
#       ifdef XMRIG_FEATURE_HWLOC
        features.PushBack("hwloc", allocator);
#       endif
#       ifdef XMRIG_FEATURE_TLS
        features.PushBack("tls", allocator);
#       endif
#       ifdef XMRIG_FEATURE_OPENCL
        features.PushBack("opencl", allocator);
#       endif
#       ifdef XMRIG_FEATURE_CUDA
        features.PushBack("cuda", allocator);
#       endif
        reply.AddMember("features", features, allocator);
    }

    for (IApiListener *listener : m_listeners) {
        listener->onRequest(request);

        if (request.isDone()) {
            return;
        }
    }

    request.done(request.isNew() ? 404 : 200);
}


void xmrig::Api::genId(const String &id)
{
    memset(m_id, 0, sizeof(m_id));

    if (id.size() > 0) {
        strncpy(m_id, id.data(), sizeof(m_id) - 1);
        return;
    }

    uv_interface_address_t *interfaces = nullptr;
    int count = 0;

    if (uv_interface_addresses(&interfaces, &count) < 0) {
        return;
    }

    for (int i = 0; i < count; i++) {
        if (!interfaces[i].is_internal && interfaces[i].address.address4.sin_family == AF_INET) {
            uint8_t hash[200];
            const size_t addrSize = sizeof(interfaces[i].phys_addr);
            const size_t inSize   = (sizeof(APP_KIND) - 1) + addrSize + sizeof(uint16_t);
            const auto port       = static_cast<uint16_t>(m_base->config()->http().port());

            auto *input = new uint8_t[inSize]();
            memcpy(input, &port, sizeof(uint16_t));
            memcpy(input + sizeof(uint16_t), interfaces[i].phys_addr, addrSize);
            memcpy(input + sizeof(uint16_t) + addrSize, APP_KIND, (sizeof(APP_KIND) - 1));

            keccak(input, inSize, hash);
            Cvt::toHex(m_id, sizeof(m_id), hash, 8);

            delete [] input;
            break;
        }
    }

    uv_free_interface_addresses(interfaces, count);
}


void xmrig::Api::genWorkerId(const String &id)
{
    m_workerId = Env::expand(id);
    if (m_workerId.isEmpty()) {
        m_workerId = Env::hostname();
    }
}
