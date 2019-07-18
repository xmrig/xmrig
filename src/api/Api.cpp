/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
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


#include <uv.h>


#ifndef _WIN32
#   include <unistd.h>
#endif


#include "3rdparty/http-parser/http_parser.h"
#include "api/Api.h"
#include "api/interfaces/IApiListener.h"
#include "api/requests/HttpApiRequest.h"
#include "base/kernel/Base.h"
#include "base/tools/Buffer.h"
#include "base/tools/Chrono.h"
#include "core/config/Config.h"
#include "core/Controller.h"
#include "crypto/common/Algorithm.h"
#include "crypto/common/keccak.h"
#include "version.h"


#ifdef XMRIG_FEATURE_HTTP
#   include "api/Httpd.h"
#endif


xmrig::Api::Api(Base *base) :
    m_base(base),
    m_id(),
    m_workerId(),
    m_httpd(nullptr),
    m_timestamp(Chrono::steadyMSecs())
{
    base->addListener(this);

    genId(base->config()->apiId());
}


xmrig::Api::~Api()
{
#   ifdef XMRIG_FEATURE_HTTP
    delete m_httpd;
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
    m_httpd = new Httpd(m_base);
    m_httpd->start();
#   endif
}


void xmrig::Api::stop()
{
#   ifdef XMRIG_FEATURE_HTTP
    m_httpd->stop();
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
        request.reply().AddMember("id",        StringRef(m_id),       allocator);
        request.reply().AddMember("worker_id", StringRef(m_workerId), allocator);
        request.reply().AddMember("uptime",    (Chrono::steadyMSecs() - m_timestamp) / 1000, allocator);

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
#       ifdef XMRIG_FEATURE_LIBCPUID
        features.PushBack("cpuid", allocator);
#       endif
#       ifdef XMRIG_FEATURE_TLS
        features.PushBack("tls", allocator);
#       endif
        request.reply().AddMember("features", features, allocator);
    }

    for (IApiListener *listener : m_listeners) {
        listener->onRequest(request);

        if (request.isDone()) {
            return;
        }
    }

    request.done(request.isNew() ? HTTP_STATUS_NOT_FOUND : HTTP_STATUS_OK);
}


void xmrig::Api::genId(const String &id)
{
    memset(m_id, 0, sizeof(m_id));

    if (id.size() > 0) {
        strncpy(m_id, id.data(), sizeof(m_id) - 1);
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
            const uint16_t port   = static_cast<uint16_t>(m_base->config()->http().port());

            uint8_t *input = new uint8_t[inSize]();
            memcpy(input, &port, sizeof(uint16_t));
            memcpy(input + sizeof(uint16_t), interfaces[i].phys_addr, addrSize);
            memcpy(input + sizeof(uint16_t) + addrSize, APP_KIND, strlen(APP_KIND));

            xmrig::keccak(input, inSize, hash);
            xmrig::Buffer::toHex(hash, 8, m_id);

            delete [] input;
            break;
        }
    }

    uv_free_interface_addresses(interfaces, count);
}


void xmrig::Api::genWorkerId(const String &id)
{
    memset(m_workerId, 0, sizeof(m_workerId));

    if (id.size() > 0) {
        strncpy(m_workerId, id.data(), sizeof(m_workerId) - 1);
    }
    else {
        gethostname(m_workerId, sizeof(m_workerId) - 1);
    }
}
