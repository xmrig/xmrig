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

#include <math.h>
#include <string.h>
#include <uv.h>


#include "api/interfaces/IApiRequest.h"
#include "api/v1/ApiRouter.h"
#include "base/kernel/Base.h"
#include "common/cpu/Cpu.h"
#include "common/Platform.h"
#include "core/config/Config.h"
#include "interfaces/IThread.h"
#include "rapidjson/document.h"
#include "version.h"
#include "workers/Hashrate.h"
#include "workers/Workers.h"


static inline rapidjson::Value normalize(double d)
{
    using namespace rapidjson;

    if (!isnormal(d)) {
        return Value(kNullType);
    }

    return Value(floor(d * 100.0) / 100.0);
}


xmrig::ApiRouter::ApiRouter(Base *base) :
    m_base(base)
{
}


xmrig::ApiRouter::~ApiRouter()
{
}


void xmrig::ApiRouter::onRequest(IApiRequest &request)
{
    if (request.method() == IApiRequest::METHOD_GET) {
        if (request.url() == "/1/summary" || request.url() == "/api.json") {
            request.accept();
            getMiner(request.reply(), request.doc());
            getHashrate(request.reply(), request.doc());
        }
        else if (request.url() == "/1/threads") {
            request.accept();
            getThreads(request.reply(), request.doc());
        }
        else if (request.url() == "/1/config") {
            if (request.isRestricted()) {
                return request.done(403);
            }

            request.accept();
            m_base->config()->getJSON(request.doc());
        }
    }
    else if (request.method() == IApiRequest::METHOD_PUT || request.method() == IApiRequest::METHOD_POST) {
        if (request.url() == "/1/config") {
            request.accept();

            if (!m_base->reload(request.json())) {
                return request.done(400);
            }

            request.done(204);
        }
    }
}


void xmrig::ApiRouter::getHashrate(rapidjson::Value &reply, rapidjson::Document &doc) const
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    Value hashrate(kObjectType);
    Value total(kArrayType);
    Value threads(kArrayType);

    const Hashrate *hr = Workers::hashrate();

    total.PushBack(normalize(hr->calc(Hashrate::ShortInterval)),  allocator);
    total.PushBack(normalize(hr->calc(Hashrate::MediumInterval)), allocator);
    total.PushBack(normalize(hr->calc(Hashrate::LargeInterval)),  allocator);

    for (size_t i = 0; i < Workers::threads(); i++) {
        Value thread(kArrayType);
        thread.PushBack(normalize(hr->calc(i, Hashrate::ShortInterval)),  allocator);
        thread.PushBack(normalize(hr->calc(i, Hashrate::MediumInterval)), allocator);
        thread.PushBack(normalize(hr->calc(i, Hashrate::LargeInterval)),  allocator);

        threads.PushBack(thread, allocator);
    }

    hashrate.AddMember("total",   total, allocator);
    hashrate.AddMember("highest", normalize(hr->highest()), allocator);
    hashrate.AddMember("threads", threads, allocator);
    reply.AddMember("hashrate", hashrate, allocator);
}


void xmrig::ApiRouter::getMiner(rapidjson::Value &reply, rapidjson::Document &doc) const
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    Value cpu(kObjectType);
    cpu.AddMember("brand",   StringRef(Cpu::info()->brand()), allocator);
    cpu.AddMember("aes",     Cpu::info()->hasAES(), allocator);
    cpu.AddMember("x64",     Cpu::info()->isX64(), allocator);
    cpu.AddMember("sockets", Cpu::info()->sockets(), allocator);

    reply.AddMember("version",      APP_VERSION, allocator);
    reply.AddMember("kind",         APP_KIND, allocator);
    reply.AddMember("ua",           StringRef(Platform::userAgent()), allocator);
    reply.AddMember("cpu",          cpu, allocator);
    reply.AddMember("hugepages",    Workers::hugePages() > 0, allocator);
    reply.AddMember("donate_level", m_base->config()->pools().donateLevel(), allocator);
}


void xmrig::ApiRouter::getThreads(rapidjson::Value &reply, rapidjson::Document &doc) const
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();
    const Hashrate *hr = Workers::hashrate();

    Workers::threadsSummary(doc);

    const std::vector<IThread *> &threads = m_base->config()->threads();
    Value list(kArrayType);

    size_t i = 0;
    for (const xmrig::IThread *thread : threads) {
        Value value = thread->toAPI(doc);

        Value hashrate(kArrayType);
        hashrate.PushBack(normalize(hr->calc(i, Hashrate::ShortInterval)),  allocator);
        hashrate.PushBack(normalize(hr->calc(i, Hashrate::MediumInterval)), allocator);
        hashrate.PushBack(normalize(hr->calc(i, Hashrate::LargeInterval)),  allocator);

        i++;

        value.AddMember("hashrate", hashrate, allocator);
        list.PushBack(value, allocator);
    }

    reply.AddMember("threads", list, allocator);
}
