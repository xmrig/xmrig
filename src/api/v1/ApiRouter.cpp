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

#include <cmath>
#include <string.h>
#include <uv.h>


#include "api/interfaces/IApiRequest.h"
#include "api/v1/ApiRouter.h"
#include "backend/common/interfaces/IThread.h"
#include "backend/cpu/Cpu.h"
#include "base/kernel/Base.h"
#include "base/kernel/Platform.h"
#include "core/config/Config.h"
#include "rapidjson/document.h"
#include "version.h"


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
        if (request.url() == "/1/config") {
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
