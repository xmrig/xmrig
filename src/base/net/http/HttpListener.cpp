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


#include "base/net/http/HttpListener.h"
#include "3rdparty/llhttp/llhttp.h"
#include "base/io/log/Log.h"
#include "base/net/http/HttpData.h"


void xmrig::HttpListener::onHttpData(const HttpData &data)
{
#   ifdef APP_DEBUG
    if (!data.isRequest()) {
        LOG_DEBUG("%s " CYAN_BOLD("http%s://%s:%u ") MAGENTA_BOLD("\"%s %s\" ") CSI "1;%dm%d" CLEAR BLACK_BOLD(" received: ") CYAN_BOLD("%zu") BLACK_BOLD(" bytes"),
                  m_tag, data.tlsVersion() ? "s" : "", data.host(), data.port(), llhttp_method_name(static_cast<llhttp_method>(data.method)), data.url.data(),
                  (data.status >= 400 || data.status < 0) ? 31 : 32, data.status, data.body.size());

        if (data.body.size() < (Log::kMaxBufferSize - 1024) && data.isJSON()) {
            Log::print(BLUE_BG_BOLD("%s:") BLACK_BOLD_S " %.*s", data.headers.at(HttpData::kContentTypeL).c_str(), static_cast<int>(data.body.size()), data.body.c_str());
        }
    }
#   endif

    m_listener->onHttpData(data);
}
