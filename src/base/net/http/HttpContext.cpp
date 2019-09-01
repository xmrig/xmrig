/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2014-2019 heapwolf    <https://github.com/heapwolf>
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


#include <algorithm>
#include <uv.h>


#include "3rdparty/http-parser/http_parser.h"
#include "base/kernel/interfaces/IHttpListener.h"
#include "base/net/http/HttpContext.h"


namespace xmrig {

static http_parser_settings http_settings;
static std::map<uint64_t, HttpContext *> storage;
static uint64_t SEQUENCE = 0;

} // namespace xmrig


xmrig::HttpContext::HttpContext(int parser_type, IHttpListener *listener) :
    HttpData(SEQUENCE++),
    m_wasHeaderValue(false),
    m_listener(listener)
{
    storage[id()] = this;

    m_parser = new http_parser;
    m_tcp    = new uv_tcp_t;

    uv_tcp_init(uv_default_loop(), m_tcp);
    uv_tcp_nodelay(m_tcp, 1);

    http_parser_init(m_parser, static_cast<http_parser_type>(parser_type));

    m_parser->data = m_tcp->data = this;

    if (http_settings.on_message_complete == nullptr) {
        attach(&http_settings);
    }
}


xmrig::HttpContext::~HttpContext()
{
    delete m_tcp;
    delete m_parser;
}


size_t xmrig::HttpContext::parse(const char *data, size_t size)
{
    return http_parser_execute(m_parser, &http_settings, data, size);
}


std::string xmrig::HttpContext::ip() const
{
    char ip[46]           = {};
    sockaddr_storage addr = {};
    int size              = sizeof(addr);

    uv_tcp_getpeername(m_tcp, reinterpret_cast<sockaddr*>(&addr), &size);
    if (reinterpret_cast<sockaddr_in *>(&addr)->sin_family == AF_INET6) {
        uv_ip6_name(reinterpret_cast<sockaddr_in6*>(&addr), ip, 45);
    }
    else {
        uv_ip4_name(reinterpret_cast<sockaddr_in*>(&addr), ip, 16);
    }

    return ip;
}


void xmrig::HttpContext::close(int status)
{
    if (status < 0 && m_listener) {
        this->status = status;
        m_listener->onHttpData(*this);
    }

    auto it = storage.find(id());
    if (it != storage.end()) {
        storage.erase(it);
    }

    if (!uv_is_closing(handle())) {
        uv_close(handle(), [](uv_handle_t *handle) -> void { delete reinterpret_cast<HttpContext*>(handle->data); });
    }
}


xmrig::HttpContext *xmrig::HttpContext::get(uint64_t id)
{
    if (storage.count(id) == 0) {
        return nullptr;
    }

    return storage[id];
}


void xmrig::HttpContext::closeAll()
{
    for (auto kv : storage) {
        if (!uv_is_closing(kv.second->handle())) {
            uv_close(kv.second->handle(), [](uv_handle_t *handle) -> void { delete reinterpret_cast<HttpContext*>(handle->data); });
        }
    }
}


int xmrig::HttpContext::onHeaderField(http_parser *parser, const char *at, size_t length)
{
    HttpContext *ctx = static_cast<HttpContext*>(parser->data);

    if (ctx->m_wasHeaderValue) {
        if (!ctx->m_lastHeaderField.empty()) {
            ctx->setHeader();
        }

        ctx->m_lastHeaderField = std::string(at, length);
        ctx->m_wasHeaderValue  = false;
    } else {
        ctx->m_lastHeaderField += std::string(at, length);
    }

    return 0;
}


int xmrig::HttpContext::onHeaderValue(http_parser *parser, const char *at, size_t length)
{
    HttpContext *ctx = static_cast<HttpContext*>(parser->data);

    if (!ctx->m_wasHeaderValue) {
        ctx->m_lastHeaderValue = std::string(at, length);
        ctx->m_wasHeaderValue  = true;
    } else {
        ctx->m_lastHeaderValue += std::string(at, length);
    }

    return 0;
}


void xmrig::HttpContext::attach(http_parser_settings *settings)
{
    settings->on_message_begin  = nullptr;
    settings->on_status         = nullptr;
    settings->on_chunk_header   = nullptr;
    settings->on_chunk_complete = nullptr;

    settings->on_url = [](http_parser *parser, const char *at, size_t length) -> int
    {
        static_cast<HttpContext*>(parser->data)->url = std::string(at, length);
        return 0;
    };

    settings->on_header_field = onHeaderField;
    settings->on_header_value = onHeaderValue;

    settings->on_headers_complete = [](http_parser* parser) -> int {
        HttpContext *ctx = static_cast<HttpContext*>(parser->data);
        ctx->status = parser->status_code;

        if (parser->type == HTTP_REQUEST) {
            ctx->method = parser->method;
        }

        if (!ctx->m_lastHeaderField.empty()) {
            ctx->setHeader();
        }

        return 0;
    };

    settings->on_body = [](http_parser *parser, const char *at, size_t len) -> int
    {
        static_cast<HttpContext*>(parser->data)->body += std::string(at, len);

        return 0;
    };

    settings->on_message_complete = [](http_parser *parser) -> int
    {
        HttpContext *ctx = static_cast<HttpContext*>(parser->data);
        ctx->m_listener->onHttpData(*ctx);
        ctx->m_listener = nullptr;

        return 0;
    };
}


void xmrig::HttpContext::setHeader()
{
    std::transform(m_lastHeaderField.begin(), m_lastHeaderField.end(), m_lastHeaderField.begin(), ::tolower);
    headers.insert({ m_lastHeaderField, m_lastHeaderValue });

    m_lastHeaderField.clear();
    m_lastHeaderValue.clear();
}

