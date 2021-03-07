/* XMRig
 * Copyright (c) 2014-2019 heapwolf    <https://github.com/heapwolf>
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


#ifndef XMRIG_HTTPCONTEXT_H
#define XMRIG_HTTPCONTEXT_H


using llhttp_settings_t     = struct llhttp_settings_s;
using llhttp_t              = struct llhttp__internal_s;
using uv_connect_t          = struct uv_connect_s;
using uv_handle_t           = struct uv_handle_s;
using uv_stream_t           = struct uv_stream_s;
using uv_tcp_t              = struct uv_tcp_s;


#include "base/net/http/HttpData.h"
#include "base/tools/Object.h"


#include <memory>


namespace xmrig {


class IHttpListener;


class HttpContext : public HttpData
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(HttpContext)

    HttpContext(int parser_type, const std::weak_ptr<IHttpListener> &listener);
    ~HttpContext() override;

    inline uv_stream_t *stream() const { return reinterpret_cast<uv_stream_t *>(m_tcp); }
    inline uv_handle_t *handle() const { return reinterpret_cast<uv_handle_t *>(m_tcp); }

    inline const char *host() const override            { return nullptr; }
    inline const char *tlsFingerprint() const override  { return nullptr; }
    inline const char *tlsVersion() const override      { return nullptr; }
    inline uint16_t port() const override               { return 0; }

    void write(std::string &&data, bool close) override;

    bool isRequest() const override;
    size_t parse(const char *data, size_t size);
    std::string ip() const override;
    uint64_t elapsed() const;
    void close(int status = 0);

    static HttpContext *get(uint64_t id);
    static void closeAll();

protected:
    uv_tcp_t *m_tcp;

private:
    inline IHttpListener *httpListener() const { return m_listener.expired() ? nullptr : m_listener.lock().get(); }

    static int onHeaderField(llhttp_t *parser, const char *at, size_t length);
    static int onHeaderValue(llhttp_t *parser, const char *at, size_t length);
    static void attach(llhttp_settings_t *settings);

    void setHeader();

    bool m_wasHeaderValue           = false;
    const uint64_t m_timestamp;
    llhttp_t *m_parser;
    std::string m_lastHeaderField;
    std::string m_lastHeaderValue;
    std::weak_ptr<IHttpListener> m_listener;
};


} // namespace xmrig


#endif // XMRIG_HTTPCONTEXT_H

