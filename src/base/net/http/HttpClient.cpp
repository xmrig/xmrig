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


#include <sstream>


#include "3rdparty/http-parser/http_parser.h"
#include "base/io/log/Log.h"
#include "base/net/dns/Dns.h"
#include "base/net/http/HttpClient.h"
#include "base/tools/Baton.h"
#include "common/Platform.h"


namespace xmrig {

static const char *kCRLF = "\r\n";


class ClientWriteBaton : public Baton<uv_write_t>
{
public:
    inline ClientWriteBaton(const std::string &header, std::string &&body) :
        m_body(std::move(body)),
        m_header(header)
    {
        bufs[0].len  = m_header.size();
        bufs[0].base = const_cast<char *>(m_header.c_str());

        if (!m_body.empty()) {
            bufs[1].len  = m_body.size();
            bufs[1].base = const_cast<char *>(m_body.c_str());
        }
        else {
            bufs[1].base = nullptr;
            bufs[1].len  = 0;
        }
    }


    inline size_t count() const                      { return bufs[1].base == nullptr ? 1 : 2; }
    inline size_t size() const                       { return bufs[0].len + bufs[1].len; }
    inline static void onWrite(uv_write_t *req, int) { delete reinterpret_cast<ClientWriteBaton *>(req->data); }


    uv_buf_t bufs[2];

private:
    std::string m_body;
    std::string m_header;
};


} // namespace xmrig


xmrig::HttpClient::HttpClient(int method, const String &url, IHttpListener *listener, const char *data, size_t size) :
    HttpContext(HTTP_RESPONSE, listener),
    m_quiet(false),
    m_port(0)
{
    this->method = method;
    this->url    = url;

    if (data) {
        body = size ? std::string(data, size) : data;
    }

    m_dns = new Dns(this);
}


xmrig::HttpClient::~HttpClient()
{
    delete m_dns;
}


bool xmrig::HttpClient::connect(const String &host, uint16_t port)
{
    m_port = port;

    return m_dns->resolve(host);
}


const xmrig::String &xmrig::HttpClient::host() const
{
    return m_dns->host();
}


void xmrig::HttpClient::onResolved(const Dns &dns, int status)
{
    this->status = status;

    if (status < 0 && dns.isEmpty()) {
        if (!m_quiet) {
            LOG_ERR("[%s:%d] DNS error: \"%s\"", dns.host().data(), m_port, uv_strerror(status));
        }

        return;
    }

    sockaddr *addr = dns.get().addr(m_port);

    uv_connect_t *req = new uv_connect_t;
    req->data = this;

    uv_tcp_connect(req, m_tcp, addr, onConnect);
}


void xmrig::HttpClient::handshake()
{
    headers.insert({ "Host",       m_dns->host().data() });
    headers.insert({ "Connection", "close" });
    headers.insert({ "User-Agent", Platform::userAgent() });

    if (body.size()) {
        headers.insert({ "Content-Length", std::to_string(body.size()) });
    }

    std::stringstream ss;
    ss << http_method_str(static_cast<http_method>(method)) << " " << url << " HTTP/1.1" << kCRLF;

    for (auto &header : headers) {
        ss << header.first << ": " << header.second << kCRLF;
    }

    ss << kCRLF;

    headers.clear();

    write(ss.str());
}


void xmrig::HttpClient::read(const char *data, size_t size)
{
    if (parse(data, size) < size) {
        close(UV_EPROTO);
    }
}


void xmrig::HttpClient::write(const std::string &header)
{
    ClientWriteBaton *baton = new ClientWriteBaton(header, std::move(body));
    uv_write(&baton->req, stream(), baton->bufs, baton->count(), ClientWriteBaton::onWrite);
}


void xmrig::HttpClient::onConnect(uv_connect_t *req, int status)
{
    HttpClient *client = static_cast<HttpClient *>(req->data);
    if (!client) {
        delete req;
        return;
    }

    if (status < 0) {
        if (!client->m_quiet) {
            LOG_ERR("[%s:%d] connect error: \"%s\"", client->m_dns->host().data(), client->m_port, uv_strerror(status));
        }

        delete req;
        client->close(status);
        return;
    }

    uv_read_start(client->stream(),
        [](uv_handle_t *, size_t suggested_size, uv_buf_t *buf)
        {
            buf->base = new char[suggested_size];

#           ifdef _WIN32
            buf->len = static_cast<unsigned int>(suggested_size);
#           else
            buf->len = suggested_size;
#           endif
        },
        [](uv_stream_t *tcp, ssize_t nread, const uv_buf_t *buf)
        {
            HttpClient *client = static_cast<HttpClient*>(tcp->data);

            if (nread >= 0) {
                client->read(buf->base, static_cast<size_t>(nread));
            } else {
                if (!client->m_quiet && nread != UV_EOF) {
                    LOG_ERR("[%s:%d] read error: \"%s\"", client->m_dns->host().data(), client->m_port, uv_strerror(static_cast<int>(nread)));
                }

                client->close(static_cast<int>(nread));
            }

            delete [] buf->base;
        });

    client->handshake();
}
