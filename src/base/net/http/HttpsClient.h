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


#ifndef XMRIG_HTTPSCLIENT_H
#define XMRIG_HTTPSCLIENT_H


typedef struct bio_st BIO;
typedef struct ssl_ctx_st SSL_CTX;
typedef struct ssl_st SSL;
typedef struct x509_st X509;


#include "base/net/http/HttpClient.h"


namespace xmrig {


class HttpsClient : public HttpClient
{
public:
    HttpsClient(int method, const String &url, IHttpListener *listener, const char *data = nullptr, size_t size = 0);
    ~HttpsClient() override;

protected:
    void handshake() override;
    void read(const char *data, size_t size) override;
    void write(const std::string &header) override;

private:
    bool verify(X509 *cert) const;
    void flush();

    BIO *m_readBio;
    BIO *m_writeBio;
    bool m_ready;
    char m_buf[1024 * 2];
    SSL *m_ssl;
    SSL_CTX *m_ctx;
};


} // namespace xmrig


#endif // XMRIG_HTTPSCLIENT_H
