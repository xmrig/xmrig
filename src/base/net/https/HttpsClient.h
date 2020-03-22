/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2014-2019 heapwolf    <https://github.com/heapwolf>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


using BIO       = struct bio_st;
using SSL_CTX   = struct ssl_ctx_st;
using SSL       = struct ssl_st;
using X509      = struct x509_st;


#include "base/net/http/HttpClient.h"
#include "base/tools/String.h"


namespace xmrig {


class HttpsClient : public HttpClient
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(HttpsClient)

    HttpsClient(FetchRequest &&req, const std::weak_ptr<IHttpListener> &listener);
    ~HttpsClient() override;

    const char *tlsFingerprint() const override;
    const char *tlsVersion() const override;

protected:
    void handshake() override;
    void read(const char *data, size_t size) override;

private:
    void write(std::string &&data, bool close) override;

    bool verify(X509 *cert);
    bool verifyFingerprint(X509 *cert);
    void flush(bool close);

    BIO *m_read                         = nullptr;
    BIO *m_write                        = nullptr;
    bool m_ready                        = false;
    char m_fingerprint[32 * 2 + 8]{};
    SSL *m_ssl                          = nullptr;
    SSL_CTX *m_ctx                      = nullptr;
};


} // namespace xmrig


#endif // XMRIG_HTTPSCLIENT_H
