/* XMRig
 * Copyright (c) 2014-2019 heapwolf    <https://github.com/heapwolf>
 * Copyright (c) 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include <cassert>
#include <openssl/ssl.h>
#include <uv.h>


#include "base/net/https/HttpsClient.h"
#include "base/io/log/Log.h"
#include "base/tools/Cvt.h"


#ifdef _MSC_VER
#   define strncasecmp(x,y,z) _strnicmp(x,y,z)
#endif


xmrig::HttpsClient::HttpsClient(const char *tag, FetchRequest &&req, const std::weak_ptr<IHttpListener> &listener) :
    HttpClient(tag, std::move(req), listener)
{
    m_ctx = SSL_CTX_new(SSLv23_method());
    assert(m_ctx != nullptr);

    if (!m_ctx) {
        return;
    }

    m_write = BIO_new(BIO_s_mem());
    m_read  = BIO_new(BIO_s_mem());
    SSL_CTX_set_options(m_ctx, SSL_OP_NO_SSLv2 | SSL_OP_NO_SSLv3);
}


xmrig::HttpsClient::~HttpsClient()
{
    if (m_ctx) {
        SSL_CTX_free(m_ctx);
    }

    if (m_ssl) {
        SSL_free(m_ssl);
    }
}


const char *xmrig::HttpsClient::tlsFingerprint() const
{
    return m_ready ? m_fingerprint : nullptr;
}


const char *xmrig::HttpsClient::tlsVersion() const
{
    return m_ready ? SSL_get_version(m_ssl) : nullptr;
}


void xmrig::HttpsClient::handshake()
{
    m_ssl = SSL_new(m_ctx);
    assert(m_ssl != nullptr);

    if (!m_ssl) {
        return;
    }

    SSL_set_connect_state(m_ssl);
    SSL_set_bio(m_ssl, m_read, m_write);
    SSL_set_tlsext_host_name(m_ssl, host());

    SSL_do_handshake(m_ssl);

    flush(false);
}


void xmrig::HttpsClient::read(const char *data, size_t size)
{
    BIO_write(m_read, data, size);

    if (!SSL_is_init_finished(m_ssl)) {
        const int rc = SSL_connect(m_ssl);

        if (rc < 0 && SSL_get_error(m_ssl, rc) == SSL_ERROR_WANT_READ) {
            flush(false);
        } else if (rc == 1) {
            X509 *cert = SSL_get_peer_certificate(m_ssl);
            if (!verify(cert)) {
                X509_free(cert);
                return close(UV_EPROTO);
            }

            X509_free(cert);
            m_ready = true;

            HttpClient::handshake();
      }

      return;
    }

    static char buf[16384]{};

    int rc = 0;
    while ((rc = SSL_read(m_ssl, buf, sizeof(buf))) > 0) {
        HttpClient::read(buf, static_cast<size_t>(rc));
    }

    if (rc == 0) {
        close(UV_EOF);
    }
}


void xmrig::HttpsClient::write(std::string &&data, bool close)
{
    const std::string body = std::move(data);
    SSL_write(m_ssl, body.data(), body.size());

    flush(close);
}


bool xmrig::HttpsClient::verify(X509 *cert)
{
    if (cert == nullptr) {
        return false;
    }

    if (!verifyFingerprint(cert)) {
        if (!isQuiet()) {
            LOG_ERR("[%s:%d] Failed to verify server certificate fingerprint", host(), port());

            if (strlen(m_fingerprint) == 64 && !req().fingerprint.isNull()) {
                LOG_ERR("\"%s\" was given", m_fingerprint);
                LOG_ERR("\"%s\" was configured", req().fingerprint.data());
            }
        }

        return false;
    }

    return true;
}


bool xmrig::HttpsClient::verifyFingerprint(X509 *cert)
{
    const EVP_MD *digest = EVP_get_digestbyname("sha256");
    if (digest == nullptr) {
        return false;
    }

    unsigned char md[EVP_MAX_MD_SIZE];
    unsigned int dlen = 0;

    if (X509_digest(cert, digest, md, &dlen) != 1) {
        return false;
    }

    Cvt::toHex(m_fingerprint, sizeof(m_fingerprint), md, 32);

    return req().fingerprint.isNull() || strncasecmp(m_fingerprint, req().fingerprint.data(), 64) == 0;
}


void xmrig::HttpsClient::flush(bool close)
{
    if (uv_is_writable(stream()) != 1) {
        return;
    }

    char *data        = nullptr;
    const long size = BIO_get_mem_data(m_write, &data); // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
    std::string body(data, (size > 0) ? size : 0);

    (void) BIO_reset(m_write);

    if (!body.empty()) {
        HttpContext::write(std::move(body), close);
    }
}
