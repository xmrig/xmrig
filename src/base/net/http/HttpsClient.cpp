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


#include <assert.h>
#include <openssl/ssl.h>
#include <uv.h>


#include "base/io/log/Log.h"
#include "base/net/http/HttpsClient.h"
#include "base/tools/Buffer.h"


#ifdef _MSC_VER
#   define strncasecmp(x,y,z) _strnicmp(x,y,z)
#endif


xmrig::HttpsClient::HttpsClient(int method, const String &url, IHttpListener *listener, const char *data, size_t size, const String &fingerprint) :
    HttpClient(method, url, listener, data, size),
    m_ready(false),
    m_buf(),
    m_ssl(nullptr),
    m_fp(fingerprint)
{
    m_ctx = SSL_CTX_new(SSLv23_method());
    assert(m_ctx != nullptr);

    if (!m_ctx) {
        return;
    }

    m_writeBio = BIO_new(BIO_s_mem());
    m_readBio  = BIO_new(BIO_s_mem());
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


const char *xmrig::HttpsClient::fingerprint() const
{
    return m_ready ? m_fingerprint : nullptr;
}


const char *xmrig::HttpsClient::version() const
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
    SSL_set_bio(m_ssl, m_readBio, m_writeBio);
    SSL_set_tlsext_host_name(m_ssl, host().data());

    SSL_do_handshake(m_ssl);

    flush();
}


void xmrig::HttpsClient::read(const char *data, size_t size)
{
    BIO_write(m_readBio, data, size);

    if (!SSL_is_init_finished(m_ssl)) {
        const int rc = SSL_connect(m_ssl);

        if (rc < 0 && SSL_get_error(m_ssl, rc) == SSL_ERROR_WANT_READ) {
            flush();
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

    int bytes_read = 0;
    while ((bytes_read = SSL_read(m_ssl, m_buf, sizeof(m_buf))) > 0) {
        HttpClient::read(m_buf, static_cast<size_t>(bytes_read));
    }
}


void xmrig::HttpsClient::write(const std::string &header)
{
    SSL_write(m_ssl, (header + body).c_str(), header.size() + body.size());
    body.clear();

    flush();
}


bool xmrig::HttpsClient::verify(X509 *cert)
{
    if (cert == nullptr) {
        return false;
    }

    if (!verifyFingerprint(cert)) {
        if (!m_quiet) {
            LOG_ERR("[%s:%d] Failed to verify server certificate fingerprint", host().data(), port());

            if (strlen(m_fingerprint) == 64 && !m_fp.isNull()) {
                LOG_ERR("\"%s\" was given", m_fingerprint);
                LOG_ERR("\"%s\" was configured", m_fp.data());
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
    unsigned int dlen;

    if (X509_digest(cert, digest, md, &dlen) != 1) {
        return false;
    }

    Buffer::toHex(md, 32, m_fingerprint);

    return m_fp.isNull() || strncasecmp(m_fingerprint, m_fp.data(), 64) == 0;
}


void xmrig::HttpsClient::flush()
{
    uv_buf_t buf;
    buf.len = BIO_get_mem_data(m_writeBio, &buf.base);

    if (buf.len == 0) {
        return;
    }

    bool result = false;
    if (uv_is_writable(stream())) {
        result = uv_try_write(stream(), &buf, 1) == static_cast<int>(buf.len);

        if (!result) {
            close(UV_EIO);
        }
    }

    (void) BIO_reset(m_writeBio);
}
