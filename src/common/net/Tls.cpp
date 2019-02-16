/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
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


#include "common/net/Client.h"
#include "common/net/Tls.h"
#include "common/log/Log.h"


#ifdef _MSC_VER
#   define strncasecmp(x,y,z) _strnicmp(x,y,z)
#endif


xmrig::Client::Tls::Tls(Client *client) :
    m_ready(false),
    m_buf(),
    m_fingerprint(),
    m_client(client),
    m_ssl(nullptr)
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


xmrig::Client::Tls::~Tls()
{
    if (m_ctx) {
        SSL_CTX_free(m_ctx);
    }

    if (m_ssl) {
        SSL_free(m_ssl);
    }
}


bool xmrig::Client::Tls::handshake()
{
    m_ssl = SSL_new(m_ctx);
    assert(m_ssl != nullptr);

    if (!m_ssl) {
        return false;
    }

    SSL_set_connect_state(m_ssl);
    SSL_set_bio(m_ssl, m_readBio, m_writeBio);
    SSL_do_handshake(m_ssl);

    return send();
}


bool xmrig::Client::Tls::send(const char *data, size_t size)
{
    SSL_write(m_ssl, data, size);

    return send();
}


const char *xmrig::Client::Tls::fingerprint() const
{
    return m_ready ? m_fingerprint : nullptr;
}


const char *xmrig::Client::Tls::version() const
{
    return m_ready ? SSL_get_version(m_ssl) : nullptr;
}


void xmrig::Client::Tls::read(const char *data, size_t size)
{
    BIO_write(m_readBio, data, size);

    if (!SSL_is_init_finished(m_ssl)) {
        const int rc = SSL_connect(m_ssl);

        if (rc < 0 && SSL_get_error(m_ssl, rc) == SSL_ERROR_WANT_READ) {
            send();
        } else if (rc == 1) {
            X509 *cert = SSL_get_peer_certificate(m_ssl);
            if (!verify(cert)) {
                X509_free(cert);
                m_client->close();

                return;
            }

            X509_free(cert);
            m_ready = true;
            m_client->login();
      }

      return;
    }

    int bytes_read = 0;
    while ((bytes_read = SSL_read(m_ssl, m_buf, sizeof(m_buf))) > 0) {
        m_client->parse(m_buf, bytes_read);
    }
}


bool xmrig::Client::Tls::send()
{
    return m_client->send(m_writeBio);
}


bool xmrig::Client::Tls::verify(X509 *cert)
{
    if (cert == nullptr) {
        LOG_ERR("[%s] Failed to get server certificate", m_client->m_pool.url());

        return false;
    }

    if (!verifyFingerprint(cert)) {
        LOG_ERR("[%s] Failed to verify server certificate fingerprint", m_client->m_pool.url());

        const char *fingerprint = m_client->m_pool.fingerprint();
        if (strlen(m_fingerprint) == 64 && fingerprint != nullptr) {
            LOG_ERR("\"%s\" was given", m_fingerprint);
            LOG_ERR("\"%s\" was configured", fingerprint);
        }

        return false;
    }

    return true;
}


bool xmrig::Client::Tls::verifyFingerprint(X509 *cert)
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

    Job::toHex(md, 32, m_fingerprint);
    const char *fingerprint = m_client->m_pool.fingerprint();

    return fingerprint == nullptr || strncasecmp(m_fingerprint, fingerprint, 64) == 0;
}
