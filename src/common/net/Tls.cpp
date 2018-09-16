/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018      SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2018 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


Client::Tls::Tls(Client *client) :
    m_buf(),
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


Client::Tls::~Tls()
{
    if (m_ctx) {
        SSL_CTX_free(m_ctx);
    }

    if (m_ssl) {
        SSL_free(m_ssl);
    }
}


bool Client::Tls::handshake()
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


bool Client::Tls::send(const char *data, size_t size)
{
    SSL_write(m_ssl, data, size);

    return send();
}


void Client::Tls::read(const char *data, size_t size)
{
    BIO_write(m_readBio, data, size);

    if (!SSL_is_init_finished(m_ssl)) {
      const int rc = SSL_connect(m_ssl);

      if (rc < 0 && SSL_get_error(m_ssl, rc) == SSL_ERROR_WANT_READ) {
          send();
      }

      if (rc == 1) {
          if (!verify()) {
              LOG_ERR("[%s] TLS certificate verification failed", m_client->m_pool.url());
              m_client->close();

              return;
          }

          m_client->login();
      }

      return;
    }

    int bytes_read = 0;
    while ((bytes_read = SSL_read(m_ssl, m_buf, sizeof(m_buf))) > 0) {
        m_client->parse(m_buf, bytes_read);
    }
}


bool Client::Tls::send()
{
    return m_client->send(m_writeBio);
}


bool Client::Tls::verify()
{
    X509* cert = SSL_get_peer_certificate(m_ssl);
    if (cert == nullptr) {
        return false;
    }

    return true;
}
