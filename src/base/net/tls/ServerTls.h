/* xmlcore
 * Copyright (c) 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2020 xmlcore       <https://github.com/xmlcore>, <support@xmlcore.com>
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


#ifndef xmlcore_SERVERTLS_H
#define xmlcore_SERVERTLS_H


using BIO       = struct bio_st;
using SSL       = struct ssl_st;
using SSL_CTX   = struct ssl_ctx_st;



#include "base/tools/Object.h"


namespace xmlcore {


class ServerTls
{
public:
    xmlcore_DISABLE_COPY_MOVE_DEFAULT(ServerTls)

    ServerTls(SSL_CTX *ctx);
    virtual ~ServerTls();

    static bool isHTTP(const char *data, size_t size);
    static bool isTLS(const char *data, size_t size);

    bool send(const char *data, size_t size);
    void read(const char *data, size_t size);

protected:
    virtual bool write(BIO *bio)                = 0;
    virtual void parse(char *data, size_t size) = 0;
    virtual void shutdown()                     = 0;

private:
    void read();

    BIO *m_read     = nullptr;
    BIO *m_write    = nullptr;
    bool m_ready    = false;
    SSL *m_ssl      = nullptr;
    SSL_CTX *m_ctx;
};


} // namespace xmlcore


#endif /* xmlcore_SERVERTLS_H */
