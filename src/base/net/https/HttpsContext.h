/* XMRig
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


#ifndef XMRIG_HTTPSCONTEXT_H
#define XMRIG_HTTPSCONTEXT_H


using BIO = struct bio_st;
using SSL = struct ssl_st;


#include "base/net/http/HttpContext.h"
#include "base/net/tls/ServerTls.h"


namespace xmrig {


class TlsContext;


class HttpsContext : public HttpContext, public ServerTls
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(HttpsContext)

    HttpsContext(TlsContext *tls, const std::weak_ptr<IHttpListener> &listener);
    ~HttpsContext() override;

    void append(char *data, size_t size);

protected:
    // ServerTls
    bool write(BIO *bio) override;
    void parse(char *data, size_t size) override;
    void shutdown() override;

    // HttpContext
    void write(std::string &&data, bool close) override;

private:
    enum TlsMode : uint32_t {
      TLS_AUTO,
      TLS_OFF,
      TLS_ON
    };

    bool m_close    = false;
    TlsMode m_mode  = TLS_AUTO;
};


} // namespace xmrig


#endif // XMRIG_HTTPSCONTEXT_H

