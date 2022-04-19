/* XMRig
 * Copyright (c) 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright (c) 2018-2022 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2022 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_TLSCONTEXT_H
#define XMRIG_TLSCONTEXT_H


#include "base/tools/Object.h"


using SSL_CTX = struct ssl_ctx_st;


namespace xmrig {


class TlsConfig;


class TlsContext
{
public:
    XMRIG_DISABLE_COPY_MOVE(TlsContext)

    ~TlsContext();

    static std::shared_ptr<TlsContext> create(const TlsConfig &config);

    SSL_CTX *handle() const;

private:
    XMRIG_DECL_PRIVATE()

    TlsContext();
};


} // namespace xmrig


#endif // XMRIG_TLSCONTEXT_H
