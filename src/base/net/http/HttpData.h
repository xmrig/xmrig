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


#ifndef XMRIG_HTTPDATA_H
#define XMRIG_HTTPDATA_H


#include "base/tools/Object.h"


#include <map>
#include <string>


namespace xmrig {


class HttpData
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(HttpData)

    static const std::string kApplicationJson;
    static const std::string kContentType;
    static const std::string kContentTypeL;
    static const std::string kTextPlain;


    inline HttpData(uint64_t id) : m_id(id) {}
    virtual ~HttpData() = default;

    inline uint64_t id() const  { return m_id; }

    virtual bool isRequest() const                      = 0;
    virtual const char *host() const                    = 0;
    virtual const char *tlsFingerprint() const          = 0;
    virtual const char *tlsVersion() const              = 0;
    virtual std::string ip() const                      = 0;
    virtual uint16_t port() const                       = 0;
    virtual void write(std::string &&data, bool close)  = 0;

    bool isJSON() const;

    int method      = 0;
    int status      = 0;
    int userType    = 0;
    std::map<const std::string, const std::string> headers;
    std::string body;
    std::string url;

private:
    const uint64_t m_id;
};


} // namespace xmrig


#endif // XMRIG_HTTPDATA_H

