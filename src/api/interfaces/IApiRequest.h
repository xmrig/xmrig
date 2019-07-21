/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
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

#ifndef XMRIG_IAPIREQUEST_H
#define XMRIG_IAPIREQUEST_H


#include "rapidjson/fwd.h"


namespace xmrig {


class String;


class IApiRequest
{
public:
    enum Method {
        METHOD_DELETE,
        METHOD_GET,
        METHOD_HEAD,
        METHOD_POST,
        METHOD_PUT
    };


    enum Source {
        SOURCE_HTTP
    };


    enum RequestType {
        REQ_UNKNOWN,
        REQ_SUMMARY
    };


    virtual ~IApiRequest() = default;

    virtual bool isDone() const                                  = 0;
    virtual bool isNew() const                                   = 0;
    virtual bool isRestricted() const                            = 0;
    virtual const rapidjson::Value &json() const                 = 0;
    virtual const String &url() const                            = 0;
    virtual int version() const                                  = 0;
    virtual Method method() const                                = 0;
    virtual rapidjson::Document &doc()                           = 0;
    virtual rapidjson::Value &reply()                            = 0;
    virtual RequestType type() const                             = 0;
    virtual Source source() const                                = 0;
    virtual void accept()                                        = 0;
    virtual void done(int status)                                = 0;
};


} /* namespace xmrig */


#endif // XMRIG_IAPIREQUEST_H
