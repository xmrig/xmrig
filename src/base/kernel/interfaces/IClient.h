/* XMRig
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_ICLIENT_H
#define XMRIG_ICLIENT_H


#include "3rdparty/rapidjson/fwd.h"
#include "base/tools/Object.h"


#include <functional>


namespace xmrig {


class Algorithm;
class Job;
class JobResult;
class Pool;
class ProxyUrl;
class String;


class IClient
{
public:
    XMRIG_DISABLE_COPY_MOVE(IClient)

    enum Extension {
        EXT_ALGO,
        EXT_NICEHASH,
        EXT_CONNECT,
        EXT_TLS,
        EXT_KEEPALIVE,
        EXT_MAX
    };

    using Callback = std::function<void(const rapidjson::Value &result, bool success, uint64_t elapsed)>;

    IClient()           = default;
    virtual ~IClient()  = default;

    virtual bool disconnect()                                               = 0;
    virtual bool hasExtension(Extension extension) const noexcept           = 0;
    virtual bool isEnabled() const                                          = 0;
    virtual bool isTLS() const                                              = 0;
    virtual bool isWSS() const                                              = 0;
    virtual const char *mode() const                                        = 0;
    virtual const char *tag() const                                         = 0;
    virtual const char *tlsFingerprint() const                              = 0;
    virtual const char *tlsVersion() const                                  = 0;
    virtual const Job &job() const                                          = 0;
    virtual const Pool &pool() const                                        = 0;
    virtual const String &ip() const                                        = 0;
    virtual int id() const                                                  = 0;
    virtual int64_t send(const rapidjson::Value &obj, Callback callback)    = 0;
    virtual int64_t send(const rapidjson::Value &obj)                       = 0;
    virtual int64_t sequence() const                                        = 0;
    virtual int64_t submit(const JobResult &result)                         = 0;
    virtual void connect()                                                  = 0;
    virtual void connect(const Pool &pool)                                  = 0;
    virtual void deleteLater()                                              = 0;
    virtual void setAlgo(const Algorithm &algo)                             = 0;
    virtual void setEnabled(bool enabled)                                   = 0;
    virtual void setPool(const Pool &pool)                                  = 0;
    virtual void setProxy(const ProxyUrl &proxy)                            = 0;
    virtual void setQuiet(bool quiet)                                       = 0;
    virtual void setRetries(int retries)                                    = 0;
    virtual void setRetryPause(uint64_t ms)                                 = 0;
    virtual void tick(uint64_t now)                                         = 0;
};


} /* namespace xmrig */


#endif // XMRIG_ICLIENT_H
