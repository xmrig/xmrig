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

#ifndef XMRIG_ICLIENTLISTENER_H
#define XMRIG_ICLIENTLISTENER_H


#include <cstdint>


#include "3rdparty/rapidjson/fwd.h"
#include "base/tools/Object.h"


namespace xmrig {


class Algorithm;
class IClient;
class Job;
class SubmitResult;


class IClientListener
{
public:
    XMRIG_DISABLE_COPY_MOVE(IClientListener)

    IClientListener()           = default;
    virtual ~IClientListener()  = default;

    virtual void onClose(IClient *client, int failures)                                           = 0;
    virtual void onJobReceived(IClient *client, const Job &job, const rapidjson::Value &params)   = 0;
    virtual void onLogin(IClient *client, rapidjson::Document &doc, rapidjson::Value &params)     = 0;
    virtual void onLoginSuccess(IClient *client)                                                  = 0;
    virtual void onResultAccepted(IClient *client, const SubmitResult &result, const char *error) = 0;
    virtual void onVerifyAlgorithm(const IClient *client, const Algorithm &algorithm, bool *ok)   = 0;
};


} /* namespace xmrig */


#endif // XMRIG_ICLIENTLISTENER_H
