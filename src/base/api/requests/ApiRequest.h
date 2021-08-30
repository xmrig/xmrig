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

#ifndef XMRIG_APIREQUEST_H
#define XMRIG_APIREQUEST_H


#include "base/api/interfaces/IApiRequest.h"
#include "base/tools/String.h"
#include "base/tools/Object.h"


namespace xmrig {


class ApiRequest : public IApiRequest
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(ApiRequest)

    ApiRequest(Source source, bool restricted);
    ~ApiRequest() override;

protected:
    enum State {
        STATE_NEW,
        STATE_ACCEPTED,
        STATE_DONE
    };

    inline bool accept() override                   { m_state = STATE_ACCEPTED; return true; }
    inline bool isDone() const override             { return m_state == STATE_DONE; }
    inline bool isNew() const override              { return m_state == STATE_NEW; }
    inline bool isRestricted() const override       { return m_restricted; }
    inline const String &rpcMethod() const override { return m_rpcMethod; }
    inline int version() const override             { return m_version; }
    inline RequestType type() const override        { return m_type; }
    inline Source source() const override           { return m_source; }
    inline void done(int) override                  { m_state = STATE_DONE; }

    int m_version       = 1;
    RequestType m_type  = REQ_UNKNOWN;
    State m_state       = STATE_NEW;
    String m_rpcMethod;

private:
    const bool m_restricted;
    const Source m_source;
};


} // namespace xmrig


#endif // XMRIG_APIREQUEST_H

