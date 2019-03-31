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


#ifndef XMRIG_APIREQUEST_H
#define XMRIG_APIREQUEST_H


#include "api/interfaces/IApiRequest.h"


namespace xmrig {


class ApiRequest : public IApiRequest
{
public:
    ApiRequest(Source source, bool restricted);
    ~ApiRequest() override;

protected:
    inline bool isDone() const override          { return m_state == STATE_DONE; }
    inline bool isNew() const override           { return m_state == STATE_NEW; }
    inline bool isRestricted() const override    { return m_restricted; }
    inline Source source() const override        { return m_source; }
    inline void accept() override                { m_state = STATE_ACCEPTED; }
    inline void done(int) override               { m_state = STATE_DONE; }

private:
    enum State {
        STATE_NEW,
        STATE_ACCEPTED,
        STATE_DONE
    };

    bool m_restricted;
    Source m_source;
    State m_state;
};


} // namespace xmrig


#endif // XMRIG_APIREQUEST_H

