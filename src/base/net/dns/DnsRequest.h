/* XMRig
 * Copyright (c) 2018-2025 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2025 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#pragma once

#include "base/kernel/interfaces/IDnsListener.h"


namespace xmrig {


class DnsRequest : public IDnsListener
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(DnsRequest)

    inline DnsRequest(IDnsListener *listener) : m_listener(listener) {}
    ~DnsRequest() override = default;

protected:
    inline void onResolved(const DnsRecords &records, int status, const char *error) override {
        m_listener->onResolved(records, status, error);
    }

private:
    IDnsListener *m_listener;
};


} // namespace xmrig
