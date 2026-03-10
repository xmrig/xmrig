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

#include "base/kernel/interfaces/IDnsBackend.h"
#include "base/net/dns/DnsRecords.h"
#include "base/net/tools/Storage.h"

#include <deque>


using uv_getaddrinfo_t = struct uv_getaddrinfo_s;


namespace xmrig {


class DnsUvBackend : public IDnsBackend
{
public:
    XMRIG_DISABLE_COPY_MOVE(DnsUvBackend)

    DnsUvBackend();
    ~DnsUvBackend() override;

protected:
    void resolve(const String &host, const std::weak_ptr<IDnsListener> &listener, const DnsConfig &config) override;

private:
    bool resolve(const String &host);
    void notify();
    void onResolved(int status, addrinfo *res);

    static void onResolved(uv_getaddrinfo_t *req, int status, addrinfo *res);

    DnsRecords m_records;
    int m_ai_family         = 0;
    int m_status            = 0;
    std::deque<std::weak_ptr<IDnsListener>> m_queue;
    std::shared_ptr<uv_getaddrinfo_t> m_req;
    uint64_t m_ts           = 0;
    uintptr_t m_key;

    static Storage<DnsUvBackend>& getStorage();
    void releaseStorage();
};


} // namespace xmrig
