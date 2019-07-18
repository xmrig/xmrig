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

#ifndef XMRIG_API_H
#define XMRIG_API_H


#include <vector>
#include <stdint.h>


#include "base/kernel/interfaces/IBaseListener.h"


namespace xmrig {


class Base;
class Httpd;
class HttpData;
class IApiListener;
class IApiRequest;
class String;


class Api : public IBaseListener
{
public:
    Api(Base *base);
    ~Api() override;

    inline const char *id() const                   { return m_id; }
    inline const char *workerId() const             { return m_workerId; }
    inline void addListener(IApiListener *listener) { m_listeners.push_back(listener); }

    void request(const HttpData &req);
    void start();
    void stop();

protected:
    void onConfigChanged(Config *config, Config *previousConfig) override;

private:
    void exec(IApiRequest &request);
    void genId(const String &id);
    void genWorkerId(const String &id);

    Base *m_base;
    char m_id[32];
    char m_workerId[128];
    Httpd *m_httpd;
    std::vector<IApiListener *> m_listeners;
    uint64_t m_timestamp;
};


} // namespace xmrig


#endif /* XMRIG_API_H */
