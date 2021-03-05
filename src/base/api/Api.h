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

#ifndef XMRIG_API_H
#define XMRIG_API_H


#include <vector>
#include <cstdint>


#include "base/kernel/interfaces/IBaseListener.h"
#include "base/tools/String.h"


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
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(Api)

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
    char m_id[32]{};
    String m_workerId;
    const uint64_t m_timestamp;
    Httpd *m_httpd = nullptr;
    std::vector<IApiListener *> m_listeners;
};


} // namespace xmrig


#endif /* XMRIG_API_H */
