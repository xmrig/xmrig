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

#ifndef XMRIG_BASE_H
#define XMRIG_BASE_H


#include "3rdparty/rapidjson/fwd.h"
#include "base/api/interfaces/IApiListener.h"
#include "base/kernel/interfaces/IConfigListener.h"
#include "base/kernel/interfaces/IWatcherListener.h"
#include "base/tools/Object.h"


namespace xmrig {


class Api;
class BasePrivate;
class Config;
class IBaseListener;
class Process;


class Base : public IWatcherListener, public IApiListener
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(Base)

    Base(Process *process);
    ~Base() override;

    virtual bool isReady() const;
    virtual int init();
    virtual void start();
    virtual void stop();

    Api *api() const;
    bool isBackground() const;
    bool reload(const rapidjson::Value &json);
    Config *config() const;
    void addListener(IBaseListener *listener);

protected:
    void onFileChanged(const String &fileName) override;

#   ifdef XMRIG_FEATURE_API
    void onRequest(IApiRequest &request) override;
#   endif

private:
    BasePrivate *d_ptr;
};


} /* namespace xmrig */


#endif /* XMRIG_BASE_H */
