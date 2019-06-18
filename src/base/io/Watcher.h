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

#ifndef XMRIG_WATCHER_H
#define XMRIG_WATCHER_H


#include "base/kernel/interfaces/ITimerListener.h"
#include "base/tools/String.h"


typedef struct uv_fs_event_s uv_fs_event_t;


namespace xmrig {


class IWatcherListener;
class Timer;


class Watcher : public ITimerListener
{
public:
    Watcher(const String &path, IWatcherListener *listener);
    ~Watcher() override;

protected:
    inline void onTimer(const Timer *) override { reload(); }

private:
    constexpr static int kDelay = 500;

    static void onFsEvent(uv_fs_event_t *handle, const char *filename, int events, int status);

    void queueUpdate();
    void reload();
    void start();

    IWatcherListener *m_listener;
    String m_path;
    Timer *m_timer;
    uv_fs_event_t *m_fsEvent;
};


} /* namespace xmrig */


#endif /* XMRIG_WATCHER_H */
