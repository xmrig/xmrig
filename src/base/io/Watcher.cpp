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


#include <uv.h>


#include "base/kernel/interfaces/IWatcherListener.h"
#include "base/io/Watcher.h"
#include "base/tools/Handle.h"


xmrig::Watcher::Watcher(const String &path, IWatcherListener *listener) :
    m_listener(listener),
    m_path(path)
{
    m_fsEvent = new uv_fs_event_t;
    uv_fs_event_init(uv_default_loop(), m_fsEvent);

    m_timer = new uv_timer_t;
    uv_timer_init(uv_default_loop(), m_timer);

    m_fsEvent->data = m_timer->data = this;

    start();
}


xmrig::Watcher::~Watcher()
{
    Handle::close(m_timer);
    Handle::close(m_fsEvent);
}


void xmrig::Watcher::onTimer(uv_timer_t *handle)
{
    static_cast<Watcher *>(handle->data)->reload();
}


void xmrig::Watcher::onFsEvent(uv_fs_event_t *handle, const char *filename, int, int)
{
    if (!filename) {
        return;
    }

    static_cast<Watcher *>(handle->data)->queueUpdate();
}


void xmrig::Watcher::queueUpdate()
{
    uv_timer_stop(m_timer);
    uv_timer_start(m_timer, xmrig::Watcher::onTimer, kDelay, 0);
}


void xmrig::Watcher::reload()
{
    m_listener->onFileChanged(m_path);

#   ifndef _WIN32
    uv_fs_event_stop(m_fsEvent);
    start();
#   endif
}


void xmrig::Watcher::start()
{
    uv_fs_event_start(m_fsEvent, xmrig::Watcher::onFsEvent, m_path, 0);
}
