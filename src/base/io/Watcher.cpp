/* XMRig
 * Copyright (c) 2018-2022 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2022 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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
 *
  * Additional permission under GNU GPL version 3 section 7
  *
  * If you modify this Program, or any covered work, by linking or combining
  * it with OpenSSL (or a modified version of that library), containing parts
  * covered by the terms of OpenSSL License and SSLeay License, the licensors
  * of this Program grant you additional permission to convey the resulting work.
 */

#include "base/io/Watcher.h"
#include "base/kernel/interfaces/IWatcherListener.h"
#include "base/tools/Handle.h"
#include "base/tools/Timer.h"


xmrig::Watcher::Watcher(const String &path, IWatcherListener *listener) :
    m_path(path),
    m_listener(listener)
{
    m_timer = std::make_shared<Timer>(this);

    startTimer();
}


xmrig::Watcher::~Watcher()
{
    Handle::close(m_event);
}


void xmrig::Watcher::onTimer(const Timer * /*timer*/)
{
    if (m_event) {
        reload();
    }
    else {
        start();
    }
}


void xmrig::Watcher::onFsEvent(uv_fs_event_t *handle, const char *filename, int, int)
{
    if (!filename) {
        return;
    }

    static_cast<Watcher *>(handle->data)->startTimer();
}


void xmrig::Watcher::reload()
{
    m_listener->onFileChanged(m_path);

#   ifndef XMRIG_OS_WIN
    stop();
    start();
#   endif
}


void xmrig::Watcher::start()
{
    if (!m_event) {
        m_event = new uv_fs_event_t;
        m_event->data = this;
        uv_fs_event_init(uv_default_loop(), m_event);
    }

    uv_fs_event_start(m_event, onFsEvent, m_path, 0);
}


void xmrig::Watcher::startTimer()
{
    m_timer->singleShot(kDelay);
}


void xmrig::Watcher::stop()
{
    uv_fs_event_stop(m_event);
}
