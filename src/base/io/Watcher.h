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

#ifndef XMRIG_WATCHER_H
#define XMRIG_WATCHER_H


#include "base/kernel/interfaces/ITimerListener.h"
#include "base/tools/String.h"


#include <memory>


using uv_fs_event_t = struct uv_fs_event_s;


namespace xmrig {


class IWatcherListener;
class Timer;


class Watcher : public ITimerListener
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(Watcher)

    Watcher(const String &path, IWatcherListener *listener);
    ~Watcher() override;

protected:
    void onTimer(const Timer *timer) override;

private:
    constexpr static int kDelay = 500;

    static void onFsEvent(uv_fs_event_t *handle, const char *filename, int events, int status);

    void reload();
    void start();
    void startTimer();
    void stop();

    const String m_path;
    IWatcherListener *m_listener;
    std::shared_ptr<Timer> m_timer;
    uv_fs_event_t *m_event  = nullptr;
};


} // namespace xmrig


#endif // XMRIG_WATCHER_H
