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
 */

#include "core/Taskbar.h"

#ifdef _WIN32


#include <Shobjidl.h>
#include <Objbase.h>


namespace xmrig {


class Taskbar::Private
{
public:
    XMRIG_DISABLE_COPY_MOVE(Private)

    Private()
    {
        HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
        if (hr < 0) {
            return;
        }

        hr = CoCreateInstance(CLSID_TaskbarList, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&taskbar));
        if (hr < 0) {
            return;
        }

        hr = taskbar->HrInit();
        if (hr < 0) {
            taskbar->Release();
            taskbar = nullptr;
            return;
        }

        console = GetConsoleWindow();
    }

    ~Private()
    {
        if (taskbar) {
            taskbar->Release();
        }

        CoUninitialize();
    }

    void update()
    {
        if (taskbar) {
            if (active) {
                taskbar->SetProgressState(console, enabled ? TBPF_NOPROGRESS : TBPF_PAUSED);
                taskbar->SetProgressValue(console, enabled ? 0 : 1, 1);
            }
            else {
                taskbar->SetProgressState(console, TBPF_ERROR);
                taskbar->SetProgressValue(console, 1, 1);
            }
        }
    }

    bool active             = false;
    bool enabled            = true;
    HWND console            = nullptr;
    ITaskbarList3 *taskbar  = nullptr;
};


Taskbar::Taskbar() :
    d(std::make_shared<Private>())
{
}


void Taskbar::setActive(bool active)
{
    d->active = active;
    d->update();
}


void Taskbar::setEnabled(bool enabled)
{
    d->enabled = enabled;
    d->update();
}


} // namespace xmrig


#else // _WIN32


namespace xmrig {


Taskbar::Taskbar()              = default;
void Taskbar::setActive(bool)   {}
void Taskbar::setEnabled(bool)  {}


} // namespace xmrig


#endif // _WIN32
