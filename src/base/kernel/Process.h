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

#ifndef XMRIG_PROCESS_H
#define XMRIG_PROCESS_H


#include "base/tools/Object.h"
#include "base/tools/String.h"


#ifdef WIN32
#   define XMRIG_DIR_SEPARATOR "\\"
#else
#   define XMRIG_DIR_SEPARATOR "/"
#endif


namespace xmrig {


class Arguments;
class Events;
class Versions;


class Process
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(Process)

    enum Location {
        ExePathLocation,
        ExeLocation,
        CwdLocation,
        DataLocation,
        HomeLocation,
        TempLocation
    };

    Process(int argc, char **argv);
    ~Process();

    static const Arguments &arguments();
    static const char *version();
    static const String &userAgent();
    static const Versions &versions();
    static int exitCode();
    static int pid();
    static int ppid();
    static String locate(Location location, const char *fileName);
    static String locate(Location location);
    static void exit(int code = -1);
    static void setUserAgent(const String &userAgent);

#   ifndef XMRIG_LEGACY
    static Events &events();
#   endif

private:
    class Private;

    static Private *d_fn();

    static Private *d;
};


} // namespace xmrig


#endif // XMRIG_PROCESS_H
