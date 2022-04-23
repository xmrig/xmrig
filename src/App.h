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

#ifndef XMRIG_APP_H
#define XMRIG_APP_H


#include "base/kernel/interfaces/IConsoleListener.h"
#include "base/kernel/interfaces/ISignalListener.h"
#include "base/tools/Object.h"


#include <memory>


namespace xmrig {


class Console;
class Controller;
class Network;
class Process;
class Signals;


class App : public IConsoleListener, public ISignalListener
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(App)

    App(Process *process);
    ~App() override;

    int exec();

protected:
    void onConsoleCommand(char command) override;
    void onSignal(int signum) override;

private:
    void close();

    std::shared_ptr<Console> m_console;
    std::shared_ptr<Controller> m_controller;
    std::shared_ptr<Signals> m_signals;
};


} /* namespace xmrig */


#endif /* XMRIG_APP_H */
