/* XMRigCC
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
 * Copyright 2017-     BenDr0id    <ben@graef.in>
 *
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

#ifndef __CC_SERVER_H__
#define __CC_SERVER_H__


#include <uv.h>


#include "interfaces/IConsoleListener.h"


class Console;
class Httpd;
class Options;

class CCServer : public IConsoleListener
{
public:
  CCServer(int argc, char **argv);
  ~CCServer();

  int start();

protected:
  void onConsoleCommand(char command) override;

private:
  void stop();
  void printCommands();

  static void onSignal(uv_signal_t *handle, int signum);

  static CCServer *m_self;

  Console *m_console;
  Httpd *m_httpd;
  Options *m_options;
  uv_signal_t m_signal;
};


#endif /* __CC_SERVER_H__ */
