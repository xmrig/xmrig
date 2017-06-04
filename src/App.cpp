/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
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


#include <uv.h>


#include "App.h"
#include "Console.h"
#include "net/Client.h"
#include "net/Network.h"
#include "Options.h"
#include "version.h"


Client *client;
uv_timer_t timer_req;


void timer_cb(uv_timer_t* handle) {
    LOG_DEBUG("TIMER");

    client->disconnect();
}


App::App(int argc, char **argv)
{
    Console::init();
    m_options = Options::parse(argc, argv);

    m_network = new Network(m_options);
}


App::~App()
{
    LOG_DEBUG("~APP");

    free(m_network);
    free(m_options);
}


App::exec()
{
    if (!m_options->isReady()) {
        return 0;
    }

    m_network->connect();

//    uv_timer_init(uv_default_loop(), &timer_req);
//    uv_timer_start(&timer_req, timer_cb, 5000, 5000);


//    client = new Client();
//    client->connect("192.168.2.34", 3333);

    const int r = uv_run(uv_default_loop(), UV_RUN_DEFAULT);
    uv_loop_close(uv_default_loop());

    return r;
}
