/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2016-2018 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include <common/log/Log.h>
#include "workers/Handle.h"


Handle::Handle(int id, xmrig::Config *config, xmrig::HasherConfig *hasherConfig, uint32_t offset) :
        m_id(id),
        m_offset(offset),
        m_config(config),
        m_hasherConfig(hasherConfig),
        m_hasher(nullptr)
{
    std::vector<Hasher *> hashers = Hasher::getHashers();
    for(Hasher *hasher : hashers) {
        if(hasherConfig->type() == hasher->subType()) {
            if(hasher->initialize(hasherConfig->algorithm(), hasherConfig->variant()) &&
                hasher->configure(*hasherConfig) &&
                hasher->deviceCount() > 0)
                m_hasher = hasher;

            std::string hasherInfo = hasher->info();

            if(config->isColors()) {
                std::string redDisabled = RED_BOLD("DISABLED");
                std::string greenEnabled = GREEN_BOLD("ENABLED");

                size_t startPos = hasherInfo.find("DISABLED");
                while (startPos != string::npos) {
                    hasherInfo.replace(startPos, 8, redDisabled);
                    startPos = hasherInfo.find("DISABLED", startPos + redDisabled.size());
                }

                startPos = hasherInfo.find("ENABLED");
                while (startPos != string::npos) {
                    hasherInfo.replace(startPos, 7, greenEnabled);
                    startPos = hasherInfo.find("ENABLED", startPos + greenEnabled.size());
                }

                Log::i()->text(GREEN_BOLD(" * Initializing %s hasher:") "\n%s", hasher->subType().c_str(), hasherInfo.c_str());
            }
            else {
                Log::i()->text(" * Initializing %s hasher:\n%s", hasher->subType().c_str(), hasherInfo.c_str());
            }
        }
    }
}

void Handle::join()
{
    for(uv_thread_t thread : m_threads)
        uv_thread_join(&thread);
}


void Handle::start(void (*callback) (void *))
{
    assert(m_hasher != nullptr);
    for(int i=0; i < m_hasher->computingThreads(); i++) {
        uv_thread_t thread;
        HandleArg *arg = new HandleArg { this, i };
        uv_thread_create(&thread, callback, arg);
        m_threads.push_back(thread);
    }
}
