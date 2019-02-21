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


#include "base/io/Watcher.h"
#include "base/kernel/interfaces/IConfigListener.h"
#include "common/config/ConfigLoader.h"
#include "common/config/ConfigWatcher.h"
#include "common/log/Log.h"
#include "core/ConfigCreator.h"


xmrig::ConfigWatcher::ConfigWatcher(const String &path, IConfigCreator *creator, IConfigListener *listener) :
    m_creator(creator),
    m_listener(listener)
{
   m_watcher = new Watcher(path, this);
}


xmrig::ConfigWatcher::~ConfigWatcher()
{
    delete m_watcher;
}



void xmrig::ConfigWatcher::onFileChanged(const String &fileName)
{
    LOG_WARN("\"%s\" was changed, reloading configuration", fileName.data());

    IConfig *config = m_creator->create();
    ConfigLoader::loadFromFile(config, fileName);

    if (!config->finalize()) {
        LOG_ERR("reloading failed");

        delete config;
        return;
    }

    m_listener->onNewConfig(config);
}
