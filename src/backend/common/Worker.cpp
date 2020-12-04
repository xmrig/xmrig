/* XMRig
 * Copyright (c) 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include "backend/common/Worker.h"
#include "base/kernel/Platform.h"
#include "crypto/common/VirtualMemory.h"


xmrig::Worker::Worker(size_t id, int64_t affinity, int priority) :
    m_affinity(affinity),
    m_id(id)
{
    m_node = VirtualMemory::bindToNUMANode(affinity);

    Platform::trySetThreadAffinity(affinity);
    Platform::setThreadPriority(priority);
}
