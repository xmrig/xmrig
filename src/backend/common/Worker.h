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

#ifndef XMRIG_WORKER_H
#define XMRIG_WORKER_H


#include "backend/common/interfaces/IWorker.h"


namespace xmrig {


class Worker : public IWorker
{
public:
    Worker(size_t id, int64_t affinity, int priority);

    size_t threads() const override                         { return 1; }

protected:
    inline int64_t affinity() const                         { return m_affinity; }
    inline size_t id() const override                       { return m_id; }
    inline uint32_t node() const                            { return m_node; }

    uint64_t m_count                = 0;

private:
    const int64_t m_affinity;
    const size_t m_id;
    uint32_t m_node                 = 0;
};


} // namespace xmrig


#endif /* XMRIG_WORKER_H */
