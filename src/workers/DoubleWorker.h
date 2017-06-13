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

#ifndef __DOUBLEWORKER_H__
#define __DOUBLEWORKER_H__


#include "align.h"
#include "net/Job.h"
#include "net/JobResult.h"
#include "workers/Worker.h"


class Handle;


class DoubleWorker : public Worker
{
public:
    DoubleWorker(Handle *handle);

    void start() override;

private:
    void consumeJob();

    Job m_job;
    uint32_t m_nonce1;
    uint32_t m_nonce2;
    uint8_t m_hash[64];
    uint8_t	m_blob[84 * 2];
};


#endif /* __SINGLEWORKER_H__ */
