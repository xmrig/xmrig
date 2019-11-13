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

#ifndef XMRIG_SUBMITRESULT_H
#define XMRIG_SUBMITRESULT_H


#include "base/tools/Chrono.h"


namespace xmrig {


class SubmitResult
{
public:
    SubmitResult() = default;

    inline SubmitResult(int64_t seq, uint64_t diff, uint64_t actualDiff, int64_t reqId, uint32_t backend) :
        reqId(reqId),
        seq(seq),
        backend(backend),
        actualDiff(actualDiff),
        diff(diff),
        m_start(Chrono::steadyMSecs())
    {}

    inline void done() { elapsed = Chrono::steadyMSecs() - m_start; }

    int64_t reqId           = 0;
    int64_t seq             = 0;
    uint32_t backend        = 0;
    uint64_t actualDiff     = 0;
    uint64_t diff           = 0;
    uint64_t elapsed        = 0;

private:
    uint64_t m_start        = 0;
};


} /* namespace xmrig */


#endif /* XMRIG_SUBMITRESULT_H */
