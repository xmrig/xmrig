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

#ifndef __CPUTHREAD_H__
#define __CPUTHREAD_H__


#include "interfaces/IThread.h"
#include "xmrig.h"


struct cryptonight_ctx;


namespace xmrig {


class CpuThread : public IThread
{
public:
    enum Multiway {
        SingleWay = 1,
        DoubleWay,
        TripleWay,
        QuadWay,
        PentaWay
    };

    CpuThread(size_t index, Algo algorithm, AlgoVariant av, Multiway multiway, int64_t affinity, int priority, bool softAES, bool prefetch);
    ~CpuThread();

    typedef void (*cn_hash_fun)(const uint8_t *input, size_t size, uint8_t *output, cryptonight_ctx *ctx);

    static cn_hash_fun fn(Algo algorithm, AlgoVariant av, Variant variant);
    static CpuThread *createFromAV(size_t index, Algo algorithm, AlgoVariant av, int64_t affinity, int priority);

    inline bool isPrefetch() const               { return m_prefetch; }
    inline bool isSoftAES() const                { return m_softAES; }
    inline cn_hash_fun fn(Variant variant) const { return fn(m_algorithm, m_av, variant); }

    inline Algo algorithm() const override       { return m_algorithm; }
    inline int multiway() const override         { return m_multiway; }
    inline int priority() const override         { return m_priority; }
    inline int64_t affinity() const override     { return m_affinity; }
    inline size_t index() const override         { return m_index; }
    inline Type type() const override            { return CPU; }

#   ifndef XMRIG_NO_API
    rapidjson::Value toAPI(rapidjson::Document &doc) const override;
#   endif

private:
    const Algo m_algorithm;
    const AlgoVariant m_av;
    const bool m_prefetch;
    const bool m_softAES;
    const int m_priority;
    const int64_t m_affinity;
    const Multiway m_multiway;
    const size_t m_index;
};


} /* namespace xmrig */


#endif /* __CPUTHREAD_H__ */
