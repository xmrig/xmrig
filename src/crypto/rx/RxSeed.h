/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018-2019 tevador     <tevador@gmail.com>
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

#ifndef XMRIG_RX_SEED_H
#define XMRIG_RX_SEED_H


#include "base/net/stratum/Job.h"
#include "base/tools/Buffer.h"


namespace xmrig
{


class RxSeed;


class RxSeed
{
public:
    RxSeed() = default;

    inline RxSeed(const Algorithm &algorithm, const Buffer &seed) : m_algorithm(algorithm), m_data(seed)    {}
    inline RxSeed(const Job &job) : m_algorithm(job.algorithm()), m_data(job.seed())                        {}

    inline bool isEqual(const Job &job) const           { return m_algorithm == job.algorithm() && m_data == job.seed(); }
    inline bool isEqual(const RxSeed &other) const      { return m_algorithm == other.m_algorithm && m_data == other.m_data; }
    inline const Algorithm &algorithm() const           { return m_algorithm; }
    inline const Buffer &data() const                   { return m_data; }

    inline bool operator!=(const Job &job) const        { return !isEqual(job); }
    inline bool operator!=(const RxSeed &other) const   { return !isEqual(other); }
    inline bool operator==(const Job &job) const        { return isEqual(job); }
    inline bool operator==(const RxSeed &other) const   { return isEqual(other); }

private:
    Algorithm m_algorithm;
    Buffer m_data;
};


} /* namespace xmrig */


#endif /* XMRIG_RX_CACHE_H */
