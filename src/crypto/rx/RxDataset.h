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

#ifndef XMRIG_RX_DATASET_H
#define XMRIG_RX_DATASET_H


#include "crypto/common/Algorithm.h"
#include "crypto/randomx/configuration.h"


struct randomx_dataset;


namespace xmrig
{


class RxCache;


class RxDataset
{
public:
    RxDataset(bool hugePages = true);
    ~RxDataset();

    inline bool isHugePages() const     { return m_flags & 1; }
    inline randomx_dataset *get() const { return m_dataset; }
    inline RxCache *cache() const       { return m_cache; }

    bool init(const void *seed, const Algorithm &algorithm, uint32_t numThreads);
    bool isReady(const void *seed, const Algorithm &algorithm) const;
    std::pair<size_t, size_t> hugePages() const;

    static inline constexpr size_t size() { return RANDOMX_DATASET_MAX_SIZE; }

private:
    Algorithm m_algorithm;
    int m_flags                = 0;
    randomx_dataset *m_dataset = nullptr;
    RxCache *m_cache           = nullptr;
};


} /* namespace xmrig */


#endif /* XMRIG_RX_DATASET_H */
