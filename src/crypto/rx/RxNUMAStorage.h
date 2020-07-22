/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018-2019 tevador     <tevador@gmail.com>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_RX_NUMASTORAGE_H
#define XMRIG_RX_NUMASTORAGE_H


#include "backend/common/interfaces/IRxStorage.h"
#include "base/tools/Object.h"


#include <vector>


namespace xmrig
{


class RxNUMAStoragePrivate;


class RxNUMAStorage : public IRxStorage
{
public:
    XMRIG_DISABLE_COPY_MOVE(RxNUMAStorage);

    RxNUMAStorage(const std::vector<uint32_t> &nodeset);
    ~RxNUMAStorage() override;

protected:
    bool isAllocated() const override;
    HugePagesInfo hugePages() const override;
    RxDataset *dataset(const Job &job, uint32_t nodeId) const override;
    void init(const RxSeed &seed, uint32_t threads, bool hugePages, bool oneGbPages, RxConfig::Mode mode, int priority) override;

private:
    RxNUMAStoragePrivate *d_ptr;
};


} /* namespace xmrig */


#endif /* XMRIG_RX_NUMASTORAGE_H */
