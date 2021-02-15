/* xmlcore
 * Copyright (c) 2018-2019 tevador     <tevador@gmail.com>
 * Copyright (c) 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2020 xmlcore       <https://github.com/xmlcore>, <support@xmlcore.com>
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

#ifndef xmlcore_RX_BASICSTORAGE_H
#define xmlcore_RX_BASICSTORAGE_H


#include "backend/common/interfaces/IRxStorage.h"


namespace xmlcore
{


class RxBasicStoragePrivate;


class RxBasicStorage : public IRxStorage
{
public:
    xmlcore_DISABLE_COPY_MOVE(RxBasicStorage);

    RxBasicStorage();
    ~RxBasicStorage() override;

protected:
    bool isAllocated() const override;
    HugePagesInfo hugePages() const override;
    RxDataset *dataset(const Job &job, uint32_t nodeId) const override;
    void init(const RxSeed &seed, uint32_t threads, bool hugePages, bool oneGbPages, RxConfig::Mode mode, int priority) override;

private:
    RxBasicStoragePrivate *d_ptr;
};


} /* namespace xmlcore */


#endif /* xmlcore_RX_BASICSTORAGE_H */
