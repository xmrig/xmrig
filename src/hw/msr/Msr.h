/* xmlcore
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 xmlcore       <https://github.com/xmlcore>, <support@xmlcore.com>
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

#ifndef xmlcore_MSR_H
#define xmlcore_MSR_H


#include "base/tools/Object.h"
#include "hw/msr/MsrItem.h"


#include <functional>
#include <memory>


namespace xmlcore
{


class MsrPrivate;


class Msr
{
public:
    xmlcore_DISABLE_COPY_MOVE(Msr)

    using Callback = std::function<bool(int32_t cpu)>;

    Msr();
    ~Msr();

    static const char *tag();
    static std::shared_ptr<Msr> get();

    inline bool write(const MsrItem &item, int32_t cpu = -1, bool verbose = true)   { return write(item.reg(), item.value(), cpu, item.mask(), verbose); }

    bool isAvailable() const;
    bool write(uint32_t reg, uint64_t value, int32_t cpu = -1, uint64_t mask = MsrItem::kNoMask, bool verbose = true);
    bool write(Callback &&callback);
    MsrItem read(uint32_t reg, int32_t cpu = -1, bool verbose = true) const;

private:
    bool rdmsr(uint32_t reg, int32_t cpu, uint64_t &value) const;
    bool wrmsr(uint32_t reg, uint64_t value, int32_t cpu);

    MsrPrivate *d_ptr = nullptr;
};


} /* namespace xmlcore */


#endif /* xmlcore_MSR_H */
