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

#ifndef __HASH_SELECTOR_H__
#define __HASH_SELECTOR_H__


#include <stddef.h>
#include <stdint.h>

#include "CryptoNight.h"
#include "AsmOptimization.h"
#include "Options.h"

class Job;
class JobResult;

class HashSelector
{
public:
    static bool init(Options::Algo algo, bool aesni);
    static void hash(size_t factor, AsmOptimization asmOptimization, uint64_t height, PowVariant powVersion, const uint8_t* input, size_t size, uint8_t* output, ScratchPad** scratchPads);

public:

private:
    static bool selfCheck(Options::Algo algo);
};


#endif /* __HASH_SELECTOR_H__ */
