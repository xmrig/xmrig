/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
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


#include "3rdparty/argon2.h"
#include "base/tools/String.h"
#include "crypto/argon2/Impl.h"



namespace xmrig {


static bool selected = false;
static String implName;


} // namespace xmrig


extern "C" {


extern int xmrig_ar2_check_avx512f();
extern int xmrig_ar2_check_avx2();
extern int xmrig_ar2_check_ssse3();
extern int xmrig_ar2_check_sse2();


}


bool xmrig::argon2::Impl::select(const String &nameHint, bool benchmark)
{
    if (!selected) {
#       if defined(__x86_64__) || defined(_M_AMD64)
        auto hint = nameHint;

        if (hint.isEmpty() && !benchmark) {
            if (xmrig_ar2_check_avx512f()) {
                hint = "AVX-512F";
            }
            else if (xmrig_ar2_check_avx2()) {
                hint = "AVX2";
            }
            else if (xmrig_ar2_check_ssse3()) {
                hint = "SSSE3";
            }
            else if (xmrig_ar2_check_sse2()) {
                hint = "SSE2";
            }
        }

        if (hint.isEmpty() || argon2_select_impl_by_name(hint) == 0) {
            argon2_select_impl();
        }
#       endif

        selected = true;
        implName = argon2_get_impl_name();

        return true;
    }

    return false;
}


const xmrig::String &xmrig::argon2::Impl::name()
{
    return implName;
}
