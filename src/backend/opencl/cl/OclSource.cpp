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


#include <string>
#include <regex>


#include "backend/opencl/cl/OclSource.h"
#include "crypto/common/Algorithm.h"


namespace xmrig {


static std::string cn_source;


} // namespace xmrig



const char *xmrig::OclSource::get(const Algorithm &algorithm)
{
    if (algorithm.family() == Algorithm::RANDOM_X) {
        return nullptr; // FIXME
    }

    return cn_source.c_str();
}


void xmrig::OclSource::init()
{
    const char *cryptonightCL =
        #include "./cn/cryptonight.cl"
    ;
    const char *cryptonightCL2 =
        #include "./cn/cryptonight2.cl"
    ;
    const char *blake256CL =
        #include "./cn/blake256.cl"
    ;
    const char *groestl256CL =
        #include "./cn/groestl256.cl"
    ;
    const char *jhCL =
        #include "./cn/jh.cl"
    ;
    const char *wolfAesCL =
        #include "./cn/wolf-aes.cl"
    ;
    const char *wolfSkeinCL =
        #include "./cn/wolf-skein.cl"
    ;
    const char *fastIntMathV2CL =
        #include "./cn/fast_int_math_v2.cl"
    ;
    const char *fastDivHeavyCL =
        #include "./cn/fast_div_heavy.cl"
    ;
    const char *cryptonight_gpu =
        #include "./cn/cryptonight_gpu.cl"
    ;

    cn_source.append(cryptonightCL);
    cn_source.append(cryptonightCL2);
    cn_source = std::regex_replace(cn_source, std::regex("XMRIG_INCLUDE_WOLF_AES"),         wolfAesCL);
    cn_source = std::regex_replace(cn_source, std::regex("XMRIG_INCLUDE_WOLF_SKEIN"),       wolfSkeinCL);
    cn_source = std::regex_replace(cn_source, std::regex("XMRIG_INCLUDE_JH"),               jhCL);
    cn_source = std::regex_replace(cn_source, std::regex("XMRIG_INCLUDE_BLAKE256"),         blake256CL);
    cn_source = std::regex_replace(cn_source, std::regex("XMRIG_INCLUDE_GROESTL256"),       groestl256CL);
    cn_source = std::regex_replace(cn_source, std::regex("XMRIG_INCLUDE_FAST_INT_MATH_V2"), fastIntMathV2CL);
    cn_source = std::regex_replace(cn_source, std::regex("XMRIG_INCLUDE_FAST_DIV_HEAVY"),   fastDivHeavyCL);
    cn_source = std::regex_replace(cn_source, std::regex("XMRIG_INCLUDE_CN_GPU"),           cryptonight_gpu);
}
