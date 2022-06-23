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


#include "backend/opencl/cl/OclSource.h"
#include "backend/opencl/cl/cn/cryptonight_cl.h"
#include "base/crypto/Algorithm.h"


#ifdef XMRIG_ALGO_CN_GPU
#   include "backend/opencl/cl/cn/cryptonight_gpu_cl.h"
#endif

#ifdef XMRIG_ALGO_RANDOMX
#   include "backend/opencl/cl/rx/randomx_cl.h"
#endif

#ifdef XMRIG_ALGO_KAWPOW
#   include "backend/opencl/cl/kawpow/kawpow_cl.h"
#   include "backend/opencl/cl/kawpow/kawpow_dag_cl.h"
#endif


const char *xmrig::OclSource::get(const Algorithm &algorithm)
{
#   ifdef XMRIG_ALGO_CN_GPU
    if (algorithm == Algorithm::CN_GPU) {
        return cryptonight_gpu_cl;
    }
#   endif

#   ifdef XMRIG_ALGO_RANDOMX
    if (algorithm.family() == Algorithm::RANDOM_X) {
        return randomx_cl;
    }
#   endif

#   ifdef XMRIG_ALGO_KAWPOW
    if (algorithm.family() == Algorithm::KAWPOW) {
        return kawpow_dag_cl;
    }
#   endif

    return cryptonight_cl;
}
