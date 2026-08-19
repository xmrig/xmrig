/* XMRig
 * Copyright (c) 2016-2025 XMRig         <support@xmrig.com>
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

#ifndef XMRIG_RISCV_VLEN_H
#define XMRIG_RISCV_VLEN_H


#include <cstdint>


namespace xmrig {


/*
 * Vector register length (VLEN) in bits for the *calling thread*, or 0 when the
 * V extension is not available.
 *
 * The RVV kernels use a fixed application vector length (AVL) with LMUL=1, so
 * they need VLMAX to be at least that big. When VLEN is too small the hardware
 * silently clamps vl and the kernels compute wrong results instead of faulting,
 * which is why every RVV dispatch has to consult this.
 *
 * VLEN is a per-hart property and RISC-V does not require it to be uniform
 * across a system: the SpacemiT K3 pairs 256-bit X100 cores with 1024-bit A100
 * cores, and a thread's VLEN depends on which cluster it is pinned to. The
 * value is therefore cached per thread, not globally.
 */
uint32_t riscv_vlen();


// Smallest VLEN the RVV code paths can run on.
constexpr uint32_t RISCV_MIN_VLEN = 256;


} // namespace xmrig


#endif /* XMRIG_RISCV_VLEN_H */
