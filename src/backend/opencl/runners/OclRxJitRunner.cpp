/* XMRig
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#include <stdexcept>


#include "backend/opencl/runners/OclRxJitRunner.h"
#include "backend/opencl/cl/rx/randomx_run_gfx803.h"
#include "backend/opencl/cl/rx/randomx_run_gfx900.h"
#include "backend/opencl/cl/rx/randomx_run_gfx1010.h"
#include "backend/opencl/kernels/rx/Blake2bHashRegistersKernel.h"
#include "backend/opencl/kernels/rx/HashAesKernel.h"
#include "backend/opencl/kernels/rx/RxJitKernel.h"
#include "backend/opencl/kernels/rx/RxRunKernel.h"
#include "backend/opencl/OclLaunchData.h"
#include "backend/opencl/wrappers/OclLib.h"
#include "backend/opencl/wrappers/OclError.h"


xmrig::OclRxJitRunner::OclRxJitRunner(size_t index, const OclLaunchData &data) : OclRxBaseRunner(index, data)
{
}


xmrig::OclRxJitRunner::~OclRxJitRunner()
{
    delete m_randomx_jit;
    delete m_randomx_run;

    OclLib::release(m_asmProgram);
    OclLib::release(m_intermediate_programs);
    OclLib::release(m_programs);
    OclLib::release(m_registers);
}


size_t xmrig::OclRxJitRunner::bufferSize() const
{
    return OclRxBaseRunner::bufferSize() + align(256 * m_intensity) + align(5120 * m_intensity) + align(10048 * m_intensity);
}


void xmrig::OclRxJitRunner::build()
{
    OclRxBaseRunner::build();

    m_hashAes1Rx4->setArgs(m_scratchpads, m_registers, 256, m_intensity);
    m_blake2b_hash_registers_32->setArgs(m_hashes, m_registers, 256);
    m_blake2b_hash_registers_64->setArgs(m_hashes, m_registers, 256);

    m_randomx_jit = new RxJitKernel(m_program);
    m_randomx_jit->setArgs(m_entropy, m_registers, m_intermediate_programs, m_programs, m_intensity, m_rounding);

    if (!loadAsmProgram()) {
        throw std::runtime_error(OclError::toString(CL_INVALID_PROGRAM));
    }

    m_randomx_run = new RxRunKernel(m_asmProgram);
    m_randomx_run->setArgs(m_dataset, m_scratchpads, m_registers, m_rounding, m_programs, m_intensity, m_algorithm);
}


void xmrig::OclRxJitRunner::execute(uint32_t iteration)
{
    m_randomx_jit->enqueue(m_queue, m_intensity, iteration);

    OclLib::finish(m_queue);

    m_randomx_run->enqueue(m_queue, m_intensity, (m_gcn_version == 15) ? 32 : 64);
}


void xmrig::OclRxJitRunner::init()
{
    OclRxBaseRunner::init();

    m_registers             = createSubBuffer(CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, 256 * m_intensity);
    m_intermediate_programs = createSubBuffer(CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, 5120 * m_intensity);
    m_programs              = createSubBuffer(CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, 10048 * m_intensity);
}


bool xmrig::OclRxJitRunner::loadAsmProgram()
{
    // Adrenaline drivers on Windows and amdgpu-pro drivers on Linux use ELF header's flags (offset 0x30) to store internal device ID
    // Read it from compiled OpenCL code and substitute this ID into pre-compiled binary to make sure the driver accepts it
    uint32_t elf_header_flags = 0;
    const uint32_t elf_header_flags_offset = 0x30;

    size_t bin_size = 0;
    if (OclLib::getProgramInfo(m_program, CL_PROGRAM_BINARY_SIZES, sizeof(bin_size), &bin_size) != CL_SUCCESS) {
        return false;
    }

    std::vector<char> binary_data(bin_size);
    char* tmp[1] = { binary_data.data() };
    if (OclLib::getProgramInfo(m_program, CL_PROGRAM_BINARIES, sizeof(char*), tmp) != CL_SUCCESS) {
        return false;
    }

    if (bin_size >= elf_header_flags_offset + sizeof(uint32_t)) {
        elf_header_flags = *reinterpret_cast<uint32_t*>((binary_data.data() + elf_header_flags_offset));
    }

    size_t len              = 0;
    unsigned char *binary   = nullptr;

    switch (m_gcn_version) {
    case 14:
        len = randomx_run_gfx900_bin_size;
        binary = randomx_run_gfx900_bin;
        break;
    case 15:
        len = randomx_run_gfx1010_bin_size;
        binary = randomx_run_gfx1010_bin;
        break;
    default:
        len = randomx_run_gfx803_bin_size;
        binary = randomx_run_gfx803_bin;
        break;
    }

    // Set correct internal device ID in the pre-compiled binary
    if (elf_header_flags) {
        *reinterpret_cast<uint32_t*>(binary + elf_header_flags_offset) = elf_header_flags;
    }

    cl_int status   = 0;
    cl_int ret      = 0;
    cl_device_id device = data().device.id();

    m_asmProgram = OclLib::createProgramWithBinary(ctx(), 1, &device, &len, (const unsigned char**) &binary, &status, &ret);
    if (ret != CL_SUCCESS) {
        return false;
    }

    return OclLib::buildProgram(m_asmProgram, 1, &device) == CL_SUCCESS;
}
