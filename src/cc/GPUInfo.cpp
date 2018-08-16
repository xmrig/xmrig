/* XMRigCC
 * Copyright 2018-     BenDr0id    <ben@graef.in>
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

#include "GPUInfo.h"

GPUInfo::GPUInfo()
    : m_deviceIdx(0),
      m_rawIntensity(0),
      m_workSize(0),
      m_maxWorkSize(0),
      m_freeMem(0),
      m_memChunk(0),
      m_compMode(0),
      m_computeUnits(0)
{

}

GPUInfo::~GPUInfo()
{

}

rapidjson::Value GPUInfo::toJson(rapidjson::MemoryPoolAllocator <rapidjson::CrtAllocator>& allocator)
{
    rapidjson::Value gpuInfo(rapidjson::kObjectType);

    gpuInfo.AddMember("name", rapidjson::StringRef(m_name.c_str()), allocator);
    gpuInfo.AddMember("device_idx", m_deviceIdx, allocator);
    gpuInfo.AddMember("raw_intensity", m_rawIntensity, allocator);
    gpuInfo.AddMember("work_size", m_workSize, allocator);
    gpuInfo.AddMember("max_work_size", m_maxWorkSize, allocator);
    gpuInfo.AddMember("free_mem", m_freeMem, allocator);
    gpuInfo.AddMember("mem_chunk", m_memChunk, allocator);
    gpuInfo.AddMember("comp_mode", m_memChunk, allocator);
    gpuInfo.AddMember("compute_units", m_memChunk, allocator);

    return gpuInfo;
}

bool GPUInfo::parseFromJson(const rapidjson::Value& gpuInfo)
{
    bool result = false;

    if (gpuInfo.HasMember("name")) {
        m_name = gpuInfo["name"].GetString();
        result = true;
    }

    if (gpuInfo.HasMember("device_idx")) {
        m_deviceIdx = static_cast<size_t>(gpuInfo["device_idx"].GetInt());
    }

    if (gpuInfo.HasMember("raw_intensity")) {
        m_rawIntensity = static_cast<size_t>(gpuInfo["raw_intensity"].GetInt());
    }

    if (gpuInfo.HasMember("work_size")) {
        m_workSize = static_cast<size_t>(gpuInfo["work_size"].GetInt());
    }

    if (gpuInfo.HasMember("max_work_size")) {
        m_maxWorkSize = static_cast<size_t>(gpuInfo["max_work_size"].GetInt());
    }

    if (gpuInfo.HasMember("free_mem")) {
        m_freeMem = static_cast<size_t>(gpuInfo["free_mem"].GetInt());
    }

    if (gpuInfo.HasMember("mem_chunk")) {
        m_memChunk = gpuInfo["mem_chunk"].GetInt();
    }

    if (gpuInfo.HasMember("comp_mode")) {
        m_compMode = gpuInfo["comp_mode"].GetInt();
    }

    if (gpuInfo.HasMember("compute_units")) {
        m_computeUnits = gpuInfo["compute_units"].GetInt();
    }

    return result;
}

size_t GPUInfo::getDeviceIdx() const
{
    return m_deviceIdx;
}

void GPUInfo::setDeviceIdx(size_t deviceIdx)
{
    m_deviceIdx = deviceIdx;
}

size_t GPUInfo::getRawIntensity() const
{
    return m_rawIntensity;
}

void GPUInfo::setRawIntensity(size_t rawIntensity)
{
    m_rawIntensity = rawIntensity;
}

size_t GPUInfo::getWorkSize() const
{
    return m_workSize;
}

void GPUInfo::setWorkSize(size_t workSize)
{
    m_workSize = workSize;
}

size_t GPUInfo::getMaxWorkSize() const
{
    return m_maxWorkSize;
}

void GPUInfo::setMaxWorkSize(size_t maxWorkSize)
{
    m_maxWorkSize = maxWorkSize;
}

size_t GPUInfo::getFreeMem() const
{
    return m_freeMem;
}

void GPUInfo::setFreeMem(size_t freeMem)
{
    m_freeMem = freeMem;
}

int GPUInfo::getMemChunk() const
{
    return m_memChunk;
}

void GPUInfo::setMemChunk(int memChunk)
{
    m_memChunk = memChunk;
}

int GPUInfo::getCompMode() const
{
    return m_compMode;
}

void GPUInfo::setCompMode(int compMode)
{
    m_compMode = compMode;
}

int GPUInfo::getComputeUnits() const
{
    return m_computeUnits;
}

void GPUInfo::setComputeUnits(int computeUnits)
{
    m_computeUnits = computeUnits;
}

std::string GPUInfo::getName() const
{
    return m_name;
}

void GPUInfo::setName(const std::string& name)
{
    m_name = name;
}
