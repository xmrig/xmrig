/* XMRigCC
 * Copyright 2018-     BenDr0id    <https://github.com/BenDr0id>, <ben@graef.in>
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
#ifndef XMRIG_GPUINFO_H
#define XMRIG_GPUINFO_H

#include <string>
#include <3rdparty/rapidjson/document.h>

class GPUInfo
{
public:
    GPUInfo();
    ~GPUInfo();

    rapidjson::Value toJson(rapidjson::MemoryPoolAllocator<rapidjson::CrtAllocator>& allocator);
    bool parseFromJson(const rapidjson::Value& gpuInfo);

    std::string getName() const;
    void setName(const std::string& name);

    size_t getDeviceIdx() const;
    void setDeviceIdx(size_t deviceIdx);

    size_t getRawIntensity() const;
    void setRawIntensity(size_t rawIntensity);

    size_t getWorkSize() const;
    void setWorkSize(size_t workSize);

    size_t getMaxWorkSize() const;
    void setMaxWorkSize(size_t maxWorkSize);

    size_t getFreeMem() const;
    void setFreeMem(size_t freeMem);

    int getMemChunk() const;
    void setMemChunk(int memChunk);

    int getCompMode() const;
    void setCompMode(int compMode);

    int getComputeUnits() const;
    void setComputeUnits(int computeUnits);

private:
    size_t m_deviceIdx;
    size_t m_rawIntensity;
    size_t m_workSize;
    size_t m_maxWorkSize;
    size_t m_freeMem;

    int m_memChunk;
    int m_compMode;
    int m_computeUnits;

    std::string m_name;
};


#endif //XMRIG_GPUINFO_H
