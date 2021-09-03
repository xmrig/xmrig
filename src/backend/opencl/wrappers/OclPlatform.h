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

#ifndef XMRIG_OCLPLATFORM_H
#define XMRIG_OCLPLATFORM_H


#include <vector>


#include "backend/opencl/wrappers/OclDevice.h"
#include "base/tools/String.h"


using cl_platform_id = struct _cl_platform_id *;


namespace xmrig {


class OclPlatform
{
public:
    OclPlatform() = default;
    OclPlatform(size_t index, cl_platform_id id) : m_id(id), m_index(index) {}

    static bool printPlatforms(int &rc);
    static std::vector<OclPlatform> get();
    static void print();

    inline bool isValid() const      { return m_id != nullptr; }
    inline cl_platform_id id() const { return m_id; }
    inline size_t index() const      { return m_index; }

    rapidjson::Value toJSON(rapidjson::Document &doc) const;
    std::vector<OclDevice> devices() const;
    String extensions() const;
    String name() const;
    String profile() const;
    String vendor() const;
    String version() const;

private:
    cl_platform_id m_id = nullptr;
    size_t m_index      = 0;
};


} // namespace xmrig


#endif /* XMRIG_OCLPLATFORM_H */
