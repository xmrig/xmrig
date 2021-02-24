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

#ifndef xmlcore_CONFIGTRANSFORM_H
#define xmlcore_CONFIGTRANSFORM_H


#include "base/kernel/config/BaseTransform.h"


namespace xmlcore {


class ConfigTransform : public BaseTransform
{
protected:
    void finalize(rapidjson::Document &doc) override;
    void transform(rapidjson::Document &doc, int key, const char *arg) override;

private:
    void transformBoolean(rapidjson::Document &doc, int key, bool enable);
    void transformUint64(rapidjson::Document &doc, int key, uint64_t arg);

#   ifdef xmlcore_FEATURE_BENCHMARK
    void transformBenchmark(rapidjson::Document &doc, int key, const char *arg);
#   endif

    bool m_opencl           = false;
    int64_t m_affinity      = -1;
    uint64_t m_intensity    = 1;
    uint64_t m_threads      = 0;
};


} // namespace xmlcore


#endif /* xmlcore_CONFIGTRANSFORM_H */
