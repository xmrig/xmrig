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

#ifndef XMRIG_ICONFIGTRANSFORM_H
#define XMRIG_ICONFIGTRANSFORM_H


#include "3rdparty/rapidjson/fwd.h"
#include "base/tools/Object.h"


namespace xmrig {


class IJsonReader;
class String;


class IConfigTransform
{
public:
    XMRIG_DISABLE_COPY_MOVE(IConfigTransform)

    IConfigTransform()          = default;
    virtual ~IConfigTransform() = default;

    virtual void finalize(rapidjson::Document &doc)                             = 0;
    virtual void transform(rapidjson::Document &doc, int key, const char *arg)  = 0;
};


} /* namespace xmrig */


#endif // XMRIG_ICONFIGTRANSFORM_H
