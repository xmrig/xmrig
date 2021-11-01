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

#ifndef XMRIG_IJSONREADER_H
#define XMRIG_IJSONREADER_H


#include "3rdparty/rapidjson/fwd.h"
#include "base/tools/Object.h"
#include "base/tools/String.h"


namespace xmrig {


class IJsonReader
{
public:
    XMRIG_DISABLE_COPY_MOVE(IJsonReader)

    IJsonReader()           = default;
    virtual ~IJsonReader()  = default;

    virtual bool getBool(const char *key, bool defaultValue = false) const                       = 0;
    virtual bool isEmpty() const                                                                 = 0;
    virtual const char *getString(const char *key, const char *defaultValue = nullptr) const     = 0;
    virtual const rapidjson::Value &getArray(const char *key) const                              = 0;
    virtual const rapidjson::Value &getObject(const char *key) const                             = 0;
    virtual const rapidjson::Value &getValue(const char *key) const                              = 0;
    virtual const rapidjson::Value &object() const                                               = 0;
    virtual double getDouble(const char *key, double defaultValue = 0) const                     = 0;
    virtual int getInt(const char *key, int defaultValue = 0) const                              = 0;
    virtual int64_t getInt64(const char *key, int64_t defaultValue = 0) const                    = 0;
    virtual String getString(const char *key, size_t maxSize) const                              = 0;
    virtual uint64_t getUint64(const char *key, uint64_t defaultValue = 0) const                 = 0;
    virtual unsigned getUint(const char *key, unsigned defaultValue = 0) const                   = 0;
};


} /* namespace xmrig */


#endif // XMRIG_IJSONREADER_H
