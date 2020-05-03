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

#ifndef XMRIG_JSON_H
#define XMRIG_JSON_H


#include "3rdparty/rapidjson/fwd.h"
#include "base/kernel/interfaces/IJsonReader.h"


#include <string>
#include <vector>
#include <fstream>


namespace xmrig {


class Json
{
public:
    static bool getBool(const rapidjson::Value &obj, const char *key, bool defaultValue = false);
    static bool isEmpty(const rapidjson::Value &obj);
    static const char *getString(const rapidjson::Value &obj, const char *key, const char *defaultValue = nullptr);
    static const rapidjson::Value &getArray(const rapidjson::Value &obj, const char *key);
    static const rapidjson::Value &getObject(const rapidjson::Value &obj, const char *key);
    static const rapidjson::Value &getValue(const rapidjson::Value &obj, const char *key);
    static int getInt(const rapidjson::Value &obj, const char *key, int defaultValue = 0);
    static int64_t getInt64(const rapidjson::Value &obj, const char *key, int64_t defaultValue = 0);
    static uint64_t getUint64(const rapidjson::Value &obj, const char *key, uint64_t defaultValue = 0);
    static unsigned getUint(const rapidjson::Value &obj, const char *key, unsigned defaultValue = 0);

    static bool get(const char *fileName, rapidjson::Document &doc);
    static bool save(const char *fileName, const rapidjson::Document &doc);

    static bool convertOffset(const char *fileName, size_t offset, size_t &line, size_t &pos, std::vector<std::string> &s);
    static rapidjson::Value normalize(double value, bool zero);

private:
    static bool convertOffset(std::istream &ifs, size_t offset, size_t &line, size_t &pos, std::vector<std::string> &s);
};


class JsonReader : public IJsonReader
{
public:
    inline JsonReader(const rapidjson::Value &obj) : m_obj(obj) {}

    inline bool getBool(const char *key, bool defaultValue = false) const override                   { return Json::getBool(m_obj, key, defaultValue); }
    inline const char *getString(const char *key, const char *defaultValue = nullptr) const override { return Json::getString(m_obj, key, defaultValue); }
    inline const rapidjson::Value &getArray(const char *key) const override                          { return Json::getArray(m_obj, key); }
    inline const rapidjson::Value &getObject(const char *key) const override                         { return Json::getObject(m_obj, key); }
    inline const rapidjson::Value &getValue(const char *key) const override                          { return Json::getValue(m_obj, key); }
    inline int getInt(const char *key, int defaultValue = 0) const override                          { return Json::getInt(m_obj, key, defaultValue); }
    inline int64_t getInt64(const char *key, int64_t defaultValue = 0) const override                { return Json::getInt64(m_obj, key, defaultValue); }
    inline uint64_t getUint64(const char *key, uint64_t defaultValue = 0) const override             { return Json::getUint64(m_obj, key, defaultValue); }
    inline unsigned getUint(const char *key, unsigned defaultValue = 0) const override               { return Json::getUint(m_obj, key, defaultValue); }

    bool isEmpty() const override;

private:
    const rapidjson::Value &m_obj;
};


} /* namespace xmrig */


#endif /* XMRIG_JSON_H */
