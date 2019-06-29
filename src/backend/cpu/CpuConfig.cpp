/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2019 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include "backend/cpu/Cpu.h"
#include "backend/cpu/CpuConfig.h"
#include "base/io/json/Json.h"
#include "rapidjson/document.h"


namespace xmrig {


static const char *kEnabled             = "enabled";
static const char *kHugePages           = "huge-pages";
static const char *kHwAes               = "hw-aes";
static const char *kPriority            = "priority";


#ifdef XMRIG_FEATURE_ASM
static const char *kAsm = "asm";
#endif

}


xmrig::CpuConfig::CpuConfig()
{
}


bool xmrig::CpuConfig::isHwAES() const
{
    return (m_aes == AES_AUTO ? (Cpu::info()->hasAES() ? AES_HW : AES_SOFT) : m_aes) == AES_HW;
}


rapidjson::Value xmrig::CpuConfig::toJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;

    auto &allocator = doc.GetAllocator();

    Value obj(kObjectType);

    obj.AddMember(StringRef(kEnabled),      m_enabled, allocator);
    obj.AddMember(StringRef(kHugePages),    m_hugePages, allocator);
    obj.AddMember(StringRef(kHwAes),        m_aes == AES_AUTO ? Value(kNullType) : Value(m_aes == AES_HW), allocator);
    obj.AddMember(StringRef(kPriority),     priority() != -1 ? Value(priority()) : Value(kNullType), allocator);

#   ifdef XMRIG_FEATURE_ASM
    obj.AddMember(StringRef(kAsm), m_assembly.toJSON(), allocator);
#   endif

    return obj;
}


void xmrig::CpuConfig::read(const rapidjson::Value &value)
{
    if (value.IsObject()) {
        m_enabled   = Json::getBool(value, kEnabled, m_enabled);
        m_hugePages = Json::getBool(value, kHugePages, m_hugePages);

        setAesMode(Json::getValue(value, kHwAes));
        setPriority(Json::getInt(value, kPriority, -1));

#       ifdef XMRIG_FEATURE_ASM
        m_assembly = Json::getValue(value, kAsm);
#       endif
    }
}


void xmrig::CpuConfig::setAesMode(const rapidjson::Value &aesMode)
{
    if (aesMode.IsBool()) {
        m_aes = aesMode.GetBool() ? AES_HW : AES_SOFT;
    }
    else {
        m_aes = AES_AUTO;
    }
}


void xmrig::CpuConfig::setPriority(int priority)
{
    m_priority = (priority >= -1 && priority <= 5) ? priority : -1;
}
