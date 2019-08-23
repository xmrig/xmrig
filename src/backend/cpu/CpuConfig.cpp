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

static const char *kCn                  = "cn";
static const char *kEnabled             = "enabled";
static const char *kHugePages           = "huge-pages";
static const char *kHwAes               = "hw-aes";
static const char *kPriority            = "priority";

#ifdef XMRIG_FEATURE_ASM
static const char *kAsm = "asm";
#endif

#ifdef XMRIG_ALGO_CN_GPU
static const char *kCnGPU = "cn/gpu";
#endif

#ifdef XMRIG_ALGO_CN_LITE
static const char *kCnLite = "cn-lite";
#endif

#ifdef XMRIG_ALGO_CN_HEAVY
static const char *kCnHeavy = "cn-heavy";
#endif

#ifdef XMRIG_ALGO_CN_PICO
static const char *kCnPico = "cn-pico";
#endif

#ifdef XMRIG_ALGO_RANDOMX
static const char *kRx    = "rx";
static const char *kRxWOW = "rx/wow";
#endif

#ifdef XMRIG_ALGO_ARGON2
static const char *kArgon2     = "argon2";
static const char *kArgon2Impl = "argon2-impl";
#endif

extern template class Threads<CpuThreads>;

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

#   ifdef XMRIG_ALGO_ARGON2
    obj.AddMember(StringRef(kArgon2Impl), m_argon2Impl.toJSON(), allocator);
#   endif

    m_threads.toJSON(obj, doc);

    return obj;
}


std::vector<xmrig::CpuLaunchData> xmrig::CpuConfig::get(const Miner *miner, const Algorithm &algorithm) const
{
    std::vector<CpuLaunchData> out;
    const CpuThreads &threads = m_threads.get(algorithm);

    if (threads.isEmpty()) {
        return out;
    }

    out.reserve(threads.count());

    for (const CpuThread &thread : threads.data()) {
        out.emplace_back(miner, algorithm, *this, thread);
    }

    return out;
}


void xmrig::CpuConfig::read(const rapidjson::Value &value, uint32_t version)
{
    if (value.IsObject()) {
        m_enabled       = Json::getBool(value, kEnabled, m_enabled);
        m_hugePages     = Json::getBool(value, kHugePages, m_hugePages);

        setAesMode(Json::getValue(value, kHwAes));
        setPriority(Json::getInt(value,  kPriority, -1));

#       ifdef XMRIG_FEATURE_ASM
        m_assembly = Json::getValue(value, kAsm);
#       endif

#       ifdef XMRIG_ALGO_ARGON2
        m_argon2Impl = Json::getString(value, kArgon2Impl);
#       endif

        if (!m_threads.read(value)) {
            generate();
        }

        if (version == 0) {
            generateArgon2();
        }
    }
    else if (value.IsBool() && value.IsFalse()) {
        m_enabled = false;
    }
    else {
        generate();
    }
}


void xmrig::CpuConfig::generate()
{
    m_shouldSave  = true;
    ICpuInfo *cpu = Cpu::info();

    m_threads.disable(Algorithm::CN_0);
    m_threads.move(kCn, cpu->threads(Algorithm::CN_0));

#   ifdef XMRIG_ALGO_CN_GPU
    m_threads.move(kCnGPU, cpu->threads(Algorithm::CN_GPU));
#   endif

#   ifdef XMRIG_ALGO_CN_LITE
    m_threads.disable(Algorithm::CN_LITE_0);
    m_threads.move(kCnLite, cpu->threads(Algorithm::CN_LITE_1));
#   endif

#   ifdef XMRIG_ALGO_CN_HEAVY
    m_threads.move(kCnHeavy, cpu->threads(Algorithm::CN_HEAVY_0));
#   endif

#   ifdef XMRIG_ALGO_CN_PICO
    m_threads.move(kCnPico, cpu->threads(Algorithm::CN_PICO_0));
#   endif

#   ifdef XMRIG_ALGO_RANDOMX
    m_threads.move(kRx, cpu->threads(Algorithm::RX_0));
    m_threads.move(kRxWOW, cpu->threads(Algorithm::RX_WOW));
#   endif

    generateArgon2();
}


void xmrig::CpuConfig::generateArgon2()
{
#   ifdef XMRIG_ALGO_ARGON2
    m_threads.move(kArgon2, Cpu::info()->threads(Algorithm::AR2_CHUKWA));
#   endif
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
