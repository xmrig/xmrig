/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
 * Copyright 2017-     BenDr0id    <ben@graef.in>
 *
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


#include <chrono>
#include <cstring>
#include <3rdparty/rapidjson/stringbuffer.h>
#include <3rdparty/rapidjson/prettywriter.h>
#include <log/Log.h>

#include "ClientStatus.h"

ClientStatus::ClientStatus()
    : m_currentStatus(Status::PAUSED),
      m_hasHugepages(false),
      m_isHugepagesEnabled(false),
      m_isCpuX64(false),
      m_hasCpuAES(false),
      m_hashFactor(1),
      m_hashrateShort(0),
      m_hashrateMedium(0),
      m_hashrateLong(0),
      m_hashrateHighest(0),
      m_currentThreads(0),
      m_cpuSockets(0),
      m_cpuCores(0),
      m_cpuThreads(0),
      m_cpuL2(0),
      m_cpuL3(0),
      m_sharesGood(0),
      m_sharesTotal(0),
      m_hashesTotal(0),
      m_uptime(0),
      m_avgTime(0),
      m_lastStatusUpdate(0)
{

}

ClientStatus::Status ClientStatus::getCurrentStatus() const
{
    return m_currentStatus;
}

void ClientStatus::setCurrentStatus(Status currentStatus)
{
    m_currentStatus = currentStatus;
}

std::string ClientStatus::getClientId() const
{
    return m_clientId;
}

void ClientStatus::setClientId(const std::string& clientId)
{
    m_clientId = clientId;
}

std::string ClientStatus::getCurrentPool() const
{
    return m_currentPool;
}

void ClientStatus::setCurrentPool(const std::string& currentPool)
{
    m_currentPool = currentPool;
}

void ClientStatus::setCurrentAlgoName(const std::string& algoName)
{
    m_currentAlgoName = algoName;
}

std::string ClientStatus::getCurrentAlgoName() const
{
    return m_currentAlgoName;
}

void ClientStatus::setCurrentPowVariantName(const std::string& powVariantName)
{
    m_currentPowVariantName = powVariantName;
}

std::string ClientStatus::getCurrentPowVariantName() const
{
    return m_currentPowVariantName;
}

std::string ClientStatus::getCpuBrand() const
{
    return m_cpuBrand;
}

void ClientStatus::setCpuBrand(const std::string& cpuBrand)
{
    m_cpuBrand = cpuBrand;
}

std::string ClientStatus::getExternalIp() const
{
    return m_externalIp;
}

void ClientStatus::setExternalIp(const std::string& externalIp)
{
    m_externalIp = externalIp;
}

std::string ClientStatus::getVersion() const
{
    return m_version;
}

void ClientStatus::setVersion(const std::string& version)
{
    m_version = version;
}

std::string ClientStatus::getLog() const
{
    return m_log;
}

void ClientStatus::setLog(const std::string& log)
{
    m_log = log;
}

bool ClientStatus::hasHugepages() const
{
    return m_hasHugepages;
}

void ClientStatus::setHugepages(bool hasHugepages)
{
    m_hasHugepages = hasHugepages;
}

bool ClientStatus::isHugepagesEnabled() const
{
    return m_isHugepagesEnabled;
}

void ClientStatus::setHugepagesEnabled(bool hugepagesEnabled)
{
    m_isHugepagesEnabled = hugepagesEnabled;
}

int ClientStatus::getHashFactor() const
{
    return m_hashFactor;
}

void ClientStatus::setHashFactor(int hashFactor)
{
    m_hashFactor = hashFactor;
}

bool ClientStatus::isCpuX64() const
{
    return m_isCpuX64;
}

void ClientStatus::setCpuX64(bool isCpuX64)
{
    m_isCpuX64 = isCpuX64;
}

bool ClientStatus::hasCpuAES() const
{
    return m_hasCpuAES;
}

void ClientStatus::setCpuAES(bool hasCpuAES)
{
    m_hasCpuAES = hasCpuAES;
}

double ClientStatus::getHashrateShort() const
{
    return m_hashrateShort;
}

void ClientStatus::setHashrateShort(double hashrateShort)
{
    m_hashrateShort = hashrateShort;
}

double ClientStatus::getHashrateMedium() const
{
    return m_hashrateMedium;
}

void ClientStatus::setHashrateMedium(double hashrateMedium)
{
    m_hashrateMedium = hashrateMedium;
}

double ClientStatus::getHashrateLong() const
{
    return m_hashrateLong;
}

void ClientStatus::setHashrateLong(double hashrateLong)
{
    m_hashrateLong = hashrateLong;
}

void ClientStatus::setHashrateHighest(double hashrateHighest)
{
    m_hashrateHighest = hashrateHighest;
}

double ClientStatus::getHashrateHighest() const
{
    return m_hashrateHighest;
}

int ClientStatus::getCurrentThreads() const
{
    return m_currentThreads;
}

void ClientStatus::setCurrentThreads(int currentThreads)
{
    m_currentThreads = currentThreads;
}

int ClientStatus::getCpuSockets() const
{
    return m_cpuSockets;
}

void ClientStatus::setCpuSockets(int cpuSockets)
{
    m_cpuSockets = cpuSockets;
}

int ClientStatus::getCpuCores() const
{
    return m_cpuCores;
}

void ClientStatus::setCpuCores(int cpuCores)
{
    m_cpuCores = cpuCores;
}

int ClientStatus::getCpuThreads() const
{
    return m_cpuThreads;
}

void ClientStatus::setCpuThreads(int cpuThreads)
{
    m_cpuThreads = cpuThreads;
}

int ClientStatus::getCpuL2() const
{
    return m_cpuL2;
}

void ClientStatus::setCpuL2(int cpuL2)
{
    m_cpuL2 = cpuL2;
}

int ClientStatus::getCpuL3() const
{
    return m_cpuL3;
}

void ClientStatus::setCpuL3(int cpuL3)
{
    m_cpuL3 = cpuL3;
}

uint64_t ClientStatus::getSharesGood() const
{
    return m_sharesGood;
}

void ClientStatus::setSharesGood(uint64_t sharesGood)
{
    m_sharesGood = sharesGood;
}

uint64_t ClientStatus::getSharesTotal() const
{
    return m_sharesTotal;
}

void ClientStatus::setSharesTotal(uint64_t sharesTotal)
{
    m_sharesTotal = sharesTotal;
}

uint64_t ClientStatus::getHashesTotal() const
{
    return m_hashesTotal;
}

void ClientStatus::setHashesTotal(uint64_t hashesTotal)
{
    m_hashesTotal = hashesTotal;
}
void ClientStatus::setAvgTime(uint32_t avgTime)
{
    m_avgTime = avgTime;
}

uint32_t ClientStatus::getAvgTime() const
{
    return m_avgTime;
}

std::time_t ClientStatus::getLastStatusUpdate() const
{
    return m_lastStatusUpdate;
}

uint64_t ClientStatus::getUptime() const
{
    return m_uptime;
}

void ClientStatus::setUptime(uint64_t uptime)
{
    m_uptime = uptime;
}

bool ClientStatus::parseFromJson(const rapidjson::Document& document)
{
    bool result = false;

    if (document.HasMember("client_status"))
    {
        rapidjson::Value::ConstObject clientStatus = document["client_status"].GetObject();

        if (clientStatus.HasMember("current_status")) {
            m_currentStatus = toStatus(clientStatus["current_status"].GetString());
        }

        if (clientStatus.HasMember("client_id")) {
            m_clientId = clientStatus["client_id"].GetString();
        }

        if (clientStatus.HasMember("current_pool")) {
            m_currentPool = clientStatus["current_pool"].GetString();
        }

        if (clientStatus.HasMember("current_algo_name")) {
            m_currentAlgoName = clientStatus["current_algo_name"].GetString();
        }

        if (clientStatus.HasMember("current_pow_variant_name")) {
            m_currentPowVariantName = clientStatus["current_pow_variant_name"].GetString();
        }

        if (clientStatus.HasMember("cpu_brand")) {
            m_cpuBrand = clientStatus["cpu_brand"].GetString();
        }

        if (clientStatus.HasMember("external_ip")) {
            m_externalIp = clientStatus["external_ip"].GetString();
        }

        if (clientStatus.HasMember("version")) {
            m_version = clientStatus["version"].GetString();
        }

        if (clientStatus.HasMember("log")) {
            m_log = clientStatus["log"].GetString();
        }

        if (clientStatus.HasMember("hugepages_available")) {
            m_hasHugepages = clientStatus["hugepages_available"].GetBool();
        }

        if (clientStatus.HasMember("hugepages_enabled")) {
            m_isHugepagesEnabled = clientStatus["hugepages_enabled"].GetBool();
        }

        if (clientStatus.HasMember("hash_factor")) {
            m_hashFactor = clientStatus["hash_factor"].GetInt();
        }

        if (clientStatus.HasMember("cpu_is_x64")) {
            m_isCpuX64 = clientStatus["cpu_is_x64"].GetBool();
        }

        if (clientStatus.HasMember("cpu_has_aes")) {
            m_hasCpuAES = clientStatus["cpu_has_aes"].GetBool();
        }

        if (clientStatus.HasMember("hashrate_short")) {
            m_hashrateShort = clientStatus["hashrate_short"].GetDouble();
        }

        if (clientStatus.HasMember("hashrate_medium")) {
            m_hashrateMedium = clientStatus["hashrate_medium"].GetDouble();
        }

        if (clientStatus.HasMember("hashrate_long")) {
            m_hashrateLong = clientStatus["hashrate_long"].GetDouble();
        }

        if (clientStatus.HasMember("hashrate_highest")) {
            m_hashrateHighest = clientStatus["hashrate_highest"].GetDouble();
        }

        if (clientStatus.HasMember("current_threads")) {
            m_currentThreads = clientStatus["current_threads"].GetInt();
        }

        if (clientStatus.HasMember("cpu_sockets")) {
            m_cpuSockets = clientStatus["cpu_sockets"].GetInt();
        }

        if (clientStatus.HasMember("cpu_cores")) {
            m_cpuCores = clientStatus["cpu_cores"].GetInt();
        }

        if (clientStatus.HasMember("cpu_threads")) {
            m_cpuThreads = clientStatus["cpu_threads"].GetInt();
        }

        if (clientStatus.HasMember("cpu_l2")) {
            m_cpuL2 = clientStatus["cpu_l2"].GetInt();
        }

        if (clientStatus.HasMember("cpu_l3")) {
            m_cpuL3 = clientStatus["cpu_l3"].GetInt();
        }

        if (clientStatus.HasMember("shares_good")) {
            m_sharesGood = clientStatus["shares_good"].GetUint64();
        }

        if (clientStatus.HasMember("shares_total")) {
            m_sharesTotal = clientStatus["shares_total"].GetUint64();
        }

        if (clientStatus.HasMember("hashes_total")) {
            m_hashesTotal = clientStatus["hashes_total"].GetUint64();
        }

        if (clientStatus.HasMember("avg_time")) {
            m_avgTime = clientStatus["avg_time"].GetUint();
        }

        if (clientStatus.HasMember("uptime")) {
            m_uptime = clientStatus["uptime"].GetUint64();
        }

        auto time_point = std::chrono::system_clock::now();
        m_lastStatusUpdate = std::chrono::system_clock::to_time_t(time_point);

        result = true;
    } else {
        LOG_ERR("Parse Error, JSON does not contain: control_command");
    }

    return result;
}

rapidjson::Value ClientStatus::toJson(rapidjson::MemoryPoolAllocator<rapidjson::CrtAllocator>& allocator)
{
    rapidjson::Value clientStatus(rapidjson::kObjectType);

    clientStatus.AddMember("current_status", rapidjson::StringRef(toString(m_currentStatus)), allocator);

    clientStatus.AddMember("client_id", rapidjson::StringRef(m_clientId.c_str()), allocator);
    clientStatus.AddMember("current_pool", rapidjson::StringRef(m_currentPool.c_str()), allocator);
    clientStatus.AddMember("current_algo_name", rapidjson::StringRef(m_currentAlgoName.c_str()), allocator);
    clientStatus.AddMember("current_pow_variant_name", rapidjson::StringRef(m_currentPowVariantName.c_str()), allocator);
    clientStatus.AddMember("cpu_brand", rapidjson::StringRef(m_cpuBrand.c_str()), allocator);
    clientStatus.AddMember("external_ip", rapidjson::StringRef(m_externalIp.c_str()), allocator);
    clientStatus.AddMember("version", rapidjson::StringRef(m_version.c_str()), allocator);

    clientStatus.AddMember("hugepages_available", m_hasHugepages, allocator);
    clientStatus.AddMember("hugepages_enabled", m_isHugepagesEnabled, allocator);
    clientStatus.AddMember("hash_factor", m_hashFactor, allocator);
    clientStatus.AddMember("cpu_is_x64", m_isCpuX64, allocator);
    clientStatus.AddMember("cpu_has_aes", m_hasCpuAES, allocator);

    clientStatus.AddMember("hashrate_short", m_hashrateShort, allocator);
    clientStatus.AddMember("hashrate_medium", m_hashrateMedium, allocator);
    clientStatus.AddMember("hashrate_long", m_hashrateLong, allocator);
    clientStatus.AddMember("hashrate_highest", m_hashrateHighest, allocator);

    clientStatus.AddMember("current_threads", m_currentThreads, allocator);
    clientStatus.AddMember("cpu_sockets", m_cpuSockets, allocator);
    clientStatus.AddMember("cpu_cores", m_cpuCores, allocator);
    clientStatus.AddMember("cpu_threads", m_cpuThreads, allocator);
    clientStatus.AddMember("cpu_l2", m_cpuL2, allocator);
    clientStatus.AddMember("cpu_l3", m_cpuL3, allocator);

    clientStatus.AddMember("shares_good", m_sharesGood, allocator);
    clientStatus.AddMember("shares_total", m_sharesTotal, allocator);
    clientStatus.AddMember("hashes_total", m_hashesTotal, allocator);

    clientStatus.AddMember("avg_time", m_avgTime, allocator);

    clientStatus.AddMember("uptime", m_uptime, allocator);

    clientStatus.AddMember("last_status_update", static_cast<uint64_t >(m_lastStatusUpdate), allocator);

    clientStatus.AddMember("log", rapidjson::StringRef(m_log.c_str()), allocator);


    return clientStatus;
}

std::string ClientStatus::toJsonString()
{
    rapidjson::Document respDocument;
    respDocument.SetObject();

    auto& allocator = respDocument.GetAllocator();

    rapidjson::Value clientStatus = ClientStatus::toJson(allocator);
    respDocument.AddMember("client_status", clientStatus, allocator);

    rapidjson::StringBuffer buffer(0, 4096);
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    writer.SetMaxDecimalPlaces(10);
    respDocument.Accept(writer);

    return strdup(buffer.GetString());
}

