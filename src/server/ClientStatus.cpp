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


#include <cstring>

#include "server/ClientStatus.h"

ClientStatus::ClientStatus()
    : m_hashrateShort(0), m_hashrateMedium(0), m_hashrateLong(0), m_sharesGood(0), m_sharedTotal(0), m_hashesTotal(0),
      m_lastStatusUpdate(0)
{

}

const std::string ClientStatus::getMiner() const
{
    return m_miner;
}

void ClientStatus::setMiner(const std::string &miner)
{
    m_miner = miner;
}

const std::string ClientStatus::getCurrentPool() const
{
    return m_currentPool;
}

void ClientStatus::setCurrentPool(const std::string &currentPool)
{
    m_currentPool = currentPool;
}

const std::string ClientStatus::getCurrentStatus() const
{
    return m_currentStatus;
}

void ClientStatus::setCurrentStatus(const std::string &currentStatus)
{
    m_currentStatus = currentStatus;
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

uint64_t ClientStatus::getSharesGood() const
{
    return m_sharesGood;
}

void ClientStatus::setSharesGood(uint64_t sharesGood)
{
    m_sharesGood = sharesGood;
}

uint64_t ClientStatus::getSharedTotal() const
{
    return m_sharedTotal;
}

void ClientStatus::setSharedTotal(uint64_t sharedTotal)
{
    m_sharedTotal = sharedTotal;
}

uint64_t ClientStatus::getHashesTotal() const
{
    return m_hashesTotal;
}

void ClientStatus::setHashesTotal(uint64_t hashesTotal)
{
    m_hashesTotal = hashesTotal;
}

const uint32_t ClientStatus::getLastStatusUpdate() const
{
    return m_lastStatusUpdate;
}

void ClientStatus::setLastStatusUpdate(uint32_t lastStatusUpdate)
{
    m_lastStatusUpdate = lastStatusUpdate;
}

void ClientStatus::parseFromJson(const rapidjson::Document &document)
{

}

std::string ClientStatus::toJson()
{

}