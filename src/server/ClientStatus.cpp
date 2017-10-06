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
    : hashrateShort(0), hashrateMedium(0), hashrateLong(0), sharesGood(0), sharedTotal(0), hashesTotal(0),
      lastStatusUpdate(0)
{

}

const std::string ClientStatus::getMiner() const
{
    return miner;
}

void ClientStatus::setMiner(const std::string &miner)
{
    ClientStatus::miner = miner;
}

const std::string ClientStatus::getCurrentPool() const
{
    return currentPool;
}

void ClientStatus::setCurrentPool(const std::string &currentPool)
{
    ClientStatus::currentPool = currentPool;
}

const std::string ClientStatus::getCurrentStatus() const
{
    return currentStatus;
}

void ClientStatus::setCurrentStatus(const std::string &currentStatus)
{
    ClientStatus::currentStatus = currentStatus;
}

double ClientStatus::getHashrateShort() const
{
    return hashrateShort;
}

void ClientStatus::setHashrateShort(double hashrateShort)
{
    ClientStatus::hashrateShort = hashrateShort;
}

double ClientStatus::getHashrateMedium() const
{
    return hashrateMedium;
}

void ClientStatus::setHashrateMedium(double hashrateMedium)
{
    ClientStatus::hashrateMedium = hashrateMedium;
}

double ClientStatus::getHashrateLong() const
{
    return hashrateLong;
}

void ClientStatus::setHashrateLong(double hashrateLong)
{
    ClientStatus::hashrateLong = hashrateLong;
}

uint64_t ClientStatus::getSharesGood() const
{
    return sharesGood;
}

void ClientStatus::setSharesGood(uint64_t sharesGood)
{
    ClientStatus::sharesGood = sharesGood;
}

uint64_t ClientStatus::getSharedTotal() const
{
    return sharedTotal;
}

void ClientStatus::setSharedTotal(uint64_t sharedTotal)
{
    ClientStatus::sharedTotal = sharedTotal;
}

uint64_t ClientStatus::getHashesTotal() const
{
    return hashesTotal;
}

void ClientStatus::setHashesTotal(uint64_t hashesTotal)
{
    ClientStatus::hashesTotal = hashesTotal;
}

const uint32_t ClientStatus::getLastStatusUpdate() const
{
    return lastStatusUpdate;
}

void ClientStatus::setLastStatusUpdate(uint32_t lastStatusUpdate)
{
    ClientStatus::lastStatusUpdate = lastStatusUpdate;
}

void ClientStatus::parseFromJson(const json_t &json)
{

}

std::string ClientStatus::toJson()
{

}