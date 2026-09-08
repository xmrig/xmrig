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

#include "base/net/stratum/AutoClient.h"
#include "3rdparty/rapidjson/document.h"
#include "base/io/json/Json.h"
// MoneroOcean: submit routing needs the result algorithm, not just the current client mode.
#include "net/JobResult.h"
// End MoneroOcean


xmrig::AutoClient::AutoClient(int id, const char *agent, IClientListener *listener) :
    EthStratumClient(id, agent, listener)
{
}


bool xmrig::AutoClient::handleResponse(int64_t id, const rapidjson::Value &result, const rapidjson::Value &error)
{
    if (m_mode == DEFAULT_MODE) {
        return Client::handleResponse(id, result, error); // NOLINT(bugprone-parent-virtual-call)
    }

    return EthStratumClient::handleResponse(id, result, error);
}


bool xmrig::AutoClient::parseLogin(const rapidjson::Value &result, int *code)
{
    m_mode = DEFAULT_MODE;

    if (result.HasMember("job")) {
        return Client::parseLogin(result, code);
    }

    setRpcId(Json::getString(result, "id"));
    if (rpcId().isNull()) {
        *code = 1;
        return false;
    }

    // Native MO login replies carry the negotiated extension set alongside
    // the initial algorithm and nonce metadata.
    parseExtensions(result);

    const Algorithm algo(Json::getString(result, "algo"));
    if (algo.family() != Algorithm::KAWPOW && algo.family() != Algorithm::GHOSTRIDER) {
        *code = 6;
        return false;
    }

    try {
        setExtraNonce(Json::getValue(result, "extra_nonce"));
    } catch (const std::exception &ex) {
        *code = 6;
        return false;
    }

    m_mode = ETH_MODE;
    setAlgo(algo);

#   ifdef XMRIG_ALGO_GHOSTRIDER
    if (algo.family() == Algorithm::GHOSTRIDER) {
        setExtraNonce2Size(Json::getUint64(result, "extra_nonce2_size"));
    }
#   endif

    return true;
}


int64_t xmrig::AutoClient::submit(const JobResult &result)
{
    // The job owns its framing.  A later pool notification may already have
    // switched the client mode while an earlier share is still queued.
    if (result.submitMode != Job::SUBMIT_ETH) {
        return Client::submit(result); // NOLINT(bugprone-parent-virtual-call)
    }
    return EthStratumClient::submit(result);
}


void xmrig::AutoClient::parseNotification(const char *method, const rapidjson::Value &params, const rapidjson::Value &error)
{
    // MoneroOcean: pools can switch between plain stratum jobs and
    // Ethereum-style jobs.  Control notifications update framing state only;
    // the native job notification is the family boundary.
    if (strcmp(method, "job") == 0) {
        m_mode = DEFAULT_MODE;
        return Client::parseNotification(method, params, error); // NOLINT(bugprone-parent-virtual-call)
    }

    if (strcmp(method, "mining.notify") == 0) {
        m_mode = ETH_MODE;
    }
    // End MoneroOcean
    return EthStratumClient::parseNotification(method, params, error);
}
