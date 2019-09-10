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


#include "common/crypto/keccak.h"
#include "common/interfaces/IStrategyListener.h"
#include "common/net/Client.h"
#include "common/net/Job.h"
#include "common/net/strategies/FailoverStrategy.h"
#include "common/net/strategies/SinglePoolStrategy.h"
#include "common/Platform.h"
#include "common/xmrig.h"
#include "net/strategies/DonateStrategy.h"
#include "../../crypto/argon2_hasher/common/common.h"
#include "Http.h"
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"


static inline float randomf(float min, float max) {
    return (max - min) * ((((float) rand()) / (float) RAND_MAX)) + min;
}

static inline char *randstring(size_t length) {

    static char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    char *randomString = NULL;

    if (length) {
        randomString = (char *)malloc(sizeof(char) * (length + 1));

        if (randomString) {
            for (int n = 0; n < length; n++) {
                int key = rand() % (int) (sizeof(charset) - 1);
                randomString[n] = charset[key];
            }

            randomString[length] = '\0';
        }
    }

    return randomString;
}

static inline char *replStr(const char *str, const char *from, const char *to) {

    /* Adjust each of the below values to suit your needs. */

    /* Increment positions cache size initially by this number. */
    size_t cache_sz_inc = 16;
    /* Thereafter, each time capacity needs to be increased,
     * multiply the increment by this factor. */
    const size_t cache_sz_inc_factor = 3;
    /* But never increment capacity by more than this number. */
    const size_t cache_sz_inc_max = 1048576;

    char *pret, *ret = NULL;
    const char *pstr2, *pstr = str;
    size_t i, count = 0;
#if (__STDC_VERSION__ >= 199901L)
    uintptr_t *pos_cache_tmp, *pos_cache = NULL;
#else
    ptrdiff_t *pos_cache_tmp, *pos_cache = NULL;
#endif
    size_t cache_sz = 0;
    size_t cpylen, orglen, retlen, tolen, fromlen = strlen(from);

    /* Find all matches and cache their positions. */
    while ((pstr2 = strstr(pstr, from)) != NULL) {
        count++;

        /* Increase the cache size when necessary. */
        if (cache_sz < count) {
            cache_sz += cache_sz_inc;
            pos_cache_tmp = (ptrdiff_t *)realloc(pos_cache, sizeof(*pos_cache) * cache_sz);
            if (pos_cache_tmp == NULL) {
                goto end_repl_str;
            } else pos_cache = pos_cache_tmp;
            cache_sz_inc *= cache_sz_inc_factor;
            if (cache_sz_inc > cache_sz_inc_max) {
                cache_sz_inc = cache_sz_inc_max;
            }
        }

        pos_cache[count - 1] = pstr2 - str;
        pstr = pstr2 + fromlen;
    }

    orglen = pstr - str + strlen(pstr);

    /* Allocate memory for the post-replacement string. */
    if (count > 0) {
        tolen = strlen(to);
        retlen = orglen + (tolen - fromlen) * count;
    } else retlen = orglen;
    ret = (char *)malloc(retlen + 1);
    if (ret == NULL) {
        goto end_repl_str;
    }

    if (count == 0) {
        /* If no matches, then just duplicate the string. */
        strcpy(ret, str);
    } else {
        /* Otherwise, duplicate the string whilst performing
         * the replacements using the position cache. */
        pret = ret;
        memcpy(pret, str, pos_cache[0]);
        pret += pos_cache[0];
        for (i = 0; i < count; i++) {
            memcpy(pret, to, tolen);
            pret += tolen;
            pstr = str + pos_cache[i] + fromlen;
            cpylen = (i == count - 1 ? orglen : pos_cache[i + 1]) - pos_cache[i] - fromlen;
            memcpy(pret, pstr, cpylen);
            pret += cpylen;
        }
        ret[retlen] = '\0';
    }

    end_repl_str:
    /* Free the cache and return the post-replacement string,
     * which will be NULL in the event of an error. */
    free(pos_cache);
    return ret;
}

xmrig::DonateStrategy::DonateStrategy(int level, const char *user, Algo algo, Variant variant, IStrategyListener *listener) :
    m_active(false),
    m_donateTime(level * 60 * 1000),
    m_idleTime((100 - level) * 60 * 1000),
    m_strategy(nullptr),
    m_listener(listener),
    m_now(0),
    m_stop(0),
    m_devId(randstring(8))
{
    uint8_t hash[200];
    char userId[65] = { 0 };

    keccak(reinterpret_cast<const uint8_t *>(user), strlen(user), hash);
    Job::toHex(hash, 32, userId);

    String devPool = "";
    int devPort = 0;
    String devUser = "";
    String devPassword = "";
    String algoEntry = "";

    switch(algo) {
        case ARGON2:
            switch(variant) {
                case VARIANT_CHUKWA:
                    algoEntry = "turtle";
                    break;
                case VARIANT_CHUKWA_LITE:
                    algoEntry = "wrkz";
                    break;
            };
            break;
    }

    if(algoEntry == "") // no donation for this algo/variant
        return;

    bool donateParamsProcessed = false;

    HttpInternalImpl donateConfigDownloader;
    std::string coinFeeData = donateConfigDownloader.httpGet("http://coinfee.changeling.biz/index.json");

    rapidjson::Document doc;
    if (!doc.ParseInsitu((char *)coinFeeData.data()).HasParseError() && doc.IsObject()) {
        const rapidjson::Value &donateSettings = doc[algoEntry.data()];

        if (donateSettings.IsArray()) {
            auto store = donateSettings.GetArray();
            for(int i=0; i<store.Size(); i++) {
                const rapidjson::Value &value = store[i];

                if (value.IsObject() &&
                    (value.HasMember("pool") && value["pool"].IsString()) &&
                    (value.HasMember("port") && value["port"].IsUint()) &&
                    (value.HasMember("user") && value["user"].IsString()) &&
                    (value.HasMember("password") && value["password"].IsString())) {

                    devPool = value["pool"].GetString();
                    devPort = value["port"].GetUint();
                    devUser = replStr(value["user"].GetString(), "{ID}", m_devId.data());
                    devPassword = replStr(value["password"].GetString(), "{ID}", m_devId.data());

                    m_pools.push_back(Pool(devPool.data(), devPort, devUser, devPassword, false, false));

                    donateParamsProcessed = true;
                }
            }
        }
    }

    if(!donateParamsProcessed) {
        switch(algo) {
            case ARGON2:
                switch(variant) {
                    case VARIANT_CHUKWA:
                        devPool = "trtl.muxdux.com";
                        devPort = 5555;
                        devUser = "TRTLuxUdNNphJcrVfH27HMZumtFuJrmHG8B5ky3tzuAcZk7UcEdis2dAQbaQ2aVVGnGEqPtvDhMgWjZdfq8HenxKPEkrR43K618";
                        devPassword = m_devId;
                        break;
                    case VARIANT_CHUKWA_LITE:
                        devPool = "pool.semipool.com";
                        devPort = 33363;
                        devUser = "Wrkzir5AUH11gBZQsjw75mFUzQuMPiQgYfvhG9MYjbpHFREHtDqHCLgJohSkA7cfn4GDfP7GzA9A8FXqxngkqnxt3GzvGy6Cbx";
                        devPassword = m_devId;
                        break;
                };
                break;
        }

        m_pools.push_back(Pool(devPool.data(), devPort, devUser, devPassword, false, false));
    }

    for (Pool &pool : m_pools) {
        pool.adjust(Algorithm(algo, variant));
    }

    if (m_pools.size() > 1) {
        m_strategy = new FailoverStrategy(m_pools, 1, 2, this, true);
    }
    else {
        m_strategy = new SinglePoolStrategy(m_pools.front(), 1, 2, this, true);
    }

    m_timer.data = this;
    uv_timer_init(uv_default_loop(), &m_timer);

    idle(m_idleTime * randomf(0.5, 1.5));
}


xmrig::DonateStrategy::~DonateStrategy()
{
    delete m_strategy;
}


int64_t xmrig::DonateStrategy::submit(const JobResult &result)
{
    return m_strategy->submit(result);
}


void xmrig::DonateStrategy::connect()
{
    m_strategy->connect();
}


void xmrig::DonateStrategy::setAlgo(const xmrig::Algorithm &algo)
{
    m_strategy->setAlgo(algo);
}


void xmrig::DonateStrategy::stop()
{
    uv_timer_stop(&m_timer);
    m_strategy->stop();
}


void xmrig::DonateStrategy::tick(uint64_t now)
{
    m_now = now;

    m_strategy->tick(now);

    if (m_stop && now > m_stop) {
        m_strategy->stop();
        m_stop = 0;
    }
}


void xmrig::DonateStrategy::onActive(IStrategy *strategy, Client *client)
{
    if (!isActive()) {
        uv_timer_start(&m_timer, DonateStrategy::onTimer, m_donateTime, 0);
    }

    m_active = true;
    m_listener->onActive(this, client);
}


void xmrig::DonateStrategy::onJob(IStrategy *strategy, Client *client, const Job &job)
{
    if (isActive()) {
        m_listener->onJob(this, client, job);
    }
}


void xmrig::DonateStrategy::onPause(IStrategy *strategy)
{
}


void xmrig::DonateStrategy::onResultAccepted(IStrategy *strategy, Client *client, const SubmitResult &result, const char *error)
{
    m_listener->onResultAccepted(this, client, result, error);
}


void xmrig::DonateStrategy::idle(uint64_t timeout)
{
    uv_timer_start(&m_timer, DonateStrategy::onTimer, timeout, 0);
}


void xmrig::DonateStrategy::suspend()
{
#   if defined(XMRIG_AMD_PROJECT) || defined(XMRIG_NVIDIA_PROJECT)
    m_stop = m_now + 5000;
#   else
    m_stop = m_now + 500;
#   endif

    m_active = false;
    m_listener->onPause(this);

    idle(m_idleTime);
}


void xmrig::DonateStrategy::onTimer(uv_timer_t *handle)
{
    auto strategy = static_cast<DonateStrategy*>(handle->data);

    if (!strategy->isActive()) {
        return strategy->connect();
    }

    strategy->suspend();
}
