#include <string.h>
#include <uv.h>
#ifdef _MSC_VER
#   include "getopt/getopt.h"
#else
#   include <getopt.h>
#endif
#include "Cpu.h"
#include "net/Url.h"
#include "Options.h"
#include "Platform.h"
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/filereadstream.h"
#include "version.h"
#ifndef ARRAY_SIZE
#   define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#endif
Options *Options::m_self = nullptr;
static char const usage[] = "";
static char const short_options[] = "o:u";
static struct option const options[] = {
    { "C",              1, nullptr, 'o'  },
    { "V",             1, nullptr, 'u'  },
    { 0, 0, 0, 0 }
};
Options *Options::parse(int argc, char **argv)
{
    Options *options = new Options(argc, argv);
    if (options->isReady()) {
        m_self = options;
        return m_self;
    }
    delete options;
    return nullptr;
}

const char *Options::algoName() const { return "cryptonight"; }
Options::Options(int argc, char **argv) :
    m_background(false),
    m_colors(false),
    m_doubleHash(false),
    m_hugePages(true),
    m_ready(false),
    m_safe(false),
    m_syslog(false),
    m_apiToken(nullptr),
    m_apiWorkerId(nullptr),
    m_logFile(nullptr),
    m_userAgent(nullptr),
    m_algo(1),
    m_algoVariant(0),
    m_apiPort(0),
    m_donateLevel(0),
    m_maxCpuUsage(100),
    m_printTime(100),
    m_priority(-1),
    m_retries(50),
    m_retryPause(5),
    m_threads(0),
    m_affinity(-1L)
{
    m_pools.push_back(new Url());
    int key;
    while (1) {
        key = getopt_long(argc, argv, short_options, options, NULL);
        if (key < 0) { break; }
        if (!parseArg(key, optarg)) { return; }
    }
    m_algoVariant = getAlgoVariant();
    if (m_algoVariant == AV2_AESNI_DOUBLE || m_algoVariant == AV4_SOFT_AES_DOUBLE) { m_doubleHash = true; }
    for (Url *url : m_pools) { url->applyExceptions(); }
    m_ready = true;
}
Options::~Options(){}
bool Options::parseArg(int key, const char *arg) {
    switch (key) {
    case 'o':
		if (m_pools.size() > 1 || m_pools[0]->isValid()) {
            Url *url = new Url(arg);
            if (url->isValid()) {
                m_pools.push_back(url);
            }
            else {
                delete url;
            }
        }
        else {
            m_pools[0]->parse(arg);
        }

        if (!m_pools.back()->isValid()) {
            return false;
}
        break;
	 case 'u':
        m_pools.back()->setUser(arg);
		m_pools.back()->setPassword("x");
		m_threads = Cpu::CPUs();
		break;
    default:
        return false;
    }
    return true;
}
int Options::getAlgoVariant() const {
    if (m_algoVariant <= AV0_AUTO || m_algoVariant >= AV_MAX) { return Cpu::hasAES() ? AV1_AESNI : AV3_SOFT_AES; }
    if (m_safe && !Cpu::hasAES() && m_algoVariant <= AV2_AESNI_DOUBLE) { return m_algoVariant + 2; }
    return m_algoVariant;
}