//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#include "../common/common.h"
#include "../crypt/base64.h"
#include "../crypt/hex.h"
#include "../crypt/random_generator.h"

#include "crypto/argon2_hasher/common/DLLExport.h"
#include "crypto/argon2_hasher/hash/argon2/Argon2.h"
#include "Hasher.h"

vector<Hasher *> *Hasher::m_registeredHashers = NULL;
string Hasher::m_appFolder = "";

typedef void (*hasherLoader)();

Hasher::Hasher() {
    m_intensity = 0;
    m_type = "";
	m_subType = "";
	m_shortSubType = "";
    m_description = "";

    m_computingThreads = 1;

    if(m_registeredHashers == NULL) {
        m_registeredHashers = new vector<Hasher*>();
    }

    m_registeredHashers->push_back(this);
}

Hasher::~Hasher() {};

string Hasher::type() {
	return m_type;
}

string Hasher::subType(bool shortName) {
    if(shortName && !(m_shortSubType.empty())) {
        string shortVersion = m_shortSubType;
        shortVersion.erase(3);
        return shortVersion;
    }
    else
    	return m_subType;
}

string Hasher::info() {
    return m_description;
}

int Hasher::computingThreads() {
    return m_computingThreads;
}

void Hasher::loadHashers(const string &appPath) {
    m_registeredHashers = new vector<Hasher*>();

    string modulePath = ".";

    size_t lastSlash = appPath.find_last_of("/\\");
    if (lastSlash != string::npos) {
        modulePath = appPath.substr(0, lastSlash);
        if(modulePath.empty()) {
            modulePath = ".";
        }
    }

    m_appFolder = modulePath;

    modulePath += "/modules/";

    vector<string> files = getFiles(modulePath);
    for(string file : files) {
        if(file.find(".hsh") != string::npos) {
            void *dllHandle = dlopen((modulePath + file).c_str(), RTLD_LAZY);
            if(dllHandle != NULL) {
                hasherLoader hasherLoaderPtr = (hasherLoader) dlsym(dllHandle, "hasherLoader");
                (*hasherLoaderPtr)();
            }
        }
    }
}

vector<Hasher *> Hasher::getHashers() {
    return *m_registeredHashers;
}

vector<Hasher *> Hasher::getActiveHashers() {
    vector<Hasher *> filtered;
    for(Hasher *hasher : *m_registeredHashers) {
        if(hasher->m_intensity != 0)
            filtered.push_back(hasher);
    }
    return filtered;
}

vector<Hasher *> Hasher::getHashers(const string &type) {
    vector<Hasher *> filtered;
    for(Hasher *hasher : *m_registeredHashers) {
        if(hasher->m_type == type)
            filtered.push_back(hasher);
    }
    return filtered;
}

map<int, DeviceInfo> &Hasher::devices() {
    return m_deviceInfos;
}

void Hasher::storeDeviceInfo(int deviceId, DeviceInfo device) {
    m_deviceInfosMutex.lock();
    m_deviceInfos[deviceId] = device;
    m_deviceInfosMutex.unlock();
}

Argon2Profile *Hasher::getArgon2Profile(xmrig::Algo algorithm, xmrig::Variant variant) {
    if(algorithm == xmrig::ARGON2) {
        switch(variant) {
            case xmrig::VARIANT_CHUKWA:
                return &argon2profile_3_1_512;
            case xmrig::VARIANT_CHUKWA_LITE:
                return &argon2profile_4_1_256;
            default:
                return nullptr;
        }
    }
    return nullptr;
}
