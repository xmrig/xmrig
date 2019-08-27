//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#ifndef ARGON2_HASHER_H
#define ARGON2_HASHER_H

#include "crypto/argon2_hasher/hash/argon2/Defs.h"
#include "../../../core/HasherConfig.h"
#include "../../../common/xmrig.h"

struct DeviceInfo {
	string name;
	string bus_id;
	double intensity;
};

#define REGISTER_HASHER(x)        extern "C"  { DLLEXPORT void hasherLoader() { x *instance = new x(); } }

class DLLEXPORT Hasher {
public:
    Hasher();
    virtual ~Hasher();

    virtual bool initialize(xmrig::Algo algorithm, xmrig::Variant variant) = 0;
    virtual bool configure(xmrig::HasherConfig &config) = 0;
    virtual void cleanup() = 0;
    virtual int compute(int threadIdx, uint8_t *input, size_t size, uint8_t *output) = 0;
    virtual size_t parallelism(int workerIdx) = 0;
    virtual size_t deviceCount() = 0;
    virtual DeviceInfo &device(int workerIdx) = 0;

    string type();
	string subType(bool shortName = false);

    string info();
    int computingThreads();

    map<int, DeviceInfo> &devices();

    static vector<Hasher*> getHashers(const string &type);
    static vector<Hasher*> getHashers();
    static vector<Hasher*> getActiveHashers();
    static void loadHashers(const string &appPath);

protected:
    double m_intensity;
    string m_type;
	string m_subType;
	string m_shortSubType; //max 3 characters
    string m_description;
    int m_computingThreads;
    static string m_appFolder;

	void storeDeviceInfo(int deviceId, DeviceInfo device);
    Argon2Profile *getArgon2Profile(xmrig::Algo algorithm, xmrig::Variant variant);

private:
    static vector<Hasher*> *m_registeredHashers;
    map<int, DeviceInfo> m_deviceInfos;
    mutex m_deviceInfosMutex;
};

#endif //ARGON2_HASHER_H
