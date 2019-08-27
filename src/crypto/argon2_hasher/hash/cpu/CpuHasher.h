//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#ifndef ARGON2_CPU_HASHER_H
#define ARGON2_CPU_HASHER_H

struct CpuHasherThread {
    Argon2 *argon2;
    HashData hashData;
    void *mem;
};

class CpuHasher : public Hasher {
public:
    CpuHasher();
    ~CpuHasher();

    virtual bool initialize(xmrig::Algo algorithm, xmrig::Variant variant);
    virtual bool configure(xmrig::HasherConfig &config);
    virtual void cleanup();
    virtual int compute(int threadIdx, uint8_t *input, size_t size, uint8_t *output);
    virtual size_t parallelism(int workerIdx);
    virtual size_t deviceCount();
    virtual DeviceInfo &device(int workerIdx);

private:
    string detectFeaturesAndMakeDescription();
    void loadArgon2BlockFiller();
    void *allocateMemory(void *&buffer);

    DeviceInfo m_deviceInfo;
    string m_optimization;
    int m_availableProcessingThr;
    int m_availableMemoryThr;
    void *m_dllHandle;
    Argon2Profile *m_profile;
    argon2BlocksFillerPtr m_argon2BlocksFillerPtr;
    CpuHasherThread *m_threadData;
};

#endif //ARGON2_CPU_HASHER_H
