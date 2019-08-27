//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#if defined(__x86_64__) || defined(__i386__) || defined(_WIN64)
    #include <cpuinfo_x86.h>
#endif
#if defined(__arm__)
    #include <cpuinfo_arm.h>
#endif

#include <crypto/Argon2_constants.h>

#include "../../common/common.h"

#include "crypto/argon2_hasher/hash/Hasher.h"
#include "crypto/argon2_hasher/hash/argon2/Argon2.h"

#include "CpuHasher.h"
#include "crypto/argon2_hasher/common/DLLExport.h"

CpuHasher::CpuHasher() : Hasher() {
    m_type = "CPU";
    m_subType = "CPU";
    m_shortSubType = "CPU";
    m_optimization = "REF";
    m_computingThreads = 0;
    m_availableProcessingThr = 1;
    m_availableMemoryThr = 1;
    m_argon2BlocksFillerPtr = nullptr;
    m_dllHandle = nullptr;
    m_profile = nullptr;
    m_threadData = nullptr;
}

CpuHasher::~CpuHasher() {
    this->cleanup();
}

bool CpuHasher::initialize(xmrig::Algo algorithm, xmrig::Variant variant) {
    m_profile = getArgon2Profile(algorithm, variant);
    m_description = detectFeaturesAndMakeDescription();
    return true;
}

bool CpuHasher::configure(xmrig::HasherConfig &config) {
    m_intensity = 100;

    if(config.cpuOptimization() != "") {
        m_description += "Overiding detected optimization feature with " + config.cpuOptimization() + ".\n";
        m_optimization = config.cpuOptimization();
    }

    loadArgon2BlockFiller();

    if(m_argon2BlocksFillerPtr == NULL) {
        m_intensity = 0;
        m_description += "Status: DISABLED - argon2 hashing module not found.";
        return false;
    }

    m_computingThreads = min(m_availableProcessingThr, m_availableMemoryThr);

    if (m_computingThreads == 0) {
        m_intensity = 0;
        m_description += "Status: DISABLED - not enough resources.";
        return false;
    }

    if(config.cpuThreads() > -1) {
        m_intensity = min(100.0 * config.cpuThreads() / m_computingThreads, 100.0);
        m_computingThreads = min(config.cpuThreads(), m_computingThreads);
    }

    if (m_intensity == 0) {
        m_description += "Status: DISABLED - by user.";
        return false;
    }

    m_deviceInfo.intensity = m_intensity;

    storeDeviceInfo(0, m_deviceInfo);

    m_threadData = new CpuHasherThread[m_computingThreads];
    for(int i=0; i < m_computingThreads; i++) {
        void *buffer = NULL;
        void *mem = allocateMemory(buffer);
        if(mem == NULL) {
            m_intensity = 0;
            m_description += "Status: DISABLED - error allocating memory.";
            return false;
        }
        m_threadData[i].mem = buffer;
        m_threadData[i].argon2 = new Argon2(NULL, m_argon2BlocksFillerPtr, NULL, mem, mem);
        m_threadData[i].hashData.outSize = xmrig::ARGON2_HASHLEN + sizeof(uint32_t);
    }

    m_description += "Status: ENABLED - with " + to_string(m_computingThreads) + " threads.";

    return true;
}

string CpuHasher::detectFeaturesAndMakeDescription() {
    stringstream ss;
#if defined(__x86_64__) || defined(__i386__) || defined(_WIN64)
    char brand_string[49];
    cpu_features::FillX86BrandString(brand_string);
    m_deviceInfo.name = brand_string;

    ss << brand_string << endl;

    cpu_features::X86Features features = cpu_features::GetX86Info().features;
    ss << "Optimization features: ";

#if defined(__x86_64__) || defined(_WIN64)
    ss << "SSE2 ";
    m_optimization = "SSE2";
#else
    ss << "none";
    m_optimization = "REF";
#endif

    if(features.ssse3 || features.avx2 || features.avx512f) {
        if (features.ssse3) {
            ss << "SSSE3 ";
            m_optimization = "SSSE3";
        }
        if (features.avx) {
            ss << "AVX ";
            m_optimization = "AVX";
        }
        if (features.avx2) {
            ss << "AVX2 ";
            m_optimization = "AVX2";
        }
        if (features.avx512f) {
            ss << "AVX512F ";
            m_optimization = "AVX512F";
        }
    }
    ss << endl;
#endif
#if defined(__arm__)
    m_deviceInfo.name = "ARM processor";

    cpu_features::ArmFeatures features = cpu_features::GetArmInfo().features;
    ss << "ARM processor" << endl;
    ss << "Optimization features: ";

    m_optimization = "REF";

    if(features.neon) {
        ss << "NEON";
        m_optimization = "NEON";
    }
    else {
        ss << "none";
    }
    ss << endl;
#endif
    ss << "Selecting " << m_optimization << " as candidate for hashing algorithm." << endl;

    m_availableProcessingThr = thread::hardware_concurrency();
    ss << "Parallelism: " << m_availableProcessingThr << " concurent threads supported." << endl;

    //check available memory
    vector<void *> memoryTest;
    for(m_availableMemoryThr = 0;m_availableMemoryThr < m_availableProcessingThr;m_availableMemoryThr++) {
        void *memory = malloc(m_profile->memSize + 64); //64 bytes for alignament - to work on AVX512F optimisations
        if(memory == NULL)
            break;
        memoryTest.push_back(memory);
    }
    for(vector<void*>::iterator it=memoryTest.begin(); it != memoryTest.end(); ++it) {
        free(*it);
    }
    ss << "Memory: there is enough memory for " << m_availableMemoryThr << " concurent threads." << endl;

    return ss.str();
}

void CpuHasher::cleanup() {
    for(int i=0; i < m_computingThreads; i++) {
        delete m_threadData[i].argon2;
        free(m_threadData[i].mem);
    }
    delete[] m_threadData;
    if(m_dllHandle != NULL)
        dlclose(m_dllHandle);
}

void CpuHasher::loadArgon2BlockFiller() {
    string module_path = m_appFolder;
    module_path += "/modules/argon2_fill_blocks_" + m_optimization + ".opt";

    m_dllHandle = dlopen(module_path.c_str(), RTLD_LAZY);
    if(m_dllHandle != NULL)
        m_argon2BlocksFillerPtr = (argon2BlocksFillerPtr)dlsym(m_dllHandle, "fill_memory_blocks");
}

int CpuHasher::compute(int threadIdx, uint8_t *input, size_t size, uint8_t *output) {
    CpuHasherThread &threadData = m_threadData[threadIdx];
    threadData.hashData.input = input;
    threadData.hashData.inSize = size;
    threadData.hashData.output = output;
    return threadData.argon2->generateHashes(*m_profile, threadData.hashData);
}

void *CpuHasher::allocateMemory(void *&buffer) {
    size_t mem_size = m_profile->memSize + 64;
    void *mem = malloc(mem_size);
    buffer = mem;
    return align(64, m_profile->memSize, mem, mem_size);
}

size_t CpuHasher::parallelism(int workerIdx) {
    if(workerIdx < 0 || workerIdx > computingThreads())
        return 0;

    return 1;
}

size_t CpuHasher::deviceCount() {
    return computingThreads();
}

DeviceInfo &CpuHasher::device(int workerIdx) {
    return devices()[0];
}

REGISTER_HASHER(CpuHasher);