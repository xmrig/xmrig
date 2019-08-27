//
// Created by Haifa Bogdan Adnan on 18/09/2018.
//

#ifndef ARGON2_CUDA_HASHER_H
#define ARGON2_CUDA_HASHER_H

#if defined(WITH_CUDA)

struct CudaKernelArguments {
    void *memoryChunk_0;
    void *memoryChunk_1;
    void *memoryChunk_2;
    void *memoryChunk_3;
    void *memoryChunk_4;
    void *memoryChunk_5;

    uint32_t *refs;
    uint32_t *idxs;
    uint32_t *segments;

	uint32_t *preseedMemory[2];
	uint32_t *seedMemory[2];
	uint32_t *outMemory[2];
	uint32_t *hashMemory[2];

    uint32_t *hostSeedMemory[2];
};

struct Argon2ProfileInfo {
    Argon2ProfileInfo() {
        threads = 0;
        threads_per_chunk = 0;
    }
    uint32_t threads;
    uint32_t threads_per_chunk;
    Argon2Profile *profile;
};

struct CudaDeviceInfo {
	CudaDeviceInfo() {
        deviceIndex = 0;
        deviceString = "";
        freeMemSize = 0;
        maxAllocableMemSize = 0;

		error = cudaSuccess;
        errorMessage = "";
	}

    int deviceIndex;
	int cudaIndex;

    string deviceString;
    uint64_t freeMemSize;
    uint64_t maxAllocableMemSize;

    Argon2ProfileInfo profileInfo;
	CudaKernelArguments arguments;

    mutex deviceLock;

    cudaError_t error;
    string errorMessage;
};

struct CudaGpuMgmtThreadData {
	void lock() {
#ifndef PARALLEL_CUDA
		device->deviceLock.lock();
#endif
	}

	void unlock() {
#ifndef PARALLEL_CUDA
		device->deviceLock.unlock();
#endif
	}

    int threadId;
    CudaDeviceInfo *device;
    Argon2 *argon2;
    HashData hashData;

	void *deviceData;

	int threads;
	int threadsIdx;
};

class CudaHasher : public Hasher {
public:
	CudaHasher();
	~CudaHasher();

    virtual bool initialize(xmrig::Algo algorithm, xmrig::Variant variant);
    virtual bool configure(xmrig::HasherConfig &config);
    virtual void cleanup();
    virtual int compute(int threadIdx, uint8_t *input, size_t size, uint8_t *output);
    virtual size_t parallelism(int workerIdx);
    virtual size_t deviceCount();
    virtual DeviceInfo &device(int workerIdx);

private:
    CudaDeviceInfo *getDeviceInfo(int device_index);
    bool setupDeviceInfo(CudaDeviceInfo *device, double intensity);
    vector<CudaDeviceInfo*> queryCudaDevices(cudaError_t &error, string &error_message);
    bool buildThreadData();

    vector<CudaDeviceInfo*> m_devices;
    vector<CudaDeviceInfo*> m_enabledDevices;
    CudaGpuMgmtThreadData *m_threadData;

    Argon2Profile *m_profile;
};

// CUDA kernel exports
extern void cuda_allocate(CudaDeviceInfo *device, double chunks, size_t chunk_size);
extern void cuda_free(CudaDeviceInfo *device);
extern bool cuda_kernel_prehasher(void *memory, int threads, Argon2Profile *profile, void *user_data);
extern void *cuda_kernel_filler(int threads, Argon2Profile *profile, void *user_data);
extern bool cuda_kernel_posthasher(void *memory, int threads, Argon2Profile *profile, void *user_data);
// end CUDA kernel exports

#endif //WITH_CUDA

#endif //ARGON2_CUDA_HASHER_H