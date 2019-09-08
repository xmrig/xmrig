//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#ifndef ARGON2_OPENCL_HASHER_H
#define ARGON2_OPENCL_HASHER_H

#if defined(WITH_OPENCL)

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif // !__APPLE__

struct OpenCLKernelArguments {
    cl_mem memoryChunk_0;
    cl_mem memoryChunk_1;
    cl_mem memoryChunk_2;
    cl_mem memoryChunk_3;
    cl_mem memoryChunk_4;
    cl_mem memoryChunk_5;
    cl_mem refs;
    cl_mem idxs;
    cl_mem segments;
    cl_mem preseedMemory[2];
    cl_mem seedMemory[2];
    cl_mem outMemory[2];
    cl_mem hashMemory[2];
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

struct OpenCLDeviceInfo {
    OpenCLDeviceInfo(cl_int err, const string &err_msg) {
        error = err;
        errorMessage = err_msg;
    }

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue[2];

    cl_program program;
    cl_kernel kernelPrehash[2];
    cl_kernel kernelFillBlocks[2];
    cl_kernel kernelPosthash[2];

    int deviceIndex;

    OpenCLKernelArguments arguments;
    Argon2ProfileInfo profileInfo;

    string deviceString;
    string deviceExtensions;
    uint64_t maxMemSize;
    uint64_t maxAllocableMemSize;

    cl_int error;
    string errorMessage;

    mutex deviceLock;
};

struct OpenCLGpuMgmtThreadData {
    void lock() {
#ifndef PARALLEL_OPENCL
        device->deviceLock.lock();
#endif
    }

    void unlock() {
#ifndef PARALLEL_OPENCL
        device->deviceLock.unlock();
#endif
    }
    int threadId;
    OpenCLDeviceInfo *device;
    Argon2 *argon2;
    HashData hashData;
    int threads;
    int threadsIdx;
};

class OpenCLHasher : public Hasher {
public:
    OpenCLHasher();
    ~OpenCLHasher();

    virtual bool initialize(xmrig::Algo algorithm, xmrig::Variant variant);
    virtual bool configure(xmrig::HasherConfig &config);
    virtual void cleanup();
    virtual int compute(int threadIdx, uint8_t *input, size_t size, uint8_t *output);
    virtual size_t parallelism(int workerIdx);
    virtual size_t deviceCount();
    virtual DeviceInfo &device(int workerIdx);

private:
    OpenCLDeviceInfo *getDeviceInfo(cl_platform_id platform, cl_device_id device);
    bool setupDeviceInfo(OpenCLDeviceInfo *device, double intensity);
    vector<OpenCLDeviceInfo*> queryOpenCLDevices(cl_int &error, string &error_message);
    void buildThreadData();

    vector<OpenCLDeviceInfo*> m_devices;
    vector<OpenCLDeviceInfo*> m_enabledDevices;
    OpenCLGpuMgmtThreadData *m_threadData;

    Argon2Profile *m_profile;
};

#endif //WITH_OPENCL

#endif //ARGON2_OPENCL_HASHER_H
