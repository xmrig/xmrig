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

struct opencl_kernel_arguments {
    cl_mem memory_chunk_0;
    cl_mem memory_chunk_1;
    cl_mem memory_chunk_2;
    cl_mem memory_chunk_3;
    cl_mem memory_chunk_4;
    cl_mem memory_chunk_5;
    cl_mem refs;
    cl_mem idxs;
    cl_mem segments;
    cl_mem preseed_memory[2];
    cl_mem seed_memory[2];
    cl_mem out_memory[2];
    cl_mem hash_memory[2];
};

struct argon2profile_info {
    argon2profile_info() {
        threads = 0;
        threads_per_chunk = 0;
    }

    uint32_t threads;
    uint32_t threads_per_chunk;
    Argon2Profile *profile;
};

struct opencl_device_info {
    opencl_device_info(cl_int err, const string &err_msg) {
        error = err;
        error_message = err_msg;
    }

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    cl_program program;
    cl_kernel kernel_prehash;
    cl_kernel kernel_fill_blocks;
    cl_kernel kernel_posthash;

    int device_index;

    opencl_kernel_arguments arguments;
    argon2profile_info profile_info;

    string device_string;
    uint64_t max_mem_size;
    uint64_t max_allocable_mem_size;

    cl_int error;
    string error_message;

    mutex device_lock;
};

struct opencl_gpumgmt_thread_data {
    int thread_id;
    opencl_device_info *device;
    Argon2 *argon2;
    HashData hashData;
};

class opencl_hasher : public Hasher {
public:
    opencl_hasher();
    ~opencl_hasher();

    virtual bool initialize(xmrig::Algo algorithm, xmrig::Variant variant);
    virtual bool configure(xmrig::HasherConfig &config);
    virtual void cleanup();
    virtual int compute(int threadIdx, uint8_t *input, size_t size, uint8_t *output);
    virtual size_t parallelism(int workerIdx);
    virtual size_t deviceCount();

private:
    opencl_device_info *__get_device_info(cl_platform_id platform, cl_device_id device);
    bool __setup_device_info(opencl_device_info *device, double intensity);
    vector<opencl_device_info*> __query_opencl_devices(cl_int &error, string &error_message);
    void buildThreadData();

    vector<opencl_device_info*> __devices;
    vector<opencl_device_info*> __enabledDevices;
    opencl_gpumgmt_thread_data *__thread_data;

    Argon2Profile *m_profile;
};

#endif //WITH_OPENCL

#endif //ARGON2_OPENCL_HASHER_H
