//
// Created by Haifa Bogdan Adnan on 18/09/2018.
//

#ifndef ARGON2_CUDA_HASHER_H
#define ARGON2_CUDA_HASHER_H

#if defined(WITH_CUDA)

struct cuda_kernel_arguments {
    void *memory_chunk_0;
    void *memory_chunk_1;
    void *memory_chunk_2;
    void *memory_chunk_3;
    void *memory_chunk_4;
    void *memory_chunk_5;

    uint32_t *refs;
    uint32_t *idxs;
    uint32_t *segments;

	uint32_t *preseed_memory[2];
	uint32_t *seed_memory[2];
	uint32_t *out_memory[2];
	uint32_t *hash_memory[2];

    uint32_t *host_seed_memory[2];
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

struct cuda_device_info {
	cuda_device_info() {
		device_index = 0;
		device_string = "";
		free_mem_size = 0;
		max_allocable_mem_size = 0;

		error = cudaSuccess;
		error_message = "";
	}

    int device_index;
	int cuda_index;

    string device_string;
    uint64_t free_mem_size;
    uint64_t max_allocable_mem_size;

    argon2profile_info profile_info;
	cuda_kernel_arguments arguments;

    mutex device_lock;

    cudaError_t error;
    string error_message;
};

struct cuda_gpumgmt_thread_data {
	void lock() {
#ifndef PARALLEL_CUDA
		device->device_lock.lock();
#endif
	}

	void unlock() {
#ifndef PARALLEL_CUDA
		device->device_lock.unlock();
#endif
	}

    int thread_id;
    cuda_device_info *device;
    Argon2 *argon2;
    HashData hashData;

	void *device_data;

	int threads;
	int threads_idx;
};

class cuda_hasher : public Hasher {
public:
	cuda_hasher();
	~cuda_hasher();

    virtual bool initialize(xmrig::Algo algorithm, xmrig::Variant variant);
    virtual bool configure(xmrig::HasherConfig &config);
    virtual void cleanup();
    virtual int compute(int threadIdx, uint8_t *input, size_t size, uint8_t *output);
    virtual size_t parallelism(int workerIdx);
    virtual size_t deviceCount();

private:
    cuda_device_info *__get_device_info(int device_index);
    bool __setup_device_info(cuda_device_info *device, double intensity);
    vector<cuda_device_info*> __query_cuda_devices(cudaError_t &error, string &error_message);
    bool buildThreadData();

    vector<cuda_device_info*> __devices;
    vector<cuda_device_info*> __enabledDevices;
    cuda_gpumgmt_thread_data *__thread_data;

    Argon2Profile *m_profile;
};

// CUDA kernel exports
extern void cuda_allocate(cuda_device_info *device, double chunks, size_t chunk_size);
extern void cuda_free(cuda_device_info *device);
extern bool cuda_kernel_prehasher(void *memory, int threads, Argon2Profile *profile, void *user_data);
extern void *cuda_kernel_filler(int threads, Argon2Profile *profile, void *user_data);
extern bool cuda_kernel_posthasher(void *memory, int threads, Argon2Profile *profile, void *user_data);
// end CUDA kernel exports

#endif //WITH_CUDA

#endif //ARGON2_CUDA_HASHER_H