//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#include <crypto/Argon2_constants.h>

#include "../../../common/common.h"

#include "crypto/argon2_hasher/hash/Hasher.h"
#include "crypto/argon2_hasher/hash/argon2/Argon2.h"

#if defined(WITH_CUDA)

#include <cuda_runtime.h>
#include <driver_types.h>

#include "cuda_hasher.h"
#include "../../../common/DLLExport.h"

cuda_hasher::cuda_hasher() {
    m_type = "GPU";
    m_subType = "CUDA";
    m_shortSubType = "NVD";
    m_intensity = 0;
    m_description = "";
    m_computingThreads = 0;
}


cuda_hasher::~cuda_hasher() {
	this->cleanup();
}

bool cuda_hasher::initialize(xmrig::Algo algorithm, xmrig::Variant variant) {
	cudaError_t error = cudaSuccess;
	string error_message;

    m_profile = getArgon2Profile(algorithm, variant);

	__devices = __query_cuda_devices(error, error_message);

	if(error != cudaSuccess) {
		m_description = "No compatible GPU detected: " + error_message;
		return false;
	}

	if (__devices.empty()) {
		m_description = "No compatible GPU detected.";
		return false;
	}

	return true;
}

vector<cuda_device_info *> cuda_hasher::__query_cuda_devices(cudaError_t &error, string &error_message) {
	vector<cuda_device_info *> devices;
	int devCount = 0;
	error = cudaGetDeviceCount(&devCount);

	if(error != cudaSuccess) {
		error_message = "Error querying CUDA device count.";
		return devices;
	}

	if(devCount == 0)
		return devices;

	for (int i = 0; i < devCount; ++i)
	{
		cuda_device_info *dev = __get_device_info(i);
		if(dev == NULL)
			continue;
		if(dev->error != cudaSuccess) {
			error = dev->error;
			error_message = dev->error_message;
			continue;
		}
		devices.push_back(dev);
	}
	return devices;
}

cuda_device_info *cuda_hasher::__get_device_info(int device_index) {
	cuda_device_info *device_info = new cuda_device_info();
	device_info->error = cudaSuccess;
	device_info->cuda_index = device_index;

	device_info->error = cudaSetDevice(device_index);
	if(device_info->error != cudaSuccess) {
		device_info->error_message = "Error setting current device.";
		return device_info;
	}

    cudaDeviceProp devProp;
	device_info->error = cudaGetDeviceProperties(&devProp, device_index);
	if(device_info->error != cudaSuccess) {
		device_info->error_message = "Error setting current device.";
		return device_info;
	}

    device_info->device_string = devProp.name;

    size_t freemem, totalmem;
    device_info->error = cudaMemGetInfo(&freemem, &totalmem);
	if(device_info->error != cudaSuccess) {
		device_info->error_message = "Error setting current device.";
		return device_info;
	}

    device_info->free_mem_size = freemem;
    device_info->max_allocable_mem_size = freemem / 4;

    double mem_in_gb = totalmem / 1073741824.0;
    stringstream ss;
    ss << setprecision(2) << mem_in_gb;
    device_info->device_string += (" (" + ss.str() + "GB)");

    return device_info;
}

bool cuda_hasher::configure(xmrig::HasherConfig &config) {
    int index = config.getGPUCardsCount();
    double intensity = 0;

    int total_threads = 0;
    intensity = config.getAverageGPUIntensity();

	if (intensity == 0) {
		m_intensity = 0;
		m_description = "Status: DISABLED - by user.";
		return false;
	}

	bool cards_selected = false;
	intensity = 0;

	for(vector<cuda_device_info *>::iterator d = __devices.begin(); d != __devices.end(); d++, index++) {
		stringstream ss;
		ss << "["<< (index + 1) << "] " << (*d)->device_string;
		string device_description = ss.str();
		(*d)->device_index = index;
        (*d)->profile_info.profile = m_profile;

        if(config.gpuFilter().size() > 0) {
			bool found = false;
            for(xmrig::GPUFilter fit : config.gpuFilter()) {
                if(device_description.find(fit.filter) != string::npos) {
                    found = true;
                    break;
                }
            }
			if(!found) {
				(*d)->profile_info.threads = 0;
				ss << " - DISABLED" << endl;
				m_description += ss.str();
				continue;
			}
			else {
				cards_selected = true;
			}
		}
		else {
			cards_selected = true;
		}

		ss << endl;

        double device_intensity = config.getGPUIntensity((*d)->device_index);

		m_description += ss.str();

		if(!(__setup_device_info((*d), device_intensity))) {
			m_description += (*d)->error_message;
			m_description += "\n";
			continue;
		};

		DeviceInfo device;

		char bus_id[100];
		if(cudaDeviceGetPCIBusId(bus_id, 100, (*d)->cuda_index) == cudaSuccess) {
			device.bus_id = bus_id;
			int domain_separator = device.bus_id.find(":");
			if(domain_separator != string::npos) {
				device.bus_id.erase(0, domain_separator + 1);
			}
		}

		device.name = (*d)->device_string;
		device.intensity = device_intensity;
        storeDeviceInfo((*d)->device_index, device);

        __enabledDevices.push_back(*d);

		total_threads += (*d)->profile_info.threads;
        intensity += device_intensity;
	}

    config.addGPUCardsCount(index - config.getGPUCardsCount());

	if(!cards_selected) {
		m_intensity = 0;
		m_description += "Status: DISABLED - no card enabled because of filtering.";
		return false;
	}

	if (total_threads == 0) {
		m_intensity = 0;
		m_description += "Status: DISABLED - not enough resources.";
		return false;
	}

    if(!buildThreadData())
        return false;

    m_intensity = intensity / __enabledDevices.size();
    m_computingThreads = __enabledDevices.size() * 2; // 2 computing threads for each device
    m_description += "Status: ENABLED - with " + to_string(total_threads) + " threads.";

	return true;
}

void cuda_hasher::cleanup() {
	for(vector<cuda_device_info *>::iterator d = __devices.begin(); d != __devices.end(); d++) {
		cuda_free(*d);
	}
}

bool cuda_hasher::__setup_device_info(cuda_device_info *device, double intensity) {
    device->profile_info.threads_per_chunk = (uint32_t)(device->max_allocable_mem_size / device->profile_info.profile->memSize);
    size_t chunk_size = device->profile_info.threads_per_chunk * device->profile_info.profile->memSize;

    if(chunk_size == 0) {
        device->error = cudaErrorInitializationError;
        device->error_message = "Not enough memory on GPU.";
        return false;
    }

    uint64_t usable_memory = device->free_mem_size;
    double chunks = (double)usable_memory / (double)chunk_size;

    uint32_t max_threads = (uint32_t)(device->profile_info.threads_per_chunk * chunks);

    if(max_threads == 0) {
        device->error = cudaErrorInitializationError;
        device->error_message = "Not enough memory on GPU.";
        return false;
    }

    device->profile_info.threads = (uint32_t)(max_threads * intensity / 100.0);
	device->profile_info.threads = (device->profile_info.threads / 2) * 2; // make it divisible by 2 to allow for parallel kernel execution
	if(max_threads > 0 && device->profile_info.threads == 0 && intensity > 0)
        device->profile_info.threads = 2;

    chunks = (double)device->profile_info.threads / (double)device->profile_info.threads_per_chunk;

	cuda_allocate(device, chunks, chunk_size);

	if(device->error != cudaSuccess)
		return false;

    return true;
}

bool cuda_hasher::buildThreadData() {
    __thread_data = new cuda_gpumgmt_thread_data[__enabledDevices.size() * 2];

    for(int i=0; i < __enabledDevices.size(); i++) {
        cuda_device_info *device = __enabledDevices[i];
        for(int threadId = 0; threadId < 2; threadId ++) {
            cuda_gpumgmt_thread_data &thread_data = __thread_data[i * 2 + threadId];
            thread_data.device = device;
            thread_data.thread_id = threadId;

            cudaStream_t stream;
            device->error = cudaStreamCreate(&stream);
            if(device->error != cudaSuccess) {
                LOG("Error running kernel: (" + to_string(device->error) + ") cannot create cuda stream.");
                return false;
            }

        	thread_data.device_data = stream;

            #ifdef PARALLEL_CUDA
                if(threadId == 0) {
                    thread_data.threads_idx = 0;
                    thread_data.threads = device->profile_info.threads / 2;
                }
                else {
                    thread_data.threads_idx = device->profile_info.threads / 2;
                    thread_data.threads = device->profile_info.threads - thread_data.threads_idx;
                }
            #else
                thread_data.threads_idx = 0;
                thread_data.threads = device->profile_info.threads;
            #endif

            thread_data.argon2 = new Argon2(cuda_kernel_prehasher, cuda_kernel_filler, cuda_kernel_posthasher,
                                            nullptr, &thread_data);
            thread_data.argon2->setThreads(thread_data.threads);
            thread_data.hashData.outSize = xmrig::ARGON2_HASHLEN + 4;
        }
    }

    return true;
}

int cuda_hasher::compute(int threadIdx, uint8_t *input, size_t size, uint8_t *output) {
    cuda_gpumgmt_thread_data &threadData = __thread_data[threadIdx];

	cudaSetDevice(threadData.device->cuda_index);

    threadData.hashData.input = input;
    threadData.hashData.inSize = size;
    threadData.hashData.output = output;
    int hashCount = threadData.argon2->generateHashes(*m_profile, threadData.hashData);
    if(threadData.device->error != cudaSuccess) {
        LOG("Error running kernel: (" + to_string(threadData.device->error) + ")" + threadData.device->error_message);
        return 0;
    }

    uint32_t *nonce = ((uint32_t *)(((uint8_t*)threadData.hashData.input) + 39));
    (*nonce) += threadData.threads;

    return hashCount;

}

size_t cuda_hasher::parallelism(int workerIdx) {
    cuda_gpumgmt_thread_data &threadData = __thread_data[workerIdx];
    return threadData.threads;
}

size_t cuda_hasher::deviceCount() {
    return __enabledDevices.size();
}

REGISTER_HASHER(cuda_hasher);

#endif //WITH_CUDA
