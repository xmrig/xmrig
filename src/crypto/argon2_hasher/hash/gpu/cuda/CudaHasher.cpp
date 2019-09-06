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

#include "CudaHasher.h"
#include "../../../common/DLLExport.h"

CudaHasher::CudaHasher() {
    m_type = "GPU";
    m_subType = "CUDA";
    m_shortSubType = "NVD";
    m_intensity = 0;
    m_description = "";
    m_computingThreads = 0;
}


CudaHasher::~CudaHasher() {
	this->cleanup();
}

bool CudaHasher::initialize(xmrig::Algo algorithm, xmrig::Variant variant) {
	cudaError_t error = cudaSuccess;
	string error_message;

    m_profile = getArgon2Profile(algorithm, variant);

    m_devices = queryCudaDevices(error, error_message);

	if(error != cudaSuccess) {
		m_description = "No compatible GPU detected: " + error_message;
		return false;
	}

	if (m_devices.empty()) {
		m_description = "No compatible GPU detected.";
		return false;
	}

	return true;
}

vector<CudaDeviceInfo *> CudaHasher::queryCudaDevices(cudaError_t &error, string &error_message) {
    cudaSetDeviceFlags(cudaDeviceBlockingSync);

	vector<CudaDeviceInfo *> devices;
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
		CudaDeviceInfo *dev = getDeviceInfo(i);
		if(dev == NULL)
			continue;
		if(dev->error != cudaSuccess) {
			error = dev->error;
			error_message = dev->errorMessage;
			continue;
		}
		devices.push_back(dev);
	}
	return devices;
}

CudaDeviceInfo *CudaHasher::getDeviceInfo(int device_index) {
	CudaDeviceInfo *device_info = new CudaDeviceInfo();
	device_info->error = cudaSuccess;
	device_info->cudaIndex = device_index;

	device_info->error = cudaSetDevice(device_index);
	if(device_info->error != cudaSuccess) {
		device_info->errorMessage = "Error setting current device.";
		return device_info;
	}

    cudaDeviceProp devProp;
	device_info->error = cudaGetDeviceProperties(&devProp, device_index);
	if(device_info->error != cudaSuccess) {
		device_info->errorMessage = "Error setting current device.";
		return device_info;
	}

    device_info->deviceString = devProp.name;

    size_t freemem, totalmem;
    device_info->error = cudaMemGetInfo(&freemem, &totalmem);
	if(device_info->error != cudaSuccess) {
		device_info->errorMessage = "Error setting current device.";
		return device_info;
	}

    device_info->freeMemSize = freemem;
    device_info->maxAllocableMemSize = freemem / 4;

    double mem_in_gb = totalmem / 1073741824.0;
    stringstream ss;
    ss << setprecision(2) << mem_in_gb;
    device_info->deviceString += (" (" + ss.str() + "GB)");

    return device_info;
}

bool CudaHasher::configure(xmrig::HasherConfig &config) {
    int deviceOffset = config.getGPUCardsCount();
    int index = deviceOffset;
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

	for(vector<CudaDeviceInfo *>::iterator d = m_devices.begin(); d != m_devices.end(); d++, index++) {
		stringstream ss;
		ss << "["<< (index + 1) << "] " << (*d)->deviceString;
		string device_description = ss.str();
		(*d)->deviceIndex = index;
        (*d)->profileInfo.profile = m_profile;

        if(config.gpuFilter().size() > 0) {
			bool found = false;
            for(xmrig::GPUFilter fit : config.gpuFilter()) {
                if(device_description.find(fit.filter) != string::npos) {
                    found = true;
                    break;
                }
            }
			if(!found) {
				(*d)->profileInfo.threads = 0;
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

        double device_intensity = config.getGPUIntensity(deviceOffset + m_enabledDevices.size());

		m_description += ss.str();

		if(!(setupDeviceInfo((*d), device_intensity))) {
			m_description += (*d)->errorMessage;
			m_description += "\n";
			continue;
		};

		DeviceInfo device;

		char bus_id[100];
		if(cudaDeviceGetPCIBusId(bus_id, 100, (*d)->cudaIndex) == cudaSuccess) {
			device.bus_id = bus_id;
			int domain_separator = device.bus_id.find(":");
			if(domain_separator != string::npos) {
				device.bus_id.erase(0, domain_separator + 1);
			}
		}

		device.name = (*d)->deviceString;
		device.intensity = device_intensity;
        storeDeviceInfo((*d)->deviceIndex, device);

        m_enabledDevices.push_back(*d);

		total_threads += (*d)->profileInfo.threads;
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

    m_intensity = intensity / m_enabledDevices.size();
    m_computingThreads = m_enabledDevices.size() * 2; // 2 computing threads for each device
    m_description += "Status: ENABLED - with " + to_string(total_threads) + " threads.";

	return true;
}

void CudaHasher::cleanup() {
	for(vector<CudaDeviceInfo *>::iterator d = m_devices.begin(); d != m_devices.end(); d++) {
		cuda_free(*d);
	}
}

bool CudaHasher::setupDeviceInfo(CudaDeviceInfo *device, double intensity) {
    device->profileInfo.threads_per_chunk = (uint32_t)(device->maxAllocableMemSize / device->profileInfo.profile->memSize);
    size_t chunk_size = device->profileInfo.threads_per_chunk * device->profileInfo.profile->memSize;

    if(chunk_size == 0) {
        device->error = cudaErrorInitializationError;
        device->errorMessage = "Not enough memory on GPU.";
        return false;
    }

    uint64_t usable_memory = device->freeMemSize;
    double chunks = (double)usable_memory / (double)chunk_size;

    uint32_t max_threads = (uint32_t)(device->profileInfo.threads_per_chunk * chunks);

    if(max_threads == 0) {
        device->error = cudaErrorInitializationError;
        device->errorMessage = "Not enough memory on GPU.";
        return false;
    }

    device->profileInfo.threads = (uint32_t)(max_threads * intensity / 100.0);
	device->profileInfo.threads = (device->profileInfo.threads / 2) * 2; // make it divisible by 2 to allow for parallel kernel execution
	if(max_threads > 0 && device->profileInfo.threads == 0 && intensity > 0)
        device->profileInfo.threads = 2;

    chunks = (double)device->profileInfo.threads / (double)device->profileInfo.threads_per_chunk;

	cuda_allocate(device, chunks, chunk_size);

	if(device->error != cudaSuccess)
		return false;

    return true;
}

bool CudaHasher::buildThreadData() {
    m_threadData = new CudaGpuMgmtThreadData[m_enabledDevices.size() * 2];

    for(int i=0; i < m_enabledDevices.size(); i++) {
        CudaDeviceInfo *device = m_enabledDevices[i];
        for(int threadId = 0; threadId < 2; threadId ++) {
            CudaGpuMgmtThreadData &thread_data = m_threadData[i * 2 + threadId];
            thread_data.device = device;
            thread_data.threadId = threadId;

            cudaStream_t stream;
            cudaSetDevice(device->cudaIndex);		
            device->error = cudaStreamCreate(&stream);
            if(device->error != cudaSuccess) {
                LOG("Error running kernel: (" + to_string(device->error) + ") cannot create cuda stream.");
                return false;
            }

        	thread_data.deviceData = stream;

            #ifdef PARALLEL_CUDA
                if(threadId == 0) {
                    thread_data.threadsIdx = 0;
                    thread_data.threads = device->profileInfo.threads / 2;
                }
                else {
                    thread_data.threadsIdx = device->profileInfo.threads / 2;
                    thread_data.threads = device->profileInfo.threads - thread_data.threadsIdx;
                }
            #else
                thread_data.threadsIdx = 0;
                thread_data.threads = device->profileInfo.threads;
            #endif

            thread_data.argon2 = new Argon2(cuda_kernel_prehasher, cuda_kernel_filler, cuda_kernel_posthasher,
                                            nullptr, &thread_data);
            thread_data.argon2->setThreads(thread_data.threads);
            thread_data.hashData.outSize = xmrig::ARGON2_HASHLEN + 4;
        }
    }

    return true;
}

int CudaHasher::compute(int threadIdx, uint8_t *input, size_t size, uint8_t *output) {
    CudaGpuMgmtThreadData &threadData = m_threadData[threadIdx];

	cudaSetDevice(threadData.device->cudaIndex);

    threadData.hashData.input = input;
    threadData.hashData.inSize = size;
    threadData.hashData.output = output;
    int hashCount = threadData.argon2->generateHashes(*m_profile, threadData.hashData);
    if(threadData.device->error != cudaSuccess) {
        LOG("Error running kernel: (" + to_string(threadData.device->error) + ")" + threadData.device->errorMessage);
        return 0;
    }

    uint32_t *nonce = ((uint32_t *)(((uint8_t*)threadData.hashData.input) + 39));
    (*nonce) += threadData.threads;

    return hashCount;

}

size_t CudaHasher::parallelism(int workerIdx) {
    CudaGpuMgmtThreadData &threadData = m_threadData[workerIdx];
    return threadData.threads;
}

size_t CudaHasher::deviceCount() {
    return m_enabledDevices.size();
}

DeviceInfo &CudaHasher::device(int workerIdx) {
    workerIdx /= 2;

    if(workerIdx < 0 || workerIdx > m_enabledDevices.size())
        return devices().begin()->second;

    return devices()[m_enabledDevices[workerIdx]->deviceIndex];
}

REGISTER_HASHER(CudaHasher);

#endif //WITH_CUDA
