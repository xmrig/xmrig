//
// Created by Haifa Bogdan Adnan on 03/08/2018.
//

#include <crypto/Argon2_constants.h>
#include "../../../common/common.h"

#include "crypto/argon2_hasher/hash/Hasher.h"
#include "crypto/argon2_hasher/hash/argon2/Argon2.h"

#include "OpenCLHasher.h"
#include "OpenCLKernel.h"

#include "crypto/argon2_hasher/common/DLLExport.h"

#if defined(WITH_OPENCL)

#ifndef CL_DEVICE_BOARD_NAME_AMD
#define CL_DEVICE_BOARD_NAME_AMD                    0x4038
#endif
#ifndef CL_DEVICE_TOPOLOGY_AMD
#define CL_DEVICE_TOPOLOGY_AMD                      0x4037
#endif
#ifndef CL_DEVICE_PCI_BUS_ID_NV
#define CL_DEVICE_PCI_BUS_ID_NV                     0x4008
#endif
#ifndef CL_DEVICE_PCI_SLOT_ID_NV
#define CL_DEVICE_PCI_SLOT_ID_NV                    0x4009
#endif

typedef union
{
    struct { cl_uint type; cl_uint data[5]; } raw;
    struct { cl_uint type; cl_char unused[17]; cl_char bus; cl_char device; cl_char function; } pcie;
} device_topology_amd;

#define KERNEL_WORKGROUP_SIZE   32

OpenCLHasher::OpenCLHasher() {
    m_type = "GPU";
    m_subType = "OPENCL";
    m_shortSubType = "OCL";
    m_intensity = 0;
    m_description = "";
    m_computingThreads = 0;
}

OpenCLHasher::~OpenCLHasher() {
//    this->cleanup();
}

bool OpenCLHasher::initialize(xmrig::Algo algorithm, xmrig::Variant variant) {
    cl_int error = CL_SUCCESS;
    string error_message;

    m_profile = getArgon2Profile(algorithm, variant);

    m_devices = queryOpenCLDevices(error, error_message);
    if(error != CL_SUCCESS) {
        m_description = "No compatible GPU detected: " + error_message;
        return false;
    }

    if (m_devices.empty()) {
        m_description = "No compatible GPU detected.";
        return false;
    }

    return true;
}

vector<OpenCLDeviceInfo*> OpenCLHasher::queryOpenCLDevices(cl_int &error, string &error_message) {
    cl_int err;

    cl_uint platform_count = 0;
    cl_uint device_count = 0;

    vector<OpenCLDeviceInfo*> result;

    clGetPlatformIDs(0, NULL, &platform_count);
    if(platform_count == 0) {
        return result;
    }

    cl_platform_id *platforms = (cl_platform_id*)malloc(platform_count * sizeof(cl_platform_id));

    err=clGetPlatformIDs(platform_count, platforms, &platform_count);
    if(err != CL_SUCCESS)  {
        free(platforms);
        error = err;
        error_message = "Error querying for opencl platforms.";
        return result;
    }

    int counter = 0;

    for(uint32_t i=0; i < platform_count; i++) {
        device_count = 0;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &device_count);
        if(device_count == 0) {
            continue;
        }

        cl_device_id * devices = (cl_device_id*)malloc(device_count * sizeof(cl_device_id));
        err=clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, device_count, devices, &device_count);

        if(err != CL_SUCCESS)  {
            free(devices);
            error = err;
            error_message = "Error querying for opencl devices.";
            continue;
        }

        for(uint32_t j=0; j < device_count; j++) {
            OpenCLDeviceInfo *info = getDeviceInfo(platforms[i], devices[j]);
            if(info->error != CL_SUCCESS) {
                error = info->error;
                error_message = info->errorMessage;
            }
            else {
                info->deviceIndex = counter;
                result.push_back(info);
                counter++;
            }
        }

        free(devices);
    }

    free(platforms);

    return result;
}

OpenCLDeviceInfo *OpenCLHasher::getDeviceInfo(cl_platform_id platform, cl_device_id device) {
    OpenCLDeviceInfo *device_info = new OpenCLDeviceInfo(CL_SUCCESS, "");

    device_info->platform = platform;
    device_info->device = device;

    char *buffer;
    size_t sz;

    // device name
    string device_vendor;
    sz = 0;
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, 0, NULL, &sz);
    buffer = (char *)malloc(sz + 1);
    device_info->error = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sz, buffer, &sz);
    if(device_info->error != CL_SUCCESS) {
        free(buffer);
        device_info->errorMessage = "Error querying device vendor.";
        return device_info;
    }
    else {
        buffer[sz] = 0;
        device_vendor = buffer;
        free(buffer);
    }

    string device_name;
    cl_device_info query_type = CL_DEVICE_NAME;

    if(device_vendor.find("Advanced Micro Devices") != string::npos)
        query_type = CL_DEVICE_BOARD_NAME_AMD;

    sz = 0;
    clGetDeviceInfo(device, query_type, 0, NULL, &sz);
    buffer = (char *) malloc(sz + 1);
    device_info->error = clGetDeviceInfo(device, query_type, sz, buffer, &sz);
    if (device_info->error != CL_SUCCESS) {
        free(buffer);
        device_info->errorMessage = "Error querying device name.";
        return device_info;
    } else {
        buffer[sz] = 0;
        device_name = buffer;
        free(buffer);
    }

    string device_version;
    sz = 0;
    clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, NULL, &sz);
    buffer = (char *)malloc(sz + 1);
    device_info->error = clGetDeviceInfo(device, CL_DEVICE_VERSION, sz, buffer, &sz);
    if(device_info->error != CL_SUCCESS) {
        free(buffer);
        device_info->errorMessage = "Error querying device version.";
        return device_info;
    }
    else {
        buffer[sz] = 0;
        device_version = buffer;
        free(buffer);
    }

    device_info->deviceString = device_vendor + " - " + device_name/* + " : " + device_version*/;

    string extensions;
    sz = 0;
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, NULL, &sz);
    buffer = (char *)malloc(sz + 1);
    device_info->error = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sz, buffer, &sz);
    if(device_info->error != CL_SUCCESS) {
        free(buffer);
        device_info->errorMessage = "Error querying device extensions.";
        return device_info;
    }
    else {
        buffer[sz] = 0;
        extensions = buffer;
        free(buffer);
    }

    device_info->deviceExtensions = extensions;

    device_info->error = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(device_info->maxMemSize), &(device_info->maxMemSize), NULL);
    if(device_info->error != CL_SUCCESS) {
        device_info->errorMessage = "Error querying device global memory size.";
        return device_info;
    }

    device_info->error = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(device_info->maxAllocableMemSize), &(device_info->maxAllocableMemSize), NULL);
    if(device_info->error != CL_SUCCESS) {
        device_info->errorMessage = "Error querying device max memory allocation.";
        return device_info;
    }

    double mem_in_gb = device_info->maxMemSize / 1073741824.0;
    stringstream ss;
    ss << setprecision(2) << mem_in_gb;
    device_info->deviceString += (" (" + ss.str() + "GB)");

    return device_info;
}

bool OpenCLHasher::configure(xmrig::HasherConfig &config) {
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

    for(vector<OpenCLDeviceInfo *>::iterator d = m_devices.begin(); d != m_devices.end(); d++, index++) {
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

        if((*d)->deviceString.find("Advanced Micro Devices") != string::npos) {
            device_topology_amd amdtopo;
            if(clGetDeviceInfo((*d)->device, CL_DEVICE_TOPOLOGY_AMD, sizeof(amdtopo), &amdtopo, NULL) == CL_SUCCESS) {
                char bus_id[50];
                sprintf(bus_id, "%02x:%02x.%x", amdtopo.pcie.bus, amdtopo.pcie.device, amdtopo.pcie.function);
                device.bus_id = bus_id;
            }
        }
        else if((*d)->deviceString.find("NVIDIA") != string::npos) {
            cl_uint bus;
            cl_uint slot;

            if(clGetDeviceInfo ((*d)->device, CL_DEVICE_PCI_BUS_ID_NV, sizeof(bus), &bus, NULL) == CL_SUCCESS) {
                if(clGetDeviceInfo ((*d)->device, CL_DEVICE_PCI_SLOT_ID_NV, sizeof(slot), &slot, NULL) == CL_SUCCESS) {
                    char bus_id[50];
                    sprintf(bus_id, "%02x:%02x.0", bus, slot);
                    device.bus_id = bus_id;
                }
            }
        }

        device.name = (*d)->deviceString;
        device.intensity = device_intensity;
        storeDeviceInfo((*d)->deviceIndex, device);

        m_enabledDevices.push_back(*d);

        total_threads += (*d)->profileInfo.threads;
        intensity += device_intensity;
    }

    config.addGPUCardsCount(index - deviceOffset);

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

    buildThreadData();

    m_intensity = intensity / m_enabledDevices.size();
    m_computingThreads = m_enabledDevices.size() * 2; // 2 computing threads for each device
    m_description += "Status: ENABLED - with " + to_string(total_threads) + " threads.";

    return true;
}

bool OpenCLHasher::setupDeviceInfo(OpenCLDeviceInfo *device, double intensity) {
    cl_int error;

    cl_context_properties properties[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties) device->platform,
            0};

    device->context = clCreateContext(properties, 1, &(device->device), NULL, NULL, &error);
    if (error != CL_SUCCESS) {
        device->error = error;
        device->errorMessage = "Error getting device context.";
        return false;
    }

    device->queue[0] = clCreateCommandQueue(device->context, device->device, NULL, &error);
    if (error != CL_SUCCESS) {
        device->error = error;
        device->errorMessage = "Error getting device command queue.";
        return false;
    }

    device->queue[1] = clCreateCommandQueue(device->context, device->device, NULL, &error);
    if (error != CL_SUCCESS) {
        device->error = error;
        device->errorMessage = "Error getting device command queue.";
        return false;
    }

    const char *srcptr[] = {OpenCLKernel.c_str()};
    size_t srcsize = OpenCLKernel.size();

    device->program = clCreateProgramWithSource(device->context, 1, srcptr, &srcsize, &error);
    if (error != CL_SUCCESS) {
        device->error = error;
        device->errorMessage = "Error creating opencl program for device.";
        return false;
    }

    string options = "";
    if(device->deviceExtensions.find("cl_amd_media_ops") != string::npos)
        options += "-D USE_AMD_BITALIGN";
    error = clBuildProgram(device->program, 1, &device->device, options.c_str(), NULL, NULL);
    if (error != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(device->program, device->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *) malloc(log_size + 1);
        clGetProgramBuildInfo(device->program, device->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size] = 0;
        string build_log = log;
        free(log);

        device->error = error;
        device->errorMessage = "Error building opencl program for device: " + build_log;
        return false;
    }

    device->kernelPrehash[0] = clCreateKernel(device->program, "prehash", &error);
    if (error != CL_SUCCESS) {
        device->error = error;
        device->errorMessage = "Error creating opencl prehash kernel for device.";
        return false;
    }
    device->kernelPrehash[1] = clCreateKernel(device->program, "prehash", &error);
    if (error != CL_SUCCESS) {
        device->error = error;
        device->errorMessage = "Error creating opencl prehash kernel for device.";
        return false;
    }
    device->kernelFillBlocks[0] = clCreateKernel(device->program, "fill_blocks", &error);
    if (error != CL_SUCCESS) {
        device->error = error;
        device->errorMessage = "Error creating opencl main kernel for device.";
        return false;
    }
    device->kernelFillBlocks[1] = clCreateKernel(device->program, "fill_blocks", &error);
    if (error != CL_SUCCESS) {
        device->error = error;
        device->errorMessage = "Error creating opencl main kernel for device.";
        return false;
    }
    device->kernelPosthash[0] = clCreateKernel(device->program, "posthash", &error);
    if (error != CL_SUCCESS) {
        device->error = error;
        device->errorMessage = "Error creating opencl posthash kernel for device.";
        return false;
    }
    device->kernelPosthash[1] = clCreateKernel(device->program, "posthash", &error);
    if (error != CL_SUCCESS) {
        device->error = error;
        device->errorMessage = "Error creating opencl posthash kernel for device.";
        return false;
    }

    device->profileInfo.threads_per_chunk = ((uint32_t) (device->maxAllocableMemSize / device->profileInfo.profile->memSize) / 2) * 2; // make it divisible by 2 to allow 2 hashes per wavefront
    size_t chunk_size = device->profileInfo.threads_per_chunk * device->profileInfo.profile->memSize;

    if (chunk_size == 0) {
        device->error = -1;
        device->errorMessage = "Not enough memory on GPU.";
        return false;
    }

    uint64_t usable_memory = device->maxMemSize;
    double chunks = (double) usable_memory / (double) chunk_size;

    uint32_t max_threads = (uint32_t) (device->profileInfo.threads_per_chunk * chunks);

    if (max_threads == 0) {
        device->error = -1;
        device->errorMessage = "Not enough memory on GPU.";
        return false;
    }

    device->profileInfo.threads = (uint32_t) (max_threads * intensity / 100.0);
    device->profileInfo.threads = (device->profileInfo.threads / 4) * 4; // make it divisible by 4
    if (max_threads > 0 && device->profileInfo.threads == 0 && intensity > 0)
        device->profileInfo.threads = 4;

    double counter = (double) device->profileInfo.threads / (double) device->profileInfo.threads_per_chunk;
    size_t allocated_mem_for_current_chunk = 0;

    if (counter > 0) {
        if (counter > 1) {
            allocated_mem_for_current_chunk = chunk_size;
        } else {
            allocated_mem_for_current_chunk = (size_t) ceil(chunk_size * counter);
        }
        counter -= 1;
    } else {
        allocated_mem_for_current_chunk = 1;
    }
    device->arguments.memoryChunk_0 = clCreateBuffer(device->context, CL_MEM_READ_WRITE,
                                                     allocated_mem_for_current_chunk, NULL, &error);
    if (error != CL_SUCCESS) {
        device->error = error;
        device->errorMessage = "Error creating memory buffer.";
        return false;
    }

    if (counter > 0) {
        if (counter > 1) {
            allocated_mem_for_current_chunk = chunk_size;
        } else {
            allocated_mem_for_current_chunk = (size_t) ceil(chunk_size * counter);
        }
        counter -= 1;
    } else {
        allocated_mem_for_current_chunk = 1;
    }
    device->arguments.memoryChunk_1 = clCreateBuffer(device->context, CL_MEM_READ_WRITE,
                                                     allocated_mem_for_current_chunk, NULL, &error);
    if (error != CL_SUCCESS) {
        device->error = error;
        device->errorMessage = "Error creating memory buffer.";
        return false;
    }

    if (counter > 0) {
        if (counter > 1) {
            allocated_mem_for_current_chunk = chunk_size;
        } else {
            allocated_mem_for_current_chunk = (size_t) ceil(chunk_size * counter);
        }
        counter -= 1;
    } else {
        allocated_mem_for_current_chunk = 1;
    }
    device->arguments.memoryChunk_2 = clCreateBuffer(device->context, CL_MEM_READ_WRITE,
                                                     allocated_mem_for_current_chunk, NULL, &error);
    if (error != CL_SUCCESS) {
        device->error = error;
        device->errorMessage = "Error creating memory buffer.";
        return false;
    }

    if (counter > 0) {
        if (counter > 1) {
            allocated_mem_for_current_chunk = chunk_size;
        } else {
            allocated_mem_for_current_chunk = (size_t) ceil(chunk_size * counter);
        }
        counter -= 1;
    } else {
        allocated_mem_for_current_chunk = 1;
    }
    device->arguments.memoryChunk_3 = clCreateBuffer(device->context, CL_MEM_READ_WRITE,
                                                     allocated_mem_for_current_chunk, NULL, &error);
    if (error != CL_SUCCESS) {
        device->error = error;
        device->errorMessage = "Error creating memory buffer.";
        return false;
    }

    if (counter > 0) {
        if (counter > 1) {
            allocated_mem_for_current_chunk = chunk_size;
        } else {
            allocated_mem_for_current_chunk = (size_t) ceil(chunk_size * counter);
        }
        counter -= 1;
    } else {
        allocated_mem_for_current_chunk = 1;
    }
    device->arguments.memoryChunk_4 = clCreateBuffer(device->context, CL_MEM_READ_WRITE,
                                                     allocated_mem_for_current_chunk, NULL, &error);
    if (error != CL_SUCCESS) {
        device->error = error;
        device->errorMessage = "Error creating memory buffer.";
        return false;
    }

    if (counter > 0) {
        if (counter > 1) {
            allocated_mem_for_current_chunk = chunk_size;
        } else {
            allocated_mem_for_current_chunk = (size_t) ceil(chunk_size * counter);
        }
        counter -= 1;
    } else {
        allocated_mem_for_current_chunk = 1;
    }
    device->arguments.memoryChunk_5 = clCreateBuffer(device->context, CL_MEM_READ_WRITE,
                                                     allocated_mem_for_current_chunk, NULL, &error);
    if (error != CL_SUCCESS) {
        device->error = error;
        device->errorMessage = "Error creating memory buffer.";
        return false;
    }

    device->arguments.refs = clCreateBuffer(device->context, CL_MEM_READ_ONLY,
                                            device->profileInfo.profile->blockRefsSize * sizeof(uint32_t), NULL,
                                            &error);
    if (error != CL_SUCCESS) {
        device->error = error;
        device->errorMessage = "Error creating memory buffer.";
        return false;
    }

    if (device->profileInfo.profile->succesiveIdxs == 1) {
        device->arguments.idxs = NULL;
    }
    else {
        device->arguments.idxs = clCreateBuffer(device->context, CL_MEM_READ_ONLY,
                                                device->profileInfo.profile->blockRefsSize * sizeof(uint32_t), NULL,
                                                &error);
        if (error != CL_SUCCESS) {
            device->error = error;
            device->errorMessage = "Error creating memory buffer.";
            return false;
        }
    }

    device->arguments.segments = clCreateBuffer(device->context, CL_MEM_READ_ONLY, device->profileInfo.profile->segCount * 3 * sizeof(uint32_t), NULL, &error);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->errorMessage = "Error creating memory buffer.";
        return false;
    }

    size_t preseed_memory_size = device->profileInfo.profile->pwdLen * 4;
    size_t seed_memory_size = device->profileInfo.threads * (device->profileInfo.profile->thrCost * 2) * ARGON2_BLOCK_SIZE;
    size_t out_memory_size = device->profileInfo.threads * ARGON2_BLOCK_SIZE;
    size_t hash_memory_size = device->profileInfo.threads * (xmrig::ARGON2_HASHLEN + 4);

    device->arguments.preseedMemory[0] = clCreateBuffer(device->context, CL_MEM_READ_ONLY, preseed_memory_size, NULL, &error);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->errorMessage = "Error creating memory buffer.";
        return false;
    }

    device->arguments.preseedMemory[1] = clCreateBuffer(device->context, CL_MEM_READ_ONLY, preseed_memory_size, NULL, &error);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->errorMessage = "Error creating memory buffer.";
        return false;
    }

    device->arguments.seedMemory[0] = clCreateBuffer(device->context, CL_MEM_READ_WRITE, seed_memory_size, NULL, &error);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->errorMessage = "Error creating memory buffer.";
        return false;
    }

    device->arguments.seedMemory[1] = clCreateBuffer(device->context, CL_MEM_READ_WRITE, seed_memory_size, NULL, &error);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->errorMessage = "Error creating memory buffer.";
        return false;
    }

    device->arguments.outMemory[0] = clCreateBuffer(device->context, CL_MEM_READ_WRITE, out_memory_size, NULL, &error);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->errorMessage = "Error creating memory buffer.";
        return false;
    }

    device->arguments.outMemory[1] = clCreateBuffer(device->context, CL_MEM_READ_WRITE, out_memory_size, NULL, &error);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->errorMessage = "Error creating memory buffer.";
        return false;
    }

    device->arguments.hashMemory[0] = clCreateBuffer(device->context, CL_MEM_WRITE_ONLY, hash_memory_size, NULL, &error);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->errorMessage = "Error creating memory buffer.";
        return false;
    }

    device->arguments.hashMemory[1] = clCreateBuffer(device->context, CL_MEM_WRITE_ONLY, hash_memory_size, NULL, &error);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->errorMessage = "Error creating memory buffer.";
        return false;
    }

	//optimise address sizes
    uint32_t *refs = (uint32_t *)malloc(device->profileInfo.profile->blockRefsSize * sizeof(uint32_t));
    for(int i=0;i<device->profileInfo.profile->blockRefsSize; i++) {
        refs[i] = device->profileInfo.profile->blockRefs[i * 3 + 1];
    }

    error=clEnqueueWriteBuffer(device->queue[0], device->arguments.refs, CL_TRUE, 0, device->profileInfo.profile->blockRefsSize * sizeof(uint32_t), refs, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->errorMessage = "Error writing to gpu memory.";
        return false;
    }

    free(refs);

    if(device->profileInfo.profile->succesiveIdxs == 0) {
        uint32_t *idxs = (uint32_t *) malloc(device->profileInfo.profile->blockRefsSize * sizeof(uint32_t));
        for (int i = 0; i < device->profileInfo.profile->blockRefsSize; i++) {
            idxs[i] = device->profileInfo.profile->blockRefs[i * 3];
            if (device->profileInfo.profile->blockRefs[i * 3 + 2] == 1) {
                idxs[i] |= 0x80000000;
            }
        }

        error=clEnqueueWriteBuffer(device->queue[0], device->arguments.idxs, CL_TRUE, 0, device->profileInfo.profile->blockRefsSize * sizeof(uint32_t), idxs, 0, NULL, NULL);
        if(error != CL_SUCCESS) {
            device->error = error;
            device->errorMessage = "Error writing to gpu memory.";
            return false;
        }

        free(idxs);
    }

    error=clEnqueueWriteBuffer(device->queue[0], device->arguments.segments, CL_TRUE, 0, device->profileInfo.profile->segCount * 3 * sizeof(uint32_t), device->profileInfo.profile->segments, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->errorMessage = "Error writing to gpu memory.";
        return false;
    }

    int passes = device->profileInfo.profile->segCount / (4 * device->profileInfo.profile->thrCost);

    clSetKernelArg(device->kernelFillBlocks[0], 0, sizeof(device->arguments.memoryChunk_0), &device->arguments.memoryChunk_0);
	clSetKernelArg(device->kernelFillBlocks[0], 1, sizeof(device->arguments.memoryChunk_1), &device->arguments.memoryChunk_1);
	clSetKernelArg(device->kernelFillBlocks[0], 2, sizeof(device->arguments.memoryChunk_2), &device->arguments.memoryChunk_2);
	clSetKernelArg(device->kernelFillBlocks[0], 3, sizeof(device->arguments.memoryChunk_3), &device->arguments.memoryChunk_3);
	clSetKernelArg(device->kernelFillBlocks[0], 4, sizeof(device->arguments.memoryChunk_4), &device->arguments.memoryChunk_4);
	clSetKernelArg(device->kernelFillBlocks[0], 5, sizeof(device->arguments.memoryChunk_5), &device->arguments.memoryChunk_5);
    clSetKernelArg(device->kernelFillBlocks[0], 8, sizeof(device->arguments.refs), &device->arguments.refs);
    if(device->profileInfo.profile->succesiveIdxs == 0)
        clSetKernelArg(device->kernelFillBlocks[0], 9, sizeof(device->arguments.idxs), &device->arguments.idxs);
    else
        clSetKernelArg(device->kernelFillBlocks[0], 9, sizeof(cl_mem), NULL);
	clSetKernelArg(device->kernelFillBlocks[0], 10, sizeof(device->arguments.segments), &device->arguments.segments);
    clSetKernelArg(device->kernelFillBlocks[0], 11, sizeof(int32_t), &device->profileInfo.profile->memSize);
    clSetKernelArg(device->kernelFillBlocks[0], 12, sizeof(int32_t), &device->profileInfo.profile->thrCost);
    clSetKernelArg(device->kernelFillBlocks[0], 13, sizeof(int32_t), &device->profileInfo.profile->segSize);
    clSetKernelArg(device->kernelFillBlocks[0], 14, sizeof(int32_t), &device->profileInfo.profile->segCount);
    clSetKernelArg(device->kernelFillBlocks[0], 15, sizeof(int32_t), &device->profileInfo.threads_per_chunk);

    clSetKernelArg(device->kernelPrehash[0], 2, sizeof(int32_t), &device->profileInfo.profile->memCost);
    clSetKernelArg(device->kernelPrehash[0], 3, sizeof(int32_t), &device->profileInfo.profile->thrCost);
    clSetKernelArg(device->kernelPrehash[0], 4, sizeof(int32_t), &passes);
    clSetKernelArg(device->kernelPrehash[0], 6, sizeof(int32_t), &device->profileInfo.profile->saltLen);

    clSetKernelArg(device->kernelFillBlocks[1], 0, sizeof(device->arguments.memoryChunk_0), &device->arguments.memoryChunk_0);
    clSetKernelArg(device->kernelFillBlocks[1], 1, sizeof(device->arguments.memoryChunk_1), &device->arguments.memoryChunk_1);
    clSetKernelArg(device->kernelFillBlocks[1], 2, sizeof(device->arguments.memoryChunk_2), &device->arguments.memoryChunk_2);
    clSetKernelArg(device->kernelFillBlocks[1], 3, sizeof(device->arguments.memoryChunk_3), &device->arguments.memoryChunk_3);
    clSetKernelArg(device->kernelFillBlocks[1], 4, sizeof(device->arguments.memoryChunk_4), &device->arguments.memoryChunk_4);
    clSetKernelArg(device->kernelFillBlocks[1], 5, sizeof(device->arguments.memoryChunk_5), &device->arguments.memoryChunk_5);
    clSetKernelArg(device->kernelFillBlocks[1], 8, sizeof(device->arguments.refs), &device->arguments.refs);
    if(device->profileInfo.profile->succesiveIdxs == 0)
        clSetKernelArg(device->kernelFillBlocks[1], 9, sizeof(device->arguments.idxs), &device->arguments.idxs);
    else
        clSetKernelArg(device->kernelFillBlocks[1], 9, sizeof(cl_mem), NULL);
    clSetKernelArg(device->kernelFillBlocks[1], 10, sizeof(device->arguments.segments), &device->arguments.segments);
    clSetKernelArg(device->kernelFillBlocks[1], 11, sizeof(int32_t), &device->profileInfo.profile->memSize);
    clSetKernelArg(device->kernelFillBlocks[1], 12, sizeof(int32_t), &device->profileInfo.profile->thrCost);
    clSetKernelArg(device->kernelFillBlocks[1], 13, sizeof(int32_t), &device->profileInfo.profile->segSize);
    clSetKernelArg(device->kernelFillBlocks[1], 14, sizeof(int32_t), &device->profileInfo.profile->segCount);
    clSetKernelArg(device->kernelFillBlocks[1], 15, sizeof(int32_t), &device->profileInfo.threads_per_chunk);

    clSetKernelArg(device->kernelPrehash[1], 2, sizeof(int32_t), &device->profileInfo.profile->memCost);
    clSetKernelArg(device->kernelPrehash[1], 3, sizeof(int32_t), &device->profileInfo.profile->thrCost);
    clSetKernelArg(device->kernelPrehash[1], 4, sizeof(int32_t), &passes);
    clSetKernelArg(device->kernelPrehash[1], 6, sizeof(int32_t), &device->profileInfo.profile->saltLen);
    return true;
}

bool opencl_kernel_prehasher(void *memory, int threads, Argon2Profile *profile, void *user_data) {
    OpenCLGpuMgmtThreadData *gpumgmt_thread = (OpenCLGpuMgmtThreadData *)user_data;
    OpenCLDeviceInfo *device = gpumgmt_thread->device;

    cl_int error;

    int sessions = max(profile->thrCost * 2, (uint32_t)16);
    double hashes_per_block = sessions / (profile->thrCost * 2.0);

    size_t total_work_items = sessions * 4 * ceil(threads / hashes_per_block);
    size_t local_work_items = sessions * 4;

    gpumgmt_thread->lock();

    error = clEnqueueWriteBuffer(device->queue[gpumgmt_thread->threadId], device->arguments.preseedMemory[gpumgmt_thread->threadId],
                                 CL_FALSE, 0, gpumgmt_thread->hashData.inSize, memory, 0, NULL, NULL);
    if (error != CL_SUCCESS) {
        device->error = error;
        device->errorMessage = "Error writing to gpu memory.";
        gpumgmt_thread->unlock();
        return false;
    }

    int inSizeInInt = gpumgmt_thread->hashData.inSize / 4;
    clSetKernelArg(device->kernelPrehash[gpumgmt_thread->threadId], 0, sizeof(device->arguments.preseedMemory[gpumgmt_thread->threadId]), &device->arguments.preseedMemory[gpumgmt_thread->threadId]);
    clSetKernelArg(device->kernelPrehash[gpumgmt_thread->threadId], 1, sizeof(device->arguments.seedMemory[gpumgmt_thread->threadId]), &device->arguments.seedMemory[gpumgmt_thread->threadId]);
    clSetKernelArg(device->kernelPrehash[gpumgmt_thread->threadId], 5, sizeof(int), &inSizeInInt);
    clSetKernelArg(device->kernelPrehash[gpumgmt_thread->threadId], 7, sizeof(int), &threads);
    clSetKernelArg(device->kernelPrehash[gpumgmt_thread->threadId], 8, sessions * sizeof(cl_ulong) * 76, NULL); // (preseed size is 16 ulongs = 128 bytes)

    error=clEnqueueNDRangeKernel(device->queue[gpumgmt_thread->threadId], device->kernelPrehash[gpumgmt_thread->threadId], 1, NULL, &total_work_items, &local_work_items, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->errorMessage = "Error running the kernel.";
        gpumgmt_thread->unlock();
        return false;
    }

    return true;
}

void *opencl_kernel_filler(int threads, Argon2Profile *profile, void *user_data) {
	OpenCLGpuMgmtThreadData *gpumgmt_thread = (OpenCLGpuMgmtThreadData *)user_data;
    OpenCLDeviceInfo *device = gpumgmt_thread->device;

    cl_int error;

	size_t total_work_items = threads * KERNEL_WORKGROUP_SIZE * profile->thrCost;
	size_t local_work_items = 2 * KERNEL_WORKGROUP_SIZE * profile->thrCost;

    size_t shared_mem = 2 * profile->thrCost * ARGON2_QWORDS_IN_BLOCK;

    clSetKernelArg(device->kernelFillBlocks[gpumgmt_thread->threadId], 6, sizeof(device->arguments.seedMemory[gpumgmt_thread->threadId]), &device->arguments.seedMemory[gpumgmt_thread->threadId]);
    clSetKernelArg(device->kernelFillBlocks[gpumgmt_thread->threadId], 7, sizeof(device->arguments.outMemory[gpumgmt_thread->threadId]), &device->arguments.outMemory[gpumgmt_thread->threadId]);
    clSetKernelArg(device->kernelFillBlocks[gpumgmt_thread->threadId], 16, sizeof(int), &(gpumgmt_thread->threadsIdx));
    clSetKernelArg(device->kernelFillBlocks[gpumgmt_thread->threadId], 17, sizeof(cl_ulong) * shared_mem, NULL);

    error=clEnqueueNDRangeKernel(device->queue[gpumgmt_thread->threadId], device->kernelFillBlocks[gpumgmt_thread->threadId], 1, NULL, &total_work_items, &local_work_items, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->errorMessage = "Error running the kernel.";
        gpumgmt_thread->unlock();
        return NULL;
    }

	return (void *)1;
}

bool opencl_kernel_posthasher(void *memory, int threads, Argon2Profile *profile, void *user_data) {
    OpenCLGpuMgmtThreadData *gpumgmt_thread = (OpenCLGpuMgmtThreadData *)user_data;
    OpenCLDeviceInfo *device = gpumgmt_thread->device;

    cl_int error;

    size_t total_work_items = threads * 4;
    size_t local_work_items = 4;

    clSetKernelArg(device->kernelPosthash[gpumgmt_thread->threadId], 0, sizeof(device->arguments.hashMemory[gpumgmt_thread->threadId]), &device->arguments.hashMemory[gpumgmt_thread->threadId]);
    clSetKernelArg(device->kernelPosthash[gpumgmt_thread->threadId], 1, sizeof(device->arguments.outMemory[gpumgmt_thread->threadId]), &device->arguments.outMemory[gpumgmt_thread->threadId]);
    clSetKernelArg(device->kernelPosthash[gpumgmt_thread->threadId], 2, sizeof(device->arguments.preseedMemory[gpumgmt_thread->threadId]), &device->arguments.preseedMemory[gpumgmt_thread->threadId]);
    clSetKernelArg(device->kernelPosthash[gpumgmt_thread->threadId], 3, sizeof(cl_ulong) * 60, NULL);

    error=clEnqueueNDRangeKernel(device->queue[gpumgmt_thread->threadId], device->kernelPosthash[gpumgmt_thread->threadId], 1, NULL, &total_work_items, &local_work_items, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->errorMessage = "Error running the kernel.";
        gpumgmt_thread->unlock();
        return false;
    }

    error = clEnqueueReadBuffer(device->queue[gpumgmt_thread->threadId], device->arguments.hashMemory[gpumgmt_thread->threadId], CL_FALSE, 0, threads * (xmrig::ARGON2_HASHLEN + 4), memory, 0, NULL, NULL);
    if (error != CL_SUCCESS) {
        device->error = error;
        device->errorMessage = "Error reading gpu memory.";
        gpumgmt_thread->unlock();
        return false;
    }

    error=clFinish(device->queue[gpumgmt_thread->threadId]);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->errorMessage = "Error flushing GPU queue.";
        gpumgmt_thread->unlock();
        return false;
    }

    gpumgmt_thread->unlock();

    return true;
}

void OpenCLHasher::buildThreadData() {
    m_threadData = new OpenCLGpuMgmtThreadData[m_enabledDevices.size() * 2];

    for(int i=0; i < m_enabledDevices.size(); i++) {
        OpenCLDeviceInfo *device = m_enabledDevices[i];
        for(int threadId = 0; threadId < 2; threadId ++) {
            OpenCLGpuMgmtThreadData &thread_data = m_threadData[i * 2 + threadId];
            thread_data.device = device;
            thread_data.threadId = threadId;

#ifdef PARALLEL_OPENCL
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

            thread_data.argon2 = new Argon2(opencl_kernel_prehasher, opencl_kernel_filler, opencl_kernel_posthasher,
                                            nullptr, &thread_data);
            thread_data.argon2->setThreads(thread_data.threads);
            thread_data.hashData.outSize = xmrig::ARGON2_HASHLEN + 4;
        }
    }
}

int OpenCLHasher::compute(int threadIdx, uint8_t *input, size_t size, uint8_t *output) {
    OpenCLGpuMgmtThreadData &threadData = m_threadData[threadIdx];
    threadData.hashData.input = input;
    threadData.hashData.inSize = size;
    threadData.hashData.output = output;
    int hashCount = threadData.argon2->generateHashes(*m_profile, threadData.hashData);
    if(threadData.device->error != CL_SUCCESS) {
        LOG("Error running kernel: (" + to_string(threadData.device->error) + ")" + threadData.device->errorMessage);
        return 0;
    }

    uint32_t *nonce = ((uint32_t *)(((uint8_t*)threadData.hashData.input) + 39));
    (*nonce) += threadData.threads;

    return hashCount;
}

void OpenCLHasher::cleanup() {
    vector<cl_platform_id> platforms;

    for(vector<OpenCLDeviceInfo *>::iterator it=m_devices.begin(); it != m_devices.end(); it++) {
		if ((*it)->profileInfo.threads != 0) {
			clReleaseMemObject((*it)->arguments.memoryChunk_0);
			clReleaseMemObject((*it)->arguments.memoryChunk_1);
			clReleaseMemObject((*it)->arguments.memoryChunk_2);
			clReleaseMemObject((*it)->arguments.memoryChunk_3);
			clReleaseMemObject((*it)->arguments.memoryChunk_4);
			clReleaseMemObject((*it)->arguments.memoryChunk_5);
			clReleaseMemObject((*it)->arguments.refs);
			clReleaseMemObject((*it)->arguments.segments);
            clReleaseMemObject((*it)->arguments.preseedMemory[0]);
            clReleaseMemObject((*it)->arguments.preseedMemory[1]);
            clReleaseMemObject((*it)->arguments.seedMemory[0]);
            clReleaseMemObject((*it)->arguments.seedMemory[1]);
            clReleaseMemObject((*it)->arguments.outMemory[0]);
            clReleaseMemObject((*it)->arguments.outMemory[1]);
            clReleaseMemObject((*it)->arguments.hashMemory[0]);
            clReleaseMemObject((*it)->arguments.hashMemory[1]);

            clReleaseKernel((*it)->kernelPrehash[0]);
            clReleaseKernel((*it)->kernelPrehash[1]);
            clReleaseKernel((*it)->kernelFillBlocks[0]);
            clReleaseKernel((*it)->kernelFillBlocks[1]);
            clReleaseKernel((*it)->kernelPosthash[0]);
            clReleaseKernel((*it)->kernelPosthash[1]);
			clReleaseProgram((*it)->program);
            clReleaseCommandQueue((*it)->queue[0]);
            clReleaseCommandQueue((*it)->queue[1]);
			clReleaseContext((*it)->context);
		}
        clReleaseDevice((*it)->device);
        delete (*it);
	}
    m_devices.clear();
}

size_t OpenCLHasher::parallelism(int workerIdx) {
    OpenCLGpuMgmtThreadData &threadData = m_threadData[workerIdx];
    return threadData.threads;
}

size_t OpenCLHasher::deviceCount() {
    return m_enabledDevices.size();
}

DeviceInfo &OpenCLHasher::device(int workerIdx) {
    workerIdx /= 2;

    if(workerIdx < 0 || workerIdx > m_enabledDevices.size())
        return devices().begin()->second;

    return devices()[m_enabledDevices[workerIdx]->deviceIndex];
}

REGISTER_HASHER(OpenCLHasher);

#endif // WITH_OPENCL
