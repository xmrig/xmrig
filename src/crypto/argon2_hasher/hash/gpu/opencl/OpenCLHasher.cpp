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

opencl_hasher::opencl_hasher() {
    m_type = "GPU";
    m_subType = "OPENCL";
    m_shortSubType = "OCL";
    m_intensity = 0;
    m_description = "";
    m_computingThreads = 0;
}

opencl_hasher::~opencl_hasher() {
//    this->cleanup();
}

bool opencl_hasher::initialize(xmrig::Algo algorithm, xmrig::Variant variant) {
    cl_int error = CL_SUCCESS;
    string error_message;

    m_profile = getArgon2Profile(algorithm, variant);

    __devices = __query_opencl_devices(error, error_message);
    if(error != CL_SUCCESS) {
        m_description = "No compatible GPU detected: " + error_message;
        return false;
    }

    if (__devices.empty()) {
        m_description = "No compatible GPU detected.";
        return false;
    }

    return true;
}

vector<opencl_device_info*> opencl_hasher::__query_opencl_devices(cl_int &error, string &error_message) {
    cl_int err;

    cl_uint platform_count = 0;
    cl_uint device_count = 0;

    vector<opencl_device_info*> result;

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
            opencl_device_info *info = __get_device_info(platforms[i], devices[j]);
            if(info->error != CL_SUCCESS) {
                error = info->error;
                error_message = info->error_message;
            }
            else {
                info->device_index = counter;
                result.push_back(info);
                counter++;
            }
        }

        free(devices);
    }

    free(platforms);

    return result;
}

opencl_device_info *opencl_hasher::__get_device_info(cl_platform_id platform, cl_device_id device) {
    opencl_device_info *device_info = new opencl_device_info(CL_SUCCESS, "");

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
        device_info->error_message = "Error querying device vendor.";
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
        device_info->error_message = "Error querying device name.";
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
        device_info->error_message = "Error querying device version.";
        return device_info;
    }
    else {
        buffer[sz] = 0;
        device_version = buffer;
        free(buffer);
    }

    device_info->device_string = device_vendor + " - " + device_name/* + " : " + device_version*/;

    device_info->error = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(device_info->max_mem_size), &(device_info->max_mem_size), NULL);
    if(device_info->error != CL_SUCCESS) {
        device_info->error_message = "Error querying device global memory size.";
        return device_info;
    }

    device_info->error = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(device_info->max_allocable_mem_size), &(device_info->max_allocable_mem_size), NULL);
    if(device_info->error != CL_SUCCESS) {
        device_info->error_message = "Error querying device max memory allocation.";
        return device_info;
    }

    double mem_in_gb = device_info->max_mem_size / 1073741824.0;
    stringstream ss;
    ss << setprecision(2) << mem_in_gb;
    device_info->device_string += (" (" + ss.str() + "GB)");

    return device_info;
}

bool opencl_hasher::configure(xmrig::HasherConfig &config) {
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

    for(vector<opencl_device_info *>::iterator d = __devices.begin(); d != __devices.end(); d++, index++) {
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

        if((*d)->device_string.find("Advanced Micro Devices") != string::npos) {
            device_topology_amd amdtopo;
            if(clGetDeviceInfo((*d)->device, CL_DEVICE_TOPOLOGY_AMD, sizeof(amdtopo), &amdtopo, NULL) == CL_SUCCESS) {
                char bus_id[50];
                sprintf(bus_id, "%02x:%02x.%x", amdtopo.pcie.bus, amdtopo.pcie.device, amdtopo.pcie.function);
                device.bus_id = bus_id;
            }
        }
        else if((*d)->device_string.find("NVIDIA") != string::npos) {
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

    buildThreadData();

    m_intensity = intensity / __enabledDevices.size();
    m_computingThreads = __enabledDevices.size() * 2; // 2 computing threads for each device
    m_description += "Status: ENABLED - with " + to_string(total_threads) + " threads.";

    return true;
}

bool opencl_hasher::__setup_device_info(opencl_device_info *device, double intensity) {
    cl_int error;

    cl_context_properties properties[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties) device->platform,
            0};

    device->context = clCreateContext(properties, 1, &(device->device), NULL, NULL, &error);
    if (error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error getting device context.";
        return false;
    }

    device->queue = clCreateCommandQueue(device->context, device->device, CL_QUEUE_PROFILING_ENABLE, &error);
    if (error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error getting device command queue.";
        return false;
    }

    const char *srcptr[] = {OpenCLKernel.c_str()};
    size_t srcsize = OpenCLKernel.size();

    device->program = clCreateProgramWithSource(device->context, 1, srcptr, &srcsize, &error);
    if (error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error creating opencl program for device.";
        return false;
    }

    error = clBuildProgram(device->program, 1, &device->device, "", NULL, NULL);
    if (error != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(device->program, device->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *) malloc(log_size + 1);
        clGetProgramBuildInfo(device->program, device->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size] = 0;
        string build_log = log;
        free(log);

        device->error = error;
        device->error_message = "Error building opencl program for device: " + build_log;
        return false;
    }

    device->kernel_prehash = clCreateKernel(device->program, "prehash", &error);
    if (error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error creating opencl prehash kernel for device.";
        return false;
    }
    device->kernel_fill_blocks = clCreateKernel(device->program, "fill_blocks", &error);
    if (error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error creating opencl main kernel for device.";
        return false;
    }
    device->kernel_posthash = clCreateKernel(device->program, "posthash", &error);
    if (error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error creating opencl posthash kernel for device.";
        return false;
    }

    device->profile_info.threads_per_chunk = (uint32_t) (device->max_allocable_mem_size / device->profile_info.profile->memSize);
    size_t chunk_size = device->profile_info.threads_per_chunk * device->profile_info.profile->memSize;

    if (chunk_size == 0) {
        device->error = -1;
        device->error_message = "Not enough memory on GPU.";
        return false;
    }

    uint64_t usable_memory = device->max_mem_size;
    double chunks = (double) usable_memory / (double) chunk_size;

    uint32_t max_threads = (uint32_t) (device->profile_info.threads_per_chunk * chunks);

    if (max_threads == 0) {
        device->error = -1;
        device->error_message = "Not enough memory on GPU.";
        return false;
    }

    device->profile_info.threads = (uint32_t) (max_threads * intensity / 100.0);
    device->profile_info.threads = (device->profile_info.threads / 4) * 4; // make it divisible by 4
    if (max_threads > 0 && device->profile_info.threads == 0 && intensity > 0)
        device->profile_info.threads = 4;

    double counter = (double) device->profile_info.threads / (double) device->profile_info.threads_per_chunk;
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
    device->arguments.memory_chunk_0 = clCreateBuffer(device->context, CL_MEM_READ_WRITE,
                                                      allocated_mem_for_current_chunk, NULL, &error);
    if (error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error creating memory buffer.";
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
    device->arguments.memory_chunk_1 = clCreateBuffer(device->context, CL_MEM_READ_WRITE,
                                                      allocated_mem_for_current_chunk, NULL, &error);
    if (error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error creating memory buffer.";
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
    device->arguments.memory_chunk_2 = clCreateBuffer(device->context, CL_MEM_READ_WRITE,
                                                      allocated_mem_for_current_chunk, NULL, &error);
    if (error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error creating memory buffer.";
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
    device->arguments.memory_chunk_3 = clCreateBuffer(device->context, CL_MEM_READ_WRITE,
                                                      allocated_mem_for_current_chunk, NULL, &error);
    if (error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error creating memory buffer.";
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
    device->arguments.memory_chunk_4 = clCreateBuffer(device->context, CL_MEM_READ_WRITE,
                                                      allocated_mem_for_current_chunk, NULL, &error);
    if (error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error creating memory buffer.";
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
    device->arguments.memory_chunk_5 = clCreateBuffer(device->context, CL_MEM_READ_WRITE,
                                                      allocated_mem_for_current_chunk, NULL, &error);
    if (error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error creating memory buffer.";
        return false;
    }

    device->arguments.refs = clCreateBuffer(device->context, CL_MEM_READ_ONLY,
                                            device->profile_info.profile->blockRefsSize * sizeof(uint32_t), NULL,
                                            &error);
    if (error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error creating memory buffer.";
        return false;
    }

    if (device->profile_info.profile->succesiveIdxs == 1) {
        device->arguments.idxs = NULL;
    }
    else {
        device->arguments.idxs = clCreateBuffer(device->context, CL_MEM_READ_ONLY,
                                                device->profile_info.profile->blockRefsSize * sizeof(uint32_t), NULL,
                                                &error);
        if (error != CL_SUCCESS) {
            device->error = error;
            device->error_message = "Error creating memory buffer.";
            return false;
        }
    }

    device->arguments.segments = clCreateBuffer(device->context, CL_MEM_READ_ONLY, device->profile_info.profile->segCount * 3 * sizeof(uint32_t), NULL, &error);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error creating memory buffer.";
        return false;
    }

    size_t preseed_memory_size = device->profile_info.profile->pwdLen * 4;
    size_t seed_memory_size = device->profile_info.threads * (device->profile_info.profile->thrCost * 2) * ARGON2_BLOCK_SIZE;
    size_t out_memory_size = device->profile_info.threads * ARGON2_BLOCK_SIZE;
    size_t hash_memory_size = device->profile_info.threads * (xmrig::ARGON2_HASHLEN + 4);

    device->arguments.preseed_memory[0] = clCreateBuffer(device->context, CL_MEM_READ_ONLY, preseed_memory_size, NULL, &error);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error creating memory buffer.";
        return false;
    }

    device->arguments.preseed_memory[1] = clCreateBuffer(device->context, CL_MEM_READ_ONLY, preseed_memory_size, NULL, &error);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error creating memory buffer.";
        return false;
    }

    device->arguments.seed_memory[0] = clCreateBuffer(device->context, CL_MEM_READ_WRITE, seed_memory_size, NULL, &error);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error creating memory buffer.";
        return false;
    }

    device->arguments.seed_memory[1] = clCreateBuffer(device->context, CL_MEM_READ_WRITE, seed_memory_size, NULL, &error);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error creating memory buffer.";
        return false;
    }

    device->arguments.out_memory[0] = clCreateBuffer(device->context, CL_MEM_READ_WRITE, out_memory_size, NULL, &error);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error creating memory buffer.";
        return false;
    }

    device->arguments.out_memory[1] = clCreateBuffer(device->context, CL_MEM_READ_WRITE, out_memory_size, NULL, &error);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error creating memory buffer.";
        return false;
    }

    device->arguments.hash_memory[0] = clCreateBuffer(device->context, CL_MEM_WRITE_ONLY, hash_memory_size, NULL, &error);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error creating memory buffer.";
        return false;
    }

    device->arguments.hash_memory[1] = clCreateBuffer(device->context, CL_MEM_WRITE_ONLY, hash_memory_size, NULL, &error);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error creating memory buffer.";
        return false;
    }

	//optimise address sizes
    uint32_t *refs = (uint32_t *)malloc(device->profile_info.profile->blockRefsSize * sizeof(uint32_t));
    for(int i=0;i<device->profile_info.profile->blockRefsSize;i++) {
        refs[i] = device->profile_info.profile->blockRefs[i*3 + 1];
    }

    error=clEnqueueWriteBuffer(device->queue, device->arguments.refs, CL_TRUE, 0, device->profile_info.profile->blockRefsSize * sizeof(uint32_t), refs, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error writing to gpu memory.";
        return false;
    }

    free(refs);

    if(device->profile_info.profile->succesiveIdxs == 0) {
        uint32_t *idxs = (uint32_t *) malloc(device->profile_info.profile->blockRefsSize * sizeof(uint32_t));
        for (int i = 0; i < device->profile_info.profile->blockRefsSize; i++) {
            idxs[i] = device->profile_info.profile->blockRefs[i * 3];
            if (device->profile_info.profile->blockRefs[i * 3 + 2] == 1) {
                idxs[i] |= 0x80000000;
            }
        }

        error=clEnqueueWriteBuffer(device->queue, device->arguments.idxs, CL_TRUE, 0, device->profile_info.profile->blockRefsSize * sizeof(uint32_t), idxs, 0, NULL, NULL);
        if(error != CL_SUCCESS) {
            device->error = error;
            device->error_message = "Error writing to gpu memory.";
            return false;
        }

        free(idxs);
    }

    error=clEnqueueWriteBuffer(device->queue, device->arguments.segments, CL_TRUE, 0, device->profile_info.profile->segCount * 3 * sizeof(uint32_t), device->profile_info.profile->segments, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error writing to gpu memory.";
        return false;
    }

	clSetKernelArg(device->kernel_fill_blocks, 0, sizeof(device->arguments.memory_chunk_0), &device->arguments.memory_chunk_0);
	clSetKernelArg(device->kernel_fill_blocks, 1, sizeof(device->arguments.memory_chunk_1), &device->arguments.memory_chunk_1);
	clSetKernelArg(device->kernel_fill_blocks, 2, sizeof(device->arguments.memory_chunk_2), &device->arguments.memory_chunk_2);
	clSetKernelArg(device->kernel_fill_blocks, 3, sizeof(device->arguments.memory_chunk_3), &device->arguments.memory_chunk_3);
	clSetKernelArg(device->kernel_fill_blocks, 4, sizeof(device->arguments.memory_chunk_4), &device->arguments.memory_chunk_4);
	clSetKernelArg(device->kernel_fill_blocks, 5, sizeof(device->arguments.memory_chunk_5), &device->arguments.memory_chunk_5);
    clSetKernelArg(device->kernel_fill_blocks, 8, sizeof(device->arguments.refs), &device->arguments.refs);
    if(device->profile_info.profile->succesiveIdxs == 0)
        clSetKernelArg(device->kernel_fill_blocks, 9, sizeof(device->arguments.idxs), &device->arguments.idxs);
    else
        clSetKernelArg(device->kernel_fill_blocks, 9, sizeof(cl_mem), NULL);
	clSetKernelArg(device->kernel_fill_blocks, 10, sizeof(device->arguments.segments), &device->arguments.segments);
    clSetKernelArg(device->kernel_fill_blocks, 11, sizeof(int32_t), &device->profile_info.profile->memSize);
    clSetKernelArg(device->kernel_fill_blocks, 12, sizeof(int32_t), &device->profile_info.profile->thrCost);
    clSetKernelArg(device->kernel_fill_blocks, 13, sizeof(int32_t), &device->profile_info.profile->segSize);
    clSetKernelArg(device->kernel_fill_blocks, 14, sizeof(int32_t), &device->profile_info.profile->segCount);
    clSetKernelArg(device->kernel_fill_blocks, 15, sizeof(int32_t), &device->profile_info.threads_per_chunk);

    clSetKernelArg(device->kernel_prehash, 2, sizeof(int32_t), &device->profile_info.profile->memCost);
    clSetKernelArg(device->kernel_prehash, 3, sizeof(int32_t), &device->profile_info.profile->thrCost);
    int passes = device->profile_info.profile->segCount / (4 * device->profile_info.profile->thrCost);
    clSetKernelArg(device->kernel_prehash, 4, sizeof(int32_t), &passes);
    clSetKernelArg(device->kernel_prehash, 6, sizeof(int32_t), &device->profile_info.profile->saltLen);

    return true;
}

bool opencl_kernel_prehasher(void *memory, int threads, Argon2Profile *profile, void *user_data) {
    opencl_gpumgmt_thread_data *gpumgmt_thread = (opencl_gpumgmt_thread_data *)user_data;
    opencl_device_info *device = gpumgmt_thread->device;

    cl_int error;

    int sessions = max(profile->thrCost * 2, (uint32_t)16);
    double hashes_per_block = sessions / (profile->thrCost * 2.0);

    size_t total_work_items = sessions * 4 * ceil(threads / hashes_per_block);
    size_t local_work_items = sessions * 4;

    device->device_lock.lock();

    error = clEnqueueWriteBuffer(device->queue, device->arguments.preseed_memory[gpumgmt_thread->thread_id],
                                 CL_FALSE, 0, gpumgmt_thread->hashData.inSize, memory, 0, NULL, NULL);
    if (error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error writing to gpu memory.";
        device->device_lock.unlock();
        return false;
    }

    int inSizeInInt = gpumgmt_thread->hashData.inSize / 4;
    clSetKernelArg(device->kernel_prehash, 0, sizeof(device->arguments.preseed_memory[gpumgmt_thread->thread_id]), &device->arguments.preseed_memory[gpumgmt_thread->thread_id]);
    clSetKernelArg(device->kernel_prehash, 1, sizeof(device->arguments.seed_memory[gpumgmt_thread->thread_id]), &device->arguments.seed_memory[gpumgmt_thread->thread_id]);
    clSetKernelArg(device->kernel_prehash, 5, sizeof(int), &inSizeInInt);
    clSetKernelArg(device->kernel_prehash, 7, sizeof(int), &threads);
    clSetKernelArg(device->kernel_prehash, 8, sessions * sizeof(cl_ulong) * 76, NULL); // (preseed size is 16 ulongs = 128 bytes)

    error=clEnqueueNDRangeKernel(device->queue, device->kernel_prehash, 1, NULL, &total_work_items, &local_work_items, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error running the kernel.";
        device->device_lock.unlock();
        return false;
    }

    return true;
}

void *opencl_kernel_filler(int threads, Argon2Profile *profile, void *user_data) {
	opencl_gpumgmt_thread_data *gpumgmt_thread = (opencl_gpumgmt_thread_data *)user_data;
    opencl_device_info *device = gpumgmt_thread->device;

    cl_int error;

	size_t total_work_items = threads * KERNEL_WORKGROUP_SIZE * profile->thrCost;
	size_t local_work_items = KERNEL_WORKGROUP_SIZE * profile->thrCost;

    size_t shared_mem = profile->thrCost * ARGON2_QWORDS_IN_BLOCK;

    clSetKernelArg(device->kernel_fill_blocks, 6, sizeof(device->arguments.seed_memory[gpumgmt_thread->thread_id]), &device->arguments.seed_memory[gpumgmt_thread->thread_id]);
    clSetKernelArg(device->kernel_fill_blocks, 7, sizeof(device->arguments.out_memory[gpumgmt_thread->thread_id]), &device->arguments.out_memory[gpumgmt_thread->thread_id]);
    clSetKernelArg(device->kernel_fill_blocks, 16, sizeof(cl_ulong) * shared_mem, NULL);

    error=clEnqueueNDRangeKernel(device->queue, device->kernel_fill_blocks, 1, NULL, &total_work_items, &local_work_items, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error running the kernel.";
        device->device_lock.unlock();
        return NULL;
    }

	return (void *)1;
}

bool opencl_kernel_posthasher(void *memory, int threads, Argon2Profile *profile, void *user_data) {
    opencl_gpumgmt_thread_data *gpumgmt_thread = (opencl_gpumgmt_thread_data *)user_data;
    opencl_device_info *device = gpumgmt_thread->device;

    cl_int error;

    size_t total_work_items = threads * 4;
    size_t local_work_items = 4;

    clSetKernelArg(device->kernel_posthash, 0, sizeof(device->arguments.hash_memory[gpumgmt_thread->thread_id]), &device->arguments.hash_memory[gpumgmt_thread->thread_id]);
    clSetKernelArg(device->kernel_posthash, 1, sizeof(device->arguments.out_memory[gpumgmt_thread->thread_id]), &device->arguments.out_memory[gpumgmt_thread->thread_id]);
    clSetKernelArg(device->kernel_posthash, 2, sizeof(device->arguments.preseed_memory[gpumgmt_thread->thread_id]), &device->arguments.preseed_memory[gpumgmt_thread->thread_id]);
    clSetKernelArg(device->kernel_posthash, 3, sizeof(cl_ulong) * 60, NULL);

    error=clEnqueueNDRangeKernel(device->queue, device->kernel_posthash, 1, NULL, &total_work_items, &local_work_items, 0, NULL, NULL);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error running the kernel.";
        device->device_lock.unlock();
        return false;
    }

    error = clEnqueueReadBuffer(device->queue, device->arguments.hash_memory[gpumgmt_thread->thread_id], CL_FALSE, 0, threads * (xmrig::ARGON2_HASHLEN + 4), memory, 0, NULL, NULL);
    if (error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error reading gpu memory.";
        device->device_lock.unlock();
        return false;
    }

    error=clFinish(device->queue);
    if(error != CL_SUCCESS) {
        device->error = error;
        device->error_message = "Error flushing GPU queue.";
        device->device_lock.unlock();
        return false;
    }

    device->device_lock.unlock();

    return true;
}

void opencl_hasher::buildThreadData() {
    __thread_data = new opencl_gpumgmt_thread_data[__enabledDevices.size() * 2];

    for(int i=0; i < __enabledDevices.size(); i++) {
        opencl_device_info *device = __enabledDevices[i];
        for(int threadId = 0; threadId < 2; threadId ++) {
            opencl_gpumgmt_thread_data &thread_data = __thread_data[i * 2 + threadId];
            thread_data.device = device;
            thread_data.thread_id = threadId;
            thread_data.argon2 = new Argon2(opencl_kernel_prehasher, opencl_kernel_filler, opencl_kernel_posthasher,
                                            nullptr, &thread_data);
            thread_data.argon2->setThreads(device->profile_info.threads);
            thread_data.hashData.outSize = xmrig::ARGON2_HASHLEN + 4;
        }
    }
}

int opencl_hasher::compute(int threadIdx, uint8_t *input, size_t size, uint8_t *output) {
    opencl_gpumgmt_thread_data &threadData = __thread_data[threadIdx];
    threadData.hashData.input = input;
    threadData.hashData.inSize = size;
    threadData.hashData.output = output;
    int hashCount = threadData.argon2->generateHashes(*m_profile, threadData.hashData);
    if(threadData.device->error != CL_SUCCESS) {
        LOG("Error running kernel: (" + to_string(threadData.device->error) + ")" + threadData.device->error_message);
        return 0;
    }

    uint32_t *nonce = ((uint32_t *)(((uint8_t*)threadData.hashData.input) + 39));
    (*nonce) += threadData.device->profile_info.threads;

    return hashCount;
}

void opencl_hasher::cleanup() {
    vector<cl_platform_id> platforms;

    for(vector<opencl_device_info *>::iterator it=__devices.begin(); it != __devices.end(); it++) {
		if ((*it)->profile_info.threads != 0) {
			clReleaseMemObject((*it)->arguments.memory_chunk_0);
			clReleaseMemObject((*it)->arguments.memory_chunk_1);
			clReleaseMemObject((*it)->arguments.memory_chunk_2);
			clReleaseMemObject((*it)->arguments.memory_chunk_3);
			clReleaseMemObject((*it)->arguments.memory_chunk_4);
			clReleaseMemObject((*it)->arguments.memory_chunk_5);
			clReleaseMemObject((*it)->arguments.refs);
			clReleaseMemObject((*it)->arguments.segments);
            clReleaseMemObject((*it)->arguments.preseed_memory[0]);
            clReleaseMemObject((*it)->arguments.preseed_memory[1]);
            clReleaseMemObject((*it)->arguments.seed_memory[0]);
            clReleaseMemObject((*it)->arguments.seed_memory[1]);
            clReleaseMemObject((*it)->arguments.out_memory[0]);
            clReleaseMemObject((*it)->arguments.out_memory[1]);
            clReleaseMemObject((*it)->arguments.hash_memory[0]);
            clReleaseMemObject((*it)->arguments.hash_memory[1]);

            clReleaseKernel((*it)->kernel_prehash);
            clReleaseKernel((*it)->kernel_fill_blocks);
            clReleaseKernel((*it)->kernel_posthash);
			clReleaseProgram((*it)->program);
			clReleaseCommandQueue((*it)->queue);
			clReleaseContext((*it)->context);
		}
        clReleaseDevice((*it)->device);
        delete (*it);
	}
    __devices.clear();
}

size_t opencl_hasher::parallelism(int workerIdx) {
    // there are 2 computing threads per device, so divide by 2 to get device index
    workerIdx /= 2;

    if(workerIdx < 0 || workerIdx > __enabledDevices.size())
        return 0;

    return __enabledDevices[workerIdx]->profile_info.threads;
}

size_t opencl_hasher::deviceCount() {
    return __enabledDevices.size();
}

REGISTER_HASHER(opencl_hasher);

#endif // WITH_OPENCL
