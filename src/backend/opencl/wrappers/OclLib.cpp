/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2019 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program. If not, see <http://www.gnu.org/licenses/>.
 */


#include <thread>
#include <uv.h>


#include "backend/opencl/wrappers/OclError.h"
#include "backend/opencl/wrappers/OclLib.h"
#include "base/io/log/Log.h"


static uv_lib_t oclLib;

static const char *kErrorTemplate                    = MAGENTA_BG_BOLD(WHITE_BOLD_S " ocl ") RED(" error ") RED_BOLD("%s") RED(" when calling ") RED_BOLD("%s");

static const char *kBuildProgram                     = "clBuildProgram";
static const char *kCreateBuffer                     = "clCreateBuffer";
static const char *kCreateCommandQueue               = "clCreateCommandQueue";
static const char *kCreateCommandQueueWithProperties = "clCreateCommandQueueWithProperties";
static const char *kCreateContext                    = "clCreateContext";
static const char *kCreateKernel                     = "clCreateKernel";
static const char *kCreateProgramWithBinary          = "clCreateProgramWithBinary";
static const char *kCreateProgramWithSource          = "clCreateProgramWithSource";
static const char *kEnqueueNDRangeKernel             = "clEnqueueNDRangeKernel";
static const char *kEnqueueReadBuffer                = "clEnqueueReadBuffer";
static const char *kEnqueueWriteBuffer               = "clEnqueueWriteBuffer";
static const char *kFinish                           = "clFinish";
static const char *kGetCommandQueueInfo              = "clGetCommandQueueInfo";
static const char *kGetDeviceIDs                     = "clGetDeviceIDs";
static const char *kGetDeviceInfo                    = "clGetDeviceInfo";
static const char *kGetKernelInfo                    = "clGetKernelInfo";
static const char *kGetPlatformIDs                   = "clGetPlatformIDs";
static const char *kGetPlatformInfo                  = "clGetPlatformInfo";
static const char *kGetProgramBuildInfo              = "clGetProgramBuildInfo";
static const char *kGetProgramInfo                   = "clGetProgramInfo";
static const char *kReleaseCommandQueue              = "clReleaseCommandQueue";
static const char *kReleaseContext                   = "clReleaseContext";
static const char *kReleaseKernel                    = "clReleaseKernel";
static const char *kReleaseMemObject                 = "clReleaseMemObject";
static const char *kReleaseProgram                   = "clReleaseProgram";
static const char *kSetKernelArg                     = "clSetKernelArg";

#if defined(CL_VERSION_2_0)
typedef cl_command_queue (CL_API_CALL *createCommandQueueWithProperties_t)(cl_context, cl_device_id, const cl_queue_properties *, cl_int *);
#endif

typedef cl_command_queue (CL_API_CALL *createCommandQueue_t)(cl_context, cl_device_id, cl_command_queue_properties, cl_int *);
typedef cl_context (CL_API_CALL *createContext_t)(const cl_context_properties *, cl_uint, const cl_device_id *, void (CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *), void *, cl_int *);
typedef cl_int (CL_API_CALL *buildProgram_t)(cl_program, cl_uint, const cl_device_id *, const char *, void (CL_CALLBACK *pfn_notify)(cl_program, void *), void *);
typedef cl_int (CL_API_CALL *enqueueNDRangeKernel_t)(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
typedef cl_int (CL_API_CALL *enqueueReadBuffer_t)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *);
typedef cl_int (CL_API_CALL *enqueueWriteBuffer_t)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *);
typedef cl_int (CL_API_CALL *finish_t)(cl_command_queue);
typedef cl_int (CL_API_CALL *getCommandQueueInfo_t)(cl_command_queue, cl_command_queue_info, size_t, void *, size_t *);
typedef cl_int (CL_API_CALL *getDeviceIDs_t)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
typedef cl_int (CL_API_CALL *getDeviceInfo_t)(cl_device_id, cl_device_info, size_t, void *, size_t *);
typedef cl_int (CL_API_CALL *getKernelInfo_t)(cl_kernel, cl_kernel_info, size_t, void *, size_t *);
typedef cl_int (CL_API_CALL *getPlatformIDs_t)(cl_uint, cl_platform_id *, cl_uint *);
typedef cl_int (CL_API_CALL *getPlatformInfo_t)(cl_platform_id, cl_platform_info, size_t, void *, size_t *);
typedef cl_int (CL_API_CALL *getProgramBuildInfo_t)(cl_program, cl_device_id, cl_program_build_info, size_t, void *, size_t *);
typedef cl_int (CL_API_CALL *getProgramInfo_t)(cl_program, cl_program_info, size_t, void *, size_t *);
typedef cl_int (CL_API_CALL *releaseCommandQueue_t)(cl_command_queue);
typedef cl_int (CL_API_CALL *releaseContext_t)(cl_context);
typedef cl_int (CL_API_CALL *releaseKernel_t)(cl_kernel);
typedef cl_int (CL_API_CALL *releaseMemObject_t)(cl_mem);
typedef cl_int (CL_API_CALL *releaseProgram_t)(cl_program);
typedef cl_int (CL_API_CALL *setKernelArg_t)(cl_kernel, cl_uint, size_t, const void *);
typedef cl_kernel (CL_API_CALL *createKernel_t)(cl_program, const char *, cl_int *);
typedef cl_mem (CL_API_CALL *createBuffer_t)(cl_context, cl_mem_flags, size_t, void *, cl_int *);
typedef cl_program (CL_API_CALL *createProgramWithBinary_t)(cl_context, cl_uint, const cl_device_id *, const size_t *, const unsigned char **, cl_int *, cl_int *);
typedef cl_program (CL_API_CALL *createProgramWithSource_t)(cl_context, cl_uint, const char **, const size_t *, cl_int *);


#if defined(CL_VERSION_2_0)
static createCommandQueueWithProperties_t pCreateCommandQueueWithProperties = nullptr;
#endif

static buildProgram_t  pBuildProgram                                        = nullptr;
static createBuffer_t pCreateBuffer                                         = nullptr;
static createCommandQueue_t pCreateCommandQueue                             = nullptr;
static createContext_t pCreateContext                                       = nullptr;
static createKernel_t pCreateKernel                                         = nullptr;
static createProgramWithBinary_t pCreateProgramWithBinary                   = nullptr;
static createProgramWithSource_t pCreateProgramWithSource                   = nullptr;
static enqueueNDRangeKernel_t pEnqueueNDRangeKernel                         = nullptr;
static enqueueReadBuffer_t pEnqueueReadBuffer                               = nullptr;
static enqueueWriteBuffer_t pEnqueueWriteBuffer                             = nullptr;
static finish_t pFinish                                                     = nullptr;
static getCommandQueueInfo_t pGetCommandQueueInfo                           = nullptr;
static getDeviceIDs_t pGetDeviceIDs                                         = nullptr;
static getDeviceInfo_t pGetDeviceInfo                                       = nullptr;
static getKernelInfo_t pGetKernelInfo                                       = nullptr;
static getPlatformIDs_t pGetPlatformIDs                                     = nullptr;
static getPlatformInfo_t pGetPlatformInfo                                   = nullptr;
static getProgramBuildInfo_t pGetProgramBuildInfo                           = nullptr;
static getProgramInfo_t pGetProgramInfo                                     = nullptr;
static releaseCommandQueue_t pReleaseCommandQueue                           = nullptr;
static releaseContext_t pReleaseContext                                     = nullptr;
static releaseKernel_t pReleaseKernel                                       = nullptr;
static releaseMemObject_t pReleaseMemObject                                 = nullptr;
static releaseProgram_t pReleaseProgram                                     = nullptr;
static setKernelArg_t pSetKernelArg                                         = nullptr;

#define DLSYM(x) if (uv_dlsym(&oclLib, k##x, reinterpret_cast<void**>(&p##x)) == -1) { return false; }


namespace xmrig {

bool OclLib::m_initialized = false;
bool OclLib::m_ready       = false;
String OclLib::m_loader;

} // namespace xmrig


bool xmrig::OclLib::init(const char *fileName)
{
    if (!m_initialized) {
        m_loader      = fileName == nullptr ? defaultLoader() : fileName;
        m_ready       = uv_dlopen(m_loader, &oclLib) == 0 && load();
        m_initialized = true;
    }

    return m_ready;
}


const char *xmrig::OclLib::lastError()
{
    return uv_dlerror(&oclLib);
}


void xmrig::OclLib::close()
{
    uv_dlclose(&oclLib);
}


bool xmrig::OclLib::load()
{
    DLSYM(CreateCommandQueue);
    DLSYM(CreateContext);
    DLSYM(BuildProgram);
    DLSYM(EnqueueNDRangeKernel);
    DLSYM(EnqueueReadBuffer);
    DLSYM(EnqueueWriteBuffer);
    DLSYM(Finish);
    DLSYM(GetDeviceIDs);
    DLSYM(GetDeviceInfo);
    DLSYM(GetPlatformInfo);
    DLSYM(GetPlatformIDs);
    DLSYM(GetProgramBuildInfo);
    DLSYM(GetProgramInfo);
    DLSYM(SetKernelArg);
    DLSYM(CreateKernel);
    DLSYM(CreateBuffer);
    DLSYM(CreateProgramWithBinary);
    DLSYM(CreateProgramWithSource);
    DLSYM(ReleaseMemObject);
    DLSYM(ReleaseProgram);
    DLSYM(ReleaseKernel);
    DLSYM(ReleaseCommandQueue);
    DLSYM(ReleaseContext);
    DLSYM(GetKernelInfo);
    DLSYM(GetCommandQueueInfo);

#   if defined(CL_VERSION_2_0)
    uv_dlsym(&oclLib, kCreateCommandQueueWithProperties, reinterpret_cast<void**>(&pCreateCommandQueueWithProperties));
#   endif

    return true;
}


const char *xmrig::OclLib::defaultLoader()
{
#   if defined(__APPLE__)
    return "/System/Library/Frameworks/OpenCL.framework/OpenCL";
#   elif defined(_WIN32)
    return "OpenCL.dll";
#   else
    return "libOpenCL.so";
#   endif
}


cl_command_queue xmrig::OclLib::createCommandQueue(cl_context context, cl_device_id device, cl_int *errcode_ret)
{
    cl_command_queue result;

#   if defined(CL_VERSION_2_0)
    if (pCreateCommandQueueWithProperties) {
        const cl_queue_properties commandQueueProperties[] = { 0, 0, 0 };
        result = pCreateCommandQueueWithProperties(context, device, commandQueueProperties, errcode_ret);
    }
    else {
#   endif
        const cl_command_queue_properties commandQueueProperties = { 0 };
        result = pCreateCommandQueue(context, device, commandQueueProperties, errcode_ret);
#   if defined(CL_VERSION_2_0)
    }
#   endif

    if (*errcode_ret != CL_SUCCESS) {
        LOG_ERR(kErrorTemplate, OclError::toString(*errcode_ret), kCreateCommandQueueWithProperties);

        return nullptr;
    }

    return result;
}


cl_context xmrig::OclLib::createContext(const cl_context_properties *properties, cl_uint num_devices, const cl_device_id *devices, void (CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *), void *user_data, cl_int *errcode_ret)
{
    assert(pCreateContext != nullptr);

    auto result = pCreateContext(properties, num_devices, devices, pfn_notify, user_data, errcode_ret);
    if (*errcode_ret != CL_SUCCESS) {
        LOG_ERR(kErrorTemplate, OclError::toString(*errcode_ret), kCreateContext);

        return nullptr;
    }

    return result;
}


cl_context xmrig::OclLib::createContext(const std::vector<cl_device_id> &ids)
{
    cl_int ret;
    return OclLib::createContext(nullptr, static_cast<cl_uint>(ids.size()), ids.data(), nullptr, nullptr, &ret);
}


cl_int xmrig::OclLib::buildProgram(cl_program program, cl_uint num_devices, const cl_device_id *device_list, const char *options, void (CL_CALLBACK *pfn_notify)(cl_program program, void *user_data), void *user_data)
{
    assert(pBuildProgram != nullptr);

    const cl_int ret = pBuildProgram(program, num_devices, device_list, options, pfn_notify, user_data);
    if (ret != CL_SUCCESS) {
        LOG_ERR(kErrorTemplate, OclError::toString(ret), kBuildProgram);
    }

    return ret;
}


cl_int xmrig::OclLib::enqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t *global_work_offset, const size_t *global_work_size, const size_t *local_work_size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event)
{
    assert(pEnqueueNDRangeKernel != nullptr);

    return pEnqueueNDRangeKernel(command_queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size, num_events_in_wait_list, event_wait_list, event);
}


cl_int xmrig::OclLib::enqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, size_t offset, size_t size, void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event)
{
    assert(pEnqueueReadBuffer != nullptr);

    const cl_int ret = pEnqueueReadBuffer(command_queue, buffer, blocking_read, offset, size, ptr, num_events_in_wait_list, event_wait_list, event);
    if (ret != CL_SUCCESS) {
        LOG_ERR(kErrorTemplate, OclError::toString(ret), kEnqueueReadBuffer);
    }

    return ret;
}


cl_int xmrig::OclLib::enqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t offset, size_t size, const void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event)
{
    assert(pEnqueueWriteBuffer != nullptr);

    const cl_int ret = pEnqueueWriteBuffer(command_queue, buffer, blocking_write, offset, size, ptr, num_events_in_wait_list, event_wait_list, event);
    if (ret != CL_SUCCESS) {
        LOG_ERR(kErrorTemplate, OclError::toString(ret), kEnqueueWriteBuffer);
    }

    return ret;
}


cl_int xmrig::OclLib::finish(cl_command_queue command_queue)
{
    assert(pFinish != nullptr);

    return pFinish(command_queue);
}


cl_int xmrig::OclLib::getDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id *devices, cl_uint *num_devices)
{
    assert(pGetDeviceIDs != nullptr);

    return pGetDeviceIDs(platform, device_type, num_entries, devices, num_devices);
}


cl_int xmrig::OclLib::getDeviceInfo(cl_device_id device, cl_device_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret)
{
    assert(pGetDeviceInfo != nullptr);

    const cl_int ret = pGetDeviceInfo(device, param_name, param_value_size, param_value, param_value_size_ret);
    if (ret != CL_SUCCESS && param_name != 0x4038) {
        LOG_ERR("Error %s when calling %s, param 0x%04x", OclError::toString(ret), kGetDeviceInfo, param_name);
    }

    return ret;
}


cl_int xmrig::OclLib::getPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms)
{
    assert(pGetPlatformIDs != nullptr);

    return pGetPlatformIDs(num_entries, platforms, num_platforms);
}


cl_int xmrig::OclLib::getPlatformInfo(cl_platform_id platform, cl_platform_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret)
{
    assert(pGetPlatformInfo != nullptr);

    return pGetPlatformInfo(platform, param_name, param_value_size, param_value, param_value_size_ret);
}


cl_int xmrig::OclLib::getProgramBuildInfo(cl_program program, cl_device_id device, cl_program_build_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret)
{
    assert(pGetProgramBuildInfo != nullptr);

    const cl_int ret = pGetProgramBuildInfo(program, device, param_name, param_value_size, param_value, param_value_size_ret);
    if (ret != CL_SUCCESS) {
        LOG_ERR(kErrorTemplate, OclError::toString(ret), kGetProgramBuildInfo);
    }

    return ret;
}


cl_int xmrig::OclLib::getProgramInfo(cl_program program, cl_program_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret)
{
    assert(pGetProgramInfo != nullptr);

    const cl_int ret = pGetProgramInfo(program, param_name, param_value_size, param_value, param_value_size_ret);
    if (ret != CL_SUCCESS) {
        LOG_ERR(kErrorTemplate, OclError::toString(ret), kGetProgramInfo);
    }

    return ret;
}


cl_int xmrig::OclLib::release(cl_command_queue command_queue)
{
    assert(pReleaseCommandQueue != nullptr);
    assert(pGetCommandQueueInfo != nullptr);

    finish(command_queue);

    cl_int ret = pReleaseCommandQueue(command_queue);
    if (ret != CL_SUCCESS) {
        LOG_ERR(kErrorTemplate, OclError::toString(ret), kReleaseCommandQueue);
    }

    return ret;
}


cl_int xmrig::OclLib::release(cl_context context)
{
    assert(pReleaseContext != nullptr);

    const cl_int ret = pReleaseContext(context);
    if (ret != CL_SUCCESS) {
        LOG_ERR(kErrorTemplate, OclError::toString(ret), kReleaseContext);
    }

    return ret;
}


cl_int xmrig::OclLib::release(cl_kernel kernel)
{
    assert(pReleaseKernel != nullptr);

    if (kernel == nullptr) {
        return CL_SUCCESS;
    }

    const cl_int ret = pReleaseKernel(kernel);
    if (ret != CL_SUCCESS) {
        LOG_ERR(kErrorTemplate, OclError::toString(ret), kReleaseKernel);
    }

    return ret;
}


cl_int xmrig::OclLib::release(cl_mem mem_obj)
{
    assert(pReleaseMemObject != nullptr);

    if (mem_obj == nullptr) {
        return CL_SUCCESS;
    }

    const cl_int ret = pReleaseMemObject(mem_obj);
    if (ret != CL_SUCCESS) {
        LOG_ERR(kErrorTemplate, OclError::toString(ret), kReleaseMemObject);
    }

    return ret;
}


cl_int xmrig::OclLib::release(cl_program program)
{
    assert(pReleaseProgram != nullptr);

    if (program == nullptr) {
        return CL_SUCCESS;
    }

    const cl_int ret = pReleaseProgram(program);
    if (ret != CL_SUCCESS) {
        LOG_ERR(kErrorTemplate, OclError::toString(ret), kReleaseProgram);
    }

    return ret;
}


cl_int xmrig::OclLib::setKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value)
{
    assert(pSetKernelArg != nullptr);

    return pSetKernelArg(kernel, arg_index, arg_size, arg_value);
}


cl_kernel xmrig::OclLib::createKernel(cl_program program, const char *kernel_name, cl_int *errcode_ret)
{
    assert(pCreateKernel != nullptr);

    auto result = pCreateKernel(program, kernel_name, errcode_ret);
    if (*errcode_ret != CL_SUCCESS) {
        LOG_ERR(MAGENTA_BG_BOLD(WHITE_BOLD_S " ocl ") RED(" error ") RED_BOLD("%s") RED(" when calling ") RED_BOLD("clCreateKernel") RED(" for kernel ") RED_BOLD("%s"),
                OclError::toString(*errcode_ret), kernel_name);

        return nullptr;
    }

    return result;
}


cl_mem xmrig::OclLib::createBuffer(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr, cl_int *errcode_ret)
{
    assert(pCreateBuffer != nullptr);

    auto result = pCreateBuffer(context, flags, size, host_ptr, errcode_ret);
    if (*errcode_ret != CL_SUCCESS) {
        LOG_ERR(MAGENTA_BG_BOLD(WHITE_BOLD_S " ocl ") RED(" error ") RED_BOLD("%s") RED(" when calling ") RED_BOLD("%s") RED(" with buffer size ") RED_BOLD("%zu"),
                OclError::toString(*errcode_ret), kCreateBuffer, size);

        return nullptr;
    }

    return result;
}


cl_program xmrig::OclLib::createProgramWithBinary(cl_context context, cl_uint num_devices, const cl_device_id *device_list, const size_t *lengths, const unsigned char **binaries, cl_int *binary_status, cl_int *errcode_ret)
{
    assert(pCreateProgramWithBinary != nullptr);

    auto result = pCreateProgramWithBinary(context, num_devices, device_list, lengths, binaries, binary_status, errcode_ret);
    if (*errcode_ret != CL_SUCCESS) {
        LOG_ERR(kErrorTemplate, OclError::toString(*errcode_ret), kCreateProgramWithBinary);

        return nullptr;
    }

    return result;
}


cl_program xmrig::OclLib::createProgramWithSource(cl_context context, cl_uint count, const char **strings, const size_t *lengths, cl_int *errcode_ret)
{
    assert(pCreateProgramWithSource != nullptr);

    auto result = pCreateProgramWithSource(context, count, strings, lengths, errcode_ret);
    if (*errcode_ret != CL_SUCCESS) {
        LOG_ERR(kErrorTemplate, OclError::toString(*errcode_ret), kCreateProgramWithSource);

        return nullptr;
    }

    return result;
}


cl_uint xmrig::OclLib::getDeviceUint(cl_device_id id, cl_device_info param, cl_uint defaultValue)
{
    OclLib::getDeviceInfo(id, param, sizeof(cl_uint), &defaultValue);

    return defaultValue;
}


cl_uint xmrig::OclLib::getNumPlatforms()
{
    cl_uint count = 0;
    cl_int ret;

    if ((ret = OclLib::getPlatformIDs(0, nullptr, &count)) != CL_SUCCESS) {
        LOG_ERR("Error %s when calling clGetPlatformIDs for number of platforms.", OclError::toString(ret));
    }

    if (count == 0) {
        LOG_ERR("No OpenCL platform found.");
    }

    return count;
}


cl_uint xmrig::OclLib::getReferenceCount(cl_program program)
{
    cl_uint out = 0;
    OclLib::getProgramInfo(program, CL_PROGRAM_REFERENCE_COUNT, sizeof(cl_uint), &out);

    return out;
}


cl_ulong xmrig::OclLib::getDeviceUlong(cl_device_id id, cl_device_info param, cl_ulong defaultValue)
{
    OclLib::getDeviceInfo(id, param, sizeof(cl_ulong), &defaultValue);

    return defaultValue;
}


std::vector<cl_platform_id> xmrig::OclLib::getPlatformIDs()
{
    const uint32_t count = getNumPlatforms();
    std::vector<cl_platform_id> platforms(count);

    if (count) {
        OclLib::getPlatformIDs(count, platforms.data(), nullptr);
    }

    return platforms;
}


xmrig::String xmrig::OclLib::getDeviceString(cl_device_id id, cl_device_info param)
{
    size_t size = 0;
    if (getDeviceInfo(id, param, 0, nullptr, &size) != CL_SUCCESS) {
        return String();
    }

    char *buf = new char[size]();
    getDeviceInfo(id, param, size, buf, nullptr);

    return String(buf);
}


xmrig::String xmrig::OclLib::getPlatformInfo(cl_platform_id platform, cl_platform_info param_name)
{
    size_t size = 0;
    if (getPlatformInfo(platform, param_name, 0, nullptr, &size) != CL_SUCCESS) {
        return String();
    }

    char *buf = new char[size]();
    OclLib::getPlatformInfo(platform, param_name, size, buf, nullptr);

    return String(buf);
}


xmrig::String xmrig::OclLib::getProgramBuildLog(cl_program program, cl_device_id device)
{
    size_t size = 0;
    if (getProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size) != CL_SUCCESS) {
        return String();
    }

    char *log = new char[size + 1]();

    if (OclLib::getProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, size, log, nullptr) != CL_SUCCESS) {
        delete [] log;
        return String();
    }

    return log;
}
