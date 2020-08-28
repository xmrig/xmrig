/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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
#include <stdexcept>
#include <uv.h>


#include "backend/opencl/wrappers/OclLib.h"
#include "backend/common/Tags.h"
#include "backend/opencl/wrappers/OclError.h"
#include "base/io/Env.h"
#include "base/io/log/Log.h"


#if defined(OCL_DEBUG_REFERENCE_COUNT)
#   define LOG_REFS(x, ...) xmrig::Log::print(xmrig::Log::WARNING, x, ##__VA_ARGS__)
#else
#   define LOG_REFS(x, ...)
#endif


static uv_lib_t oclLib;

static const char *kErrorTemplate                    = MAGENTA_BG_BOLD(WHITE_BOLD_S " opencl  ") RED(" error ") RED_BOLD("%s") RED(" when calling ") RED_BOLD("%s");

static const char *kBuildProgram                     = "clBuildProgram";
static const char *kCreateBuffer                     = "clCreateBuffer";
static const char *kCreateCommandQueue               = "clCreateCommandQueue";
static const char *kCreateCommandQueueWithProperties = "clCreateCommandQueueWithProperties";
static const char *kCreateContext                    = "clCreateContext";
static const char *kCreateKernel                     = "clCreateKernel";
static const char *kCreateProgramWithBinary          = "clCreateProgramWithBinary";
static const char *kCreateProgramWithSource          = "clCreateProgramWithSource";
static const char *kCreateSubBuffer                  = "clCreateSubBuffer";
static const char *kEnqueueNDRangeKernel             = "clEnqueueNDRangeKernel";
static const char *kEnqueueReadBuffer                = "clEnqueueReadBuffer";
static const char *kEnqueueWriteBuffer               = "clEnqueueWriteBuffer";
static const char *kFinish                           = "clFinish";
static const char *kGetCommandQueueInfo              = "clGetCommandQueueInfo";
static const char *kGetContextInfo                   = "clGetContextInfo";
static const char *kGetDeviceIDs                     = "clGetDeviceIDs";
static const char *kGetDeviceInfo                    = "clGetDeviceInfo";
static const char *kGetKernelInfo                    = "clGetKernelInfo";
static const char *kGetMemObjectInfo                 = "clGetMemObjectInfo";
static const char *kGetPlatformIDs                   = "clGetPlatformIDs";
static const char *kGetPlatformInfo                  = "clGetPlatformInfo";
static const char *kGetProgramBuildInfo              = "clGetProgramBuildInfo";
static const char *kGetProgramInfo                   = "clGetProgramInfo";
static const char *kReleaseCommandQueue              = "clReleaseCommandQueue";
static const char *kReleaseContext                   = "clReleaseContext";
static const char *kReleaseDevice                    = "clReleaseDevice";
static const char *kReleaseKernel                    = "clReleaseKernel";
static const char *kReleaseMemObject                 = "clReleaseMemObject";
static const char *kReleaseProgram                   = "clReleaseProgram";
static const char *kRetainMemObject                  = "clRetainMemObject";
static const char *kRetainProgram                    = "clRetainProgram";
static const char *kSetKernelArg                     = "clSetKernelArg";
static const char *kSetMemObjectDestructorCallback   = "clSetMemObjectDestructorCallback";
static const char *kSymbolNotFound                   = "symbol not found";
static const char *kUnloadPlatformCompiler           = "clUnloadPlatformCompiler";


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
typedef cl_int (CL_API_CALL *getContextInfo_t)(cl_context, cl_context_info, size_t, void *, size_t *);
typedef cl_int (CL_API_CALL *getDeviceIDs_t)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
typedef cl_int (CL_API_CALL *getDeviceInfo_t)(cl_device_id, cl_device_info, size_t, void *, size_t *);
typedef cl_int (CL_API_CALL *getKernelInfo_t)(cl_kernel, cl_kernel_info, size_t, void *, size_t *);
typedef cl_int (CL_API_CALL *getMemObjectInfo_t)(cl_mem, cl_mem_info, size_t, void *, size_t *);
typedef cl_int (CL_API_CALL *getPlatformIDs_t)(cl_uint, cl_platform_id *, cl_uint *);
typedef cl_int (CL_API_CALL *getPlatformInfo_t)(cl_platform_id, cl_platform_info, size_t, void *, size_t *);
typedef cl_int (CL_API_CALL *getProgramBuildInfo_t)(cl_program, cl_device_id, cl_program_build_info, size_t, void *, size_t *);
typedef cl_int (CL_API_CALL *getProgramInfo_t)(cl_program, cl_program_info, size_t, void *, size_t *);
typedef cl_int (CL_API_CALL *releaseCommandQueue_t)(cl_command_queue);
typedef cl_int (CL_API_CALL *releaseContext_t)(cl_context);
typedef cl_int (CL_API_CALL *releaseDevice_t)(cl_device_id device);
typedef cl_int (CL_API_CALL *releaseKernel_t)(cl_kernel);
typedef cl_int (CL_API_CALL *releaseMemObject_t)(cl_mem);
typedef cl_int (CL_API_CALL *releaseProgram_t)(cl_program);
typedef cl_int (CL_API_CALL *retainMemObject_t)(cl_mem);
typedef cl_int (CL_API_CALL *retainProgram_t)(cl_program);
typedef cl_int (CL_API_CALL *setKernelArg_t)(cl_kernel, cl_uint, size_t, const void *);
typedef cl_int (CL_API_CALL *setMemObjectDestructorCallback_t)(cl_mem, void (CL_CALLBACK *)(cl_mem, void *), void *);
typedef cl_int (CL_API_CALL *unloadPlatformCompiler_t)(cl_platform_id);
typedef cl_kernel (CL_API_CALL *createKernel_t)(cl_program, const char *, cl_int *);
typedef cl_mem (CL_API_CALL *createBuffer_t)(cl_context, cl_mem_flags, size_t, void *, cl_int *);
typedef cl_mem (CL_API_CALL *createSubBuffer_t)(cl_mem, cl_mem_flags, cl_buffer_create_type, const void *, cl_int *);
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
static createSubBuffer_t pCreateSubBuffer                                   = nullptr;
static enqueueNDRangeKernel_t pEnqueueNDRangeKernel                         = nullptr;
static enqueueReadBuffer_t pEnqueueReadBuffer                               = nullptr;
static enqueueWriteBuffer_t pEnqueueWriteBuffer                             = nullptr;
static finish_t pFinish                                                     = nullptr;
static getCommandQueueInfo_t pGetCommandQueueInfo                           = nullptr;
static getContextInfo_t pGetContextInfo                                     = nullptr;
static getDeviceIDs_t pGetDeviceIDs                                         = nullptr;
static getDeviceInfo_t pGetDeviceInfo                                       = nullptr;
static getKernelInfo_t pGetKernelInfo                                       = nullptr;
static getMemObjectInfo_t pGetMemObjectInfo                                 = nullptr;
static getPlatformIDs_t pGetPlatformIDs                                     = nullptr;
static getPlatformInfo_t pGetPlatformInfo                                   = nullptr;
static getProgramBuildInfo_t pGetProgramBuildInfo                           = nullptr;
static getProgramInfo_t pGetProgramInfo                                     = nullptr;
static releaseCommandQueue_t pReleaseCommandQueue                           = nullptr;
static releaseContext_t pReleaseContext                                     = nullptr;
static releaseDevice_t pReleaseDevice                                       = nullptr;
static releaseKernel_t pReleaseKernel                                       = nullptr;
static releaseMemObject_t pReleaseMemObject                                 = nullptr;
static releaseProgram_t pReleaseProgram                                     = nullptr;
static retainMemObject_t pRetainMemObject                                   = nullptr;
static retainProgram_t pRetainProgram                                       = nullptr;
static setKernelArg_t pSetKernelArg                                         = nullptr;
static setMemObjectDestructorCallback_t pSetMemObjectDestructorCallback     = nullptr;
static unloadPlatformCompiler_t pUnloadPlatformCompiler                     = nullptr;

#define DLSYM(x) if (uv_dlsym(&oclLib, k##x, reinterpret_cast<void**>(&p##x)) == -1) { throw std::runtime_error(kSymbolNotFound); }


namespace xmrig {

bool OclLib::m_initialized = false;
bool OclLib::m_ready       = false;
String OclLib::m_loader;


template<typename FUNC, typename OBJ, typename PARAM>
static String getOclString(FUNC fn, OBJ obj, PARAM param)
{
    size_t size = 0;
    if (fn(obj, param, 0, nullptr, &size) != CL_SUCCESS) {
        return String();
    }

    char *buf = new char[size]();
    fn(obj, param, size, buf, nullptr);

    return String(buf);
}


} // namespace xmrig


bool xmrig::OclLib::init(const char *fileName)
{
    if (!m_initialized) {
        m_loader      = fileName == nullptr ? defaultLoader() : Env::expand(fileName);
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
    try {
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
        DLSYM(GetMemObjectInfo);
        DLSYM(GetContextInfo);
        DLSYM(ReleaseDevice);
        DLSYM(UnloadPlatformCompiler);
        DLSYM(SetMemObjectDestructorCallback);
        DLSYM(CreateSubBuffer);
        DLSYM(RetainProgram);
        DLSYM(RetainMemObject);
    } catch (std::exception &ex) {
        return false;
    }

#   if defined(CL_VERSION_2_0)
    uv_dlsym(&oclLib, kCreateCommandQueueWithProperties, reinterpret_cast<void**>(&pCreateCommandQueueWithProperties));
#   endif

    return true;
}


xmrig::String xmrig::OclLib::defaultLoader()
{
#   if defined(__APPLE__)
    return "/System/Library/Frameworks/OpenCL.framework/OpenCL";
#   elif defined(_WIN32)
    return "OpenCL.dll";
#   else
    return "libOpenCL.so";
#   endif
}


cl_command_queue xmrig::OclLib::createCommandQueue(cl_context context, cl_device_id device, cl_int *errcode_ret) noexcept
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


cl_command_queue xmrig::OclLib::createCommandQueue(cl_context context, cl_device_id device)
{
    cl_int ret;
    cl_command_queue queue = createCommandQueue(context, device, &ret);
    if (ret != CL_SUCCESS) {
        throw std::runtime_error(OclError::toString(ret));
    }

    return queue;
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
    return createContext(nullptr, static_cast<cl_uint>(ids.size()), ids.data(), nullptr, nullptr, &ret);
}


cl_int xmrig::OclLib::buildProgram(cl_program program, cl_uint num_devices, const cl_device_id *device_list, const char *options, void (CL_CALLBACK *pfn_notify)(cl_program program, void *user_data), void *user_data) noexcept
{
    assert(pBuildProgram != nullptr);

    const cl_int ret = pBuildProgram(program, num_devices, device_list, options, pfn_notify, user_data);
    if (ret != CL_SUCCESS) {
        LOG_ERR(kErrorTemplate, OclError::toString(ret), kBuildProgram);
    }

    return ret;
}


cl_int xmrig::OclLib::enqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t *global_work_offset, const size_t *global_work_size, const size_t *local_work_size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) noexcept
{
    assert(pEnqueueNDRangeKernel != nullptr);

    return pEnqueueNDRangeKernel(command_queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size, num_events_in_wait_list, event_wait_list, event);
}


cl_int xmrig::OclLib::enqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, size_t offset, size_t size, void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) noexcept
{
    assert(pEnqueueReadBuffer != nullptr);

    const cl_int ret = pEnqueueReadBuffer(command_queue, buffer, blocking_read, offset, size, ptr, num_events_in_wait_list, event_wait_list, event);
    if (ret != CL_SUCCESS) {
        LOG_ERR(kErrorTemplate, OclError::toString(ret), kEnqueueReadBuffer);
    }

    return ret;
}


cl_int xmrig::OclLib::enqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t offset, size_t size, const void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) noexcept
{
    assert(pEnqueueWriteBuffer != nullptr);

    const cl_int ret = pEnqueueWriteBuffer(command_queue, buffer, blocking_write, offset, size, ptr, num_events_in_wait_list, event_wait_list, event);
    if (ret != CL_SUCCESS) {
        LOG_ERR(kErrorTemplate, OclError::toString(ret), kEnqueueWriteBuffer);
    }

    return ret;
}


cl_int xmrig::OclLib::finish(cl_command_queue command_queue) noexcept
{
    assert(pFinish != nullptr);

    return pFinish(command_queue);
}


cl_int xmrig::OclLib::getCommandQueueInfo(cl_command_queue command_queue, cl_command_queue_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret) noexcept
{
    return pGetCommandQueueInfo(command_queue, param_name, param_value_size, param_value, param_value_size_ret);
}


cl_int xmrig::OclLib::getContextInfo(cl_context context, cl_context_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret) noexcept
{
    return pGetContextInfo(context, param_name, param_value_size, param_value, param_value_size_ret);
}


cl_int xmrig::OclLib::getDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id *devices, cl_uint *num_devices) noexcept
{
    assert(pGetDeviceIDs != nullptr);

    return pGetDeviceIDs(platform, device_type, num_entries, devices, num_devices);
}


cl_int xmrig::OclLib::getDeviceInfo(cl_device_id device, cl_device_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret) noexcept
{
    assert(pGetDeviceInfo != nullptr);

    const cl_int ret = pGetDeviceInfo(device, param_name, param_value_size, param_value, param_value_size_ret);
    if (ret != CL_SUCCESS && param_name != 0x4038) {
        LOG_ERR("Error %s when calling %s, param 0x%04x", OclError::toString(ret), kGetDeviceInfo, param_name);
    }

    return ret;
}


cl_int xmrig::OclLib::getKernelInfo(cl_kernel kernel, cl_kernel_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret) noexcept
{
    return pGetKernelInfo(kernel, param_name, param_value_size, param_value, param_value_size_ret);
}


cl_int xmrig::OclLib::getMemObjectInfo(cl_mem memobj, cl_mem_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret) noexcept
{
    return pGetMemObjectInfo(memobj, param_name, param_value_size, param_value, param_value_size_ret);
}


cl_int xmrig::OclLib::getPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms)
{
    assert(pGetPlatformIDs != nullptr);

    return pGetPlatformIDs(num_entries, platforms, num_platforms);
}


cl_int xmrig::OclLib::getPlatformInfo(cl_platform_id platform, cl_platform_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret) noexcept
{
    assert(pGetPlatformInfo != nullptr);

    return pGetPlatformInfo(platform, param_name, param_value_size, param_value, param_value_size_ret);
}


cl_int xmrig::OclLib::getProgramBuildInfo(cl_program program, cl_device_id device, cl_program_build_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret) noexcept
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


cl_int xmrig::OclLib::release(cl_command_queue command_queue) noexcept
{
    assert(pReleaseCommandQueue != nullptr);
    assert(pGetCommandQueueInfo != nullptr);

    if (command_queue == nullptr) {
        return CL_SUCCESS;
    }

    LOG_REFS("%p %u ~queue", command_queue, getUint(command_queue, CL_QUEUE_REFERENCE_COUNT));

    finish(command_queue);

    cl_int ret = pReleaseCommandQueue(command_queue);
    if (ret != CL_SUCCESS) {
        LOG_ERR(kErrorTemplate, OclError::toString(ret), kReleaseCommandQueue);
    }

    return ret;
}


cl_int xmrig::OclLib::release(cl_context context) noexcept
{
    assert(pReleaseContext != nullptr);

    LOG_REFS("%p %u ~context", context, getUint(context, CL_CONTEXT_REFERENCE_COUNT));

    const cl_int ret = pReleaseContext(context);
    if (ret != CL_SUCCESS) {
        LOG_ERR(kErrorTemplate, OclError::toString(ret), kReleaseContext);
    }

    return ret;
}


cl_int xmrig::OclLib::release(cl_device_id id) noexcept
{
    assert(pReleaseDevice != nullptr);

    LOG_REFS("%p %u ~device", id, getUint(id, CL_DEVICE_REFERENCE_COUNT));

    const cl_int ret = pReleaseDevice(id);
    if (ret != CL_SUCCESS) {
        LOG_ERR(kErrorTemplate, OclError::toString(ret), kReleaseDevice);
    }

    return ret;
}


cl_int xmrig::OclLib::release(cl_kernel kernel) noexcept
{
    assert(pReleaseKernel != nullptr);

    if (kernel == nullptr) {
        return CL_SUCCESS;
    }

    LOG_REFS("%p %u ~kernel %s", kernel, getUint(kernel, CL_KERNEL_REFERENCE_COUNT), getString(kernel, CL_KERNEL_FUNCTION_NAME).data());

    const cl_int ret = pReleaseKernel(kernel);
    if (ret != CL_SUCCESS) {
        LOG_ERR(kErrorTemplate, OclError::toString(ret), kReleaseKernel);
    }

    return ret;
}


cl_int xmrig::OclLib::release(cl_mem mem_obj) noexcept
{
    assert(pReleaseMemObject != nullptr);

    if (mem_obj == nullptr) {
        return CL_SUCCESS;
    }

    LOG_REFS("%p %u ~mem %zub", mem_obj, getUint(mem_obj, CL_MEM_REFERENCE_COUNT), getUlong(mem_obj, CL_MEM_SIZE));

    const cl_int ret = pReleaseMemObject(mem_obj);
    if (ret != CL_SUCCESS) {
        LOG_ERR(kErrorTemplate, OclError::toString(ret), kReleaseMemObject);
    }

    return ret;
}


cl_int xmrig::OclLib::release(cl_program program) noexcept
{
    assert(pReleaseProgram != nullptr);

    if (program == nullptr) {
        return CL_SUCCESS;
    }

    LOG_REFS("%p %u ~program %s", program, getUint(program, CL_PROGRAM_REFERENCE_COUNT), getString(program, CL_PROGRAM_KERNEL_NAMES).data());

    const cl_int ret = pReleaseProgram(program);
    if (ret != CL_SUCCESS) {
        LOG_ERR(kErrorTemplate, OclError::toString(ret), kReleaseProgram);
    }

    return ret;
}


cl_int xmrig::OclLib::setKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value) noexcept
{
    assert(pSetKernelArg != nullptr);

    return pSetKernelArg(kernel, arg_index, arg_size, arg_value);
}


cl_int xmrig::OclLib::unloadPlatformCompiler(cl_platform_id platform) noexcept
{
    return pUnloadPlatformCompiler(platform);
}


cl_kernel xmrig::OclLib::createKernel(cl_program program, const char *kernel_name, cl_int *errcode_ret) noexcept
{
    assert(pCreateKernel != nullptr);

    auto result = pCreateKernel(program, kernel_name, errcode_ret);
    if (*errcode_ret != CL_SUCCESS) {
        LOG_ERR("%s" RED(" error ") RED_BOLD("%s") RED(" when calling ") RED_BOLD("clCreateKernel") RED(" for kernel ") RED_BOLD("%s"),
                ocl_tag(), OclError::toString(*errcode_ret), kernel_name);

        return nullptr;
    }

    return result;
}


cl_kernel xmrig::OclLib::createKernel(cl_program program, const char *kernel_name)
{
    cl_int ret;
    cl_kernel kernel = createKernel(program, kernel_name, &ret);
    if (ret != CL_SUCCESS) {
        throw std::runtime_error(OclError::toString(ret));
    }

    return kernel;
}


cl_mem xmrig::OclLib::createBuffer(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr)
{
    cl_int ret;
    cl_mem mem = createBuffer(context, flags, size, host_ptr, &ret);
    if (ret != CL_SUCCESS) {
        throw std::runtime_error(OclError::toString(ret));
    }

    return mem;
}


cl_mem xmrig::OclLib::createBuffer(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr, cl_int *errcode_ret) noexcept
{
    assert(pCreateBuffer != nullptr);

    auto result = pCreateBuffer(context, flags, size, host_ptr, errcode_ret);
    if (*errcode_ret != CL_SUCCESS) {
        LOG_ERR("%s" RED(" error ") RED_BOLD("%s") RED(" when calling ") RED_BOLD("%s") RED(" with buffer size ") RED_BOLD("%zu"),
                ocl_tag(), OclError::toString(*errcode_ret), kCreateBuffer, size);

        return nullptr;
    }

    return result;
}


cl_mem xmrig::OclLib::createSubBuffer(cl_mem buffer, cl_mem_flags flags, size_t offset, size_t size, cl_int *errcode_ret) noexcept
{
    const cl_buffer_region region = { offset, size };

    auto result = pCreateSubBuffer(buffer, flags, CL_BUFFER_CREATE_TYPE_REGION, &region, errcode_ret);
    if (*errcode_ret != CL_SUCCESS) {
        LOG_ERR("%s" RED(" error ") RED_BOLD("%s") RED(" when calling ") RED_BOLD("%s") RED(" with offset ") RED_BOLD("%zu") RED(" and size ") RED_BOLD("%zu"),
                ocl_tag(), OclError::toString(*errcode_ret), kCreateSubBuffer, offset, size);

        return nullptr;
    }

    return result;
}


cl_mem xmrig::OclLib::createSubBuffer(cl_mem buffer, cl_mem_flags flags, size_t offset, size_t size)
{
    cl_int ret;
    cl_mem mem = createSubBuffer(buffer, flags, offset, size, &ret);
    if (ret != CL_SUCCESS) {
        throw std::runtime_error(OclError::toString(ret));
    }

    return mem;
}


cl_mem xmrig::OclLib::retain(cl_mem memobj) noexcept
{
    assert(pRetainMemObject != nullptr);

    if (memobj != nullptr) {
        pRetainMemObject(memobj);
    }

    return memobj;
}


cl_program xmrig::OclLib::createProgramWithBinary(cl_context context, cl_uint num_devices, const cl_device_id *device_list, const size_t *lengths, const unsigned char **binaries, cl_int *binary_status, cl_int *errcode_ret) noexcept
{
    assert(pCreateProgramWithBinary != nullptr);

    auto result = pCreateProgramWithBinary(context, num_devices, device_list, lengths, binaries, binary_status, errcode_ret);
    if (*errcode_ret != CL_SUCCESS) {
        LOG_ERR(kErrorTemplate, OclError::toString(*errcode_ret), kCreateProgramWithBinary);

        return nullptr;
    }

    return result;
}


cl_program xmrig::OclLib::createProgramWithSource(cl_context context, cl_uint count, const char **strings, const size_t *lengths, cl_int *errcode_ret) noexcept
{
    assert(pCreateProgramWithSource != nullptr);

    auto result = pCreateProgramWithSource(context, count, strings, lengths, errcode_ret);
    if (*errcode_ret != CL_SUCCESS) {
        LOG_ERR(kErrorTemplate, OclError::toString(*errcode_ret), kCreateProgramWithSource);

        return nullptr;
    }

    return result;
}


cl_program xmrig::OclLib::retain(cl_program program) noexcept
{
    assert(pRetainProgram != nullptr);

    if (program != nullptr) {
        pRetainProgram(program);
    }

    return program;
}


cl_uint xmrig::OclLib::getNumPlatforms() noexcept
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


cl_uint xmrig::OclLib::getUint(cl_command_queue command_queue, cl_command_queue_info param_name, cl_uint defaultValue) noexcept
{
    getCommandQueueInfo(command_queue, param_name, sizeof(cl_uint), &defaultValue);

    return defaultValue;
}


cl_uint xmrig::OclLib::getUint(cl_context context, cl_context_info param_name, cl_uint defaultValue) noexcept
{
    getContextInfo(context, param_name, sizeof(cl_uint), &defaultValue);

    return defaultValue;
}


cl_uint xmrig::OclLib::getUint(cl_device_id id, cl_device_info param, cl_uint defaultValue) noexcept
{
    getDeviceInfo(id, param, sizeof(cl_uint), &defaultValue);

    return defaultValue;
}


cl_uint xmrig::OclLib::getUint(cl_kernel kernel, cl_kernel_info  param_name, cl_uint defaultValue) noexcept
{
    getKernelInfo(kernel, param_name, sizeof(cl_uint), &defaultValue);

    return defaultValue;
}


cl_uint xmrig::OclLib::getUint(cl_mem memobj, cl_mem_info param_name, cl_uint defaultValue) noexcept
{
    getMemObjectInfo(memobj, param_name, sizeof(cl_uint), &defaultValue);

    return defaultValue;
}


cl_uint xmrig::OclLib::getUint(cl_program program, cl_program_info param, cl_uint defaultValue) noexcept
{
    getProgramInfo(program, param, sizeof(cl_uint), &defaultValue);

    return defaultValue;
}


cl_ulong xmrig::OclLib::getUlong(cl_device_id id, cl_device_info param, cl_ulong defaultValue) noexcept
{
    getDeviceInfo(id, param, sizeof(cl_ulong), &defaultValue);

    return defaultValue;
}


cl_ulong xmrig::OclLib::getUlong(cl_mem memobj, cl_mem_info param_name, cl_ulong defaultValue) noexcept
{
    getMemObjectInfo(memobj, param_name, sizeof(cl_ulong), &defaultValue);

    return defaultValue;
}


std::vector<cl_platform_id> xmrig::OclLib::getPlatformIDs() noexcept
{
    const uint32_t count = getNumPlatforms();
    std::vector<cl_platform_id> platforms(count);

    if (count) {
        getPlatformIDs(count, platforms.data(), nullptr);
    }

    return platforms;
}


xmrig::String xmrig::OclLib::getProgramBuildLog(cl_program program, cl_device_id device) noexcept
{
    size_t size = 0;
    if (getProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size) != CL_SUCCESS) {
        return String();
    }

    char *log = new char[size + 1]();

    if (getProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, size, log, nullptr) != CL_SUCCESS) {
        delete [] log;
        return String();
    }

    return log;
}


xmrig::String xmrig::OclLib::getString(cl_device_id id, cl_device_info param) noexcept
{
    return getOclString(OclLib::getDeviceInfo, id, param);
}


xmrig::String xmrig::OclLib::getString(cl_kernel kernel, cl_kernel_info param_name) noexcept
{
    return getOclString(OclLib::getKernelInfo, kernel, param_name);
}


xmrig::String xmrig::OclLib::getString(cl_platform_id platform, cl_platform_info param_name) noexcept
{
    return getOclString(OclLib::getPlatformInfo, platform, param_name);
}


xmrig::String xmrig::OclLib::getString(cl_program program, cl_program_info param_name) noexcept
{
    return getOclString(OclLib::getProgramInfo, program, param_name);
}
