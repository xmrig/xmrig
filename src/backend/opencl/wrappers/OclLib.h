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

#ifndef XMRIG_OCLLIB_H
#define XMRIG_OCLLIB_H


#include <vector>


#include "3rdparty/cl.h"
#include "base/tools/String.h"


namespace xmrig {


class OclLib
{
public:
    static bool init(const char *fileName = nullptr);
    static const char *lastError();
    static void close();

    static inline bool isInitialized()   { return m_initialized; }
    static inline const String &loader() { return m_loader; }

    static cl_command_queue createCommandQueue(cl_context context, cl_device_id device, cl_int *errcode_ret);
    static cl_context createContext(const cl_context_properties *properties, cl_uint num_devices, const cl_device_id *devices, void (CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *), void *user_data, cl_int *errcode_ret);
    static cl_context createContext(const std::vector<cl_device_id> &ids);
    static cl_int buildProgram(cl_program program, cl_uint num_devices, const cl_device_id *device_list, const char *options = nullptr, void (CL_CALLBACK *pfn_notify)(cl_program program, void *user_data) = nullptr, void *user_data = nullptr);
    static cl_int enqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t *global_work_offset, const size_t *global_work_size, const size_t *local_work_size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);
    static cl_int enqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, size_t offset, size_t size, void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);
    static cl_int enqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t offset, size_t size, const void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);
    static cl_int finish(cl_command_queue command_queue);
    static cl_int getDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id *devices, cl_uint *num_devices);
    static cl_int getDeviceInfo(cl_device_id device, cl_device_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret = nullptr);
    static cl_int getPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms);
    static cl_int getPlatformInfo(cl_platform_id platform, cl_platform_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
    static cl_int getProgramBuildInfo(cl_program program, cl_device_id device, cl_program_build_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
    static cl_int getProgramInfo(cl_program program, cl_program_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret = nullptr);
    static cl_int release(cl_command_queue command_queue);
    static cl_int release(cl_context context);
    static cl_int release(cl_kernel kernel);
    static cl_int release(cl_mem mem_obj);
    static cl_int release(cl_program program);
    static cl_int setKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value);
    static cl_kernel createKernel(cl_program program, const char *kernel_name, cl_int *errcode_ret);
    static cl_mem createBuffer(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr, cl_int *errcode_ret);
    static cl_program createProgramWithBinary(cl_context context, cl_uint num_devices, const cl_device_id *device_list, const size_t *lengths, const unsigned char **binaries, cl_int *binary_status, cl_int *errcode_ret);
    static cl_program createProgramWithSource(cl_context context, cl_uint count, const char **strings, const size_t *lengths, cl_int *errcode_ret);
    static cl_uint getDeviceUint(cl_device_id id, cl_device_info param, cl_uint defaultValue = 0);
    static cl_uint getNumPlatforms();
    static cl_uint getReferenceCount(cl_program program);
    static cl_ulong getDeviceUlong(cl_device_id id, cl_device_info param, cl_ulong defaultValue = 0);
    static std::vector<cl_platform_id> getPlatformIDs();
    static String getDeviceString(cl_device_id id, cl_device_info param);
    static String getPlatformInfo(cl_platform_id platform, cl_platform_info param_name);
    static String getProgramBuildLog(cl_program program, cl_device_id device);

private:
    static bool load();
    static const char *defaultLoader();

    static bool m_initialized;
    static bool m_ready;
    static String m_loader;
};


} // namespace xmrig


#endif /* XMRIG_OCLLIB_H */
