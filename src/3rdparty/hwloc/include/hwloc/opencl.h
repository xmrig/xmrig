/*
 * Copyright © 2012-2018 Inria.  All rights reserved.
 * Copyright © 2013, 2018 Université Bordeaux.  All right reserved.
 * See COPYING in top-level directory.
 */

/** \file
 * \brief Macros to help interaction between hwloc and the OpenCL interface.
 *
 * Applications that use both hwloc and OpenCL may want to
 * include this file so as to get topology information for OpenCL devices.
 */

#ifndef HWLOC_OPENCL_H
#define HWLOC_OPENCL_H

#include <hwloc.h>
#include <hwloc/autogen/config.h>
#include <hwloc/helper.h>
#ifdef HWLOC_LINUX_SYS
#include <hwloc/linux.h>
#endif

#ifdef __APPLE__
#include <OpenCL/cl.h>
#include <OpenCL/cl_ext.h>
#else
#include <CL/cl.h>
#include <CL/cl_ext.h>
#endif

#include <stdio.h>


#ifdef __cplusplus
extern "C" {
#endif


/** \defgroup hwlocality_opencl Interoperability with OpenCL
 *
 * This interface offers ways to retrieve topology information about
 * OpenCL devices.
 *
 * Only the AMD OpenCL interface currently offers useful locality information
 * about its devices.
 *
 * @{
 */

/** \brief Get the CPU set of logical processors that are physically
 * close to OpenCL device \p device.
 *
 * Return the CPU set describing the locality of the OpenCL device \p device.
 *
 * Topology \p topology and device \p device must match the local machine.
 * I/O devices detection and the OpenCL component are not needed in the topology.
 *
 * The function only returns the locality of the device.
 * If more information about the device is needed, OS objects should
 * be used instead, see hwloc_opencl_get_device_osdev()
 * and hwloc_opencl_get_device_osdev_by_index().
 *
 * This function is currently only implemented in a meaningful way for
 * Linux with the AMD OpenCL implementation; other systems will simply
 * get a full cpuset.
 */
static __hwloc_inline int
hwloc_opencl_get_device_cpuset(hwloc_topology_t topology __hwloc_attribute_unused,
			       cl_device_id device __hwloc_attribute_unused,
			       hwloc_cpuset_t set)
{
#if (defined HWLOC_LINUX_SYS) && (defined CL_DEVICE_TOPOLOGY_AMD)
	/* If we're on Linux + AMD OpenCL, use the AMD extension + the sysfs mechanism to get the local cpus */
#define HWLOC_OPENCL_DEVICE_SYSFS_PATH_MAX 128
	char path[HWLOC_OPENCL_DEVICE_SYSFS_PATH_MAX];
	cl_device_topology_amd amdtopo;
	cl_int clret;

	if (!hwloc_topology_is_thissystem(topology)) {
		errno = EINVAL;
		return -1;
	}

	clret = clGetDeviceInfo(device, CL_DEVICE_TOPOLOGY_AMD, sizeof(amdtopo), &amdtopo, NULL);
	if (CL_SUCCESS != clret) {
		hwloc_bitmap_copy(set, hwloc_topology_get_complete_cpuset(topology));
		return 0;
	}
	if (CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD != amdtopo.raw.type) {
		hwloc_bitmap_copy(set, hwloc_topology_get_complete_cpuset(topology));
		return 0;
	}

	sprintf(path, "/sys/bus/pci/devices/0000:%02x:%02x.%01x/local_cpus",
		(unsigned) amdtopo.pcie.bus, (unsigned) amdtopo.pcie.device, (unsigned) amdtopo.pcie.function);
	if (hwloc_linux_read_path_as_cpumask(path, set) < 0
	    || hwloc_bitmap_iszero(set))
		hwloc_bitmap_copy(set, hwloc_topology_get_complete_cpuset(topology));
#else
	/* Non-Linux + AMD OpenCL systems simply get a full cpuset */
	hwloc_bitmap_copy(set, hwloc_topology_get_complete_cpuset(topology));
#endif
  return 0;
}

/** \brief Get the hwloc OS device object corresponding to the
 * OpenCL device for the given indexes.
 *
 * Return the OS device object describing the OpenCL device
 * whose platform index is \p platform_index,
 * and whose device index within this platform if \p device_index.
 * Return NULL if there is none.
 *
 * The topology \p topology does not necessarily have to match the current
 * machine. For instance the topology may be an XML import of a remote host.
 * I/O devices detection and the OpenCL component must be enabled in the topology.
 *
 * \note The corresponding PCI device object can be obtained by looking
 * at the OS device parent object (unless PCI devices are filtered out).
 */
static __hwloc_inline hwloc_obj_t
hwloc_opencl_get_device_osdev_by_index(hwloc_topology_t topology,
				       unsigned platform_index, unsigned device_index)
{
	unsigned x = (unsigned) -1, y = (unsigned) -1;
	hwloc_obj_t osdev = NULL;
	while ((osdev = hwloc_get_next_osdev(topology, osdev)) != NULL) {
		if (HWLOC_OBJ_OSDEV_COPROC == osdev->attr->osdev.type
                    && osdev->name
		    && sscanf(osdev->name, "opencl%ud%u", &x, &y) == 2
		    && platform_index == x && device_index == y)
                        return osdev;
        }
        return NULL;
}

/** \brief Get the hwloc OS device object corresponding to OpenCL device \p deviceX.
 *
 * Use OpenCL device attributes to find the corresponding hwloc OS device object.
 * Return NULL if there is none or if useful attributes are not available.
 *
 * This function currently only works on AMD OpenCL devices that support
 * the CL_DEVICE_TOPOLOGY_AMD extension. hwloc_opencl_get_device_osdev_by_index()
 * should be preferred whenever possible, i.e. when platform and device index
 * are known.
 *
 * Topology \p topology and device \p device must match the local machine.
 * I/O devices detection and the OpenCL component must be enabled in the topology.
 * If not, the locality of the object may still be found using
 * hwloc_opencl_get_device_cpuset().
 *
 * \note This function cannot work if PCI devices are filtered out.
 *
 * \note The corresponding hwloc PCI device may be found by looking
 * at the result parent pointer (unless PCI devices are filtered out).
 */
static __hwloc_inline hwloc_obj_t
hwloc_opencl_get_device_osdev(hwloc_topology_t topology __hwloc_attribute_unused,
			      cl_device_id device __hwloc_attribute_unused)
{
#ifdef CL_DEVICE_TOPOLOGY_AMD
	hwloc_obj_t osdev;
	cl_device_topology_amd amdtopo;
	cl_int clret;

	clret = clGetDeviceInfo(device, CL_DEVICE_TOPOLOGY_AMD, sizeof(amdtopo), &amdtopo, NULL);
	if (CL_SUCCESS != clret) {
		errno = EINVAL;
		return NULL;
	}
	if (CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD != amdtopo.raw.type) {
		errno = EINVAL;
		return NULL;
	}

	osdev = NULL;
	while ((osdev = hwloc_get_next_osdev(topology, osdev)) != NULL) {
		hwloc_obj_t pcidev = osdev->parent;
		if (strncmp(osdev->name, "opencl", 6))
			continue;
		if (pcidev
		    && pcidev->type == HWLOC_OBJ_PCI_DEVICE
		    && pcidev->attr->pcidev.domain == 0
		    && pcidev->attr->pcidev.bus == amdtopo.pcie.bus
		    && pcidev->attr->pcidev.dev == amdtopo.pcie.device
		    && pcidev->attr->pcidev.func == amdtopo.pcie.function)
			return osdev;
		/* if PCI are filtered out, we need a info attr to match on */
	}

	return NULL;
#else
	return NULL;
#endif
}

/** @} */


#ifdef __cplusplus
} /* extern "C" */
#endif


#endif /* HWLOC_OPENCL_H */
