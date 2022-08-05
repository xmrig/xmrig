/*
 * Copyright © 2012-2021 Inria.  All rights reserved.
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

#include "hwloc.h"
#include "hwloc/autogen/config.h"
#include "hwloc/helper.h"
#ifdef HWLOC_LINUX_SYS
#include "hwloc/linux.h"
#endif

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <stdio.h>


#ifdef __cplusplus
extern "C" {
#endif


/* OpenCL extensions aren't always shipped with default headers, and
 * they don't always reflect what the installed implementations support.
 * Try everything and let the implementation return errors when non supported.
 */
/* Copyright (c) 2008-2018 The Khronos Group Inc. */

/* needs "cl_amd_device_attribute_query" device extension, but not strictly required for clGetDeviceInfo() */
#define HWLOC_CL_DEVICE_TOPOLOGY_AMD 0x4037
typedef union {
    struct { cl_uint type; cl_uint data[5]; } raw;
    struct { cl_uint type; cl_char unused[17]; cl_char bus; cl_char device; cl_char function; } pcie;
} hwloc_cl_device_topology_amd;
#define HWLOC_CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD 1

/* needs "cl_nv_device_attribute_query" device extension, but not strictly required for clGetDeviceInfo() */
#define HWLOC_CL_DEVICE_PCI_BUS_ID_NV 0x4008
#define HWLOC_CL_DEVICE_PCI_SLOT_ID_NV 0x4009
#define HWLOC_CL_DEVICE_PCI_DOMAIN_ID_NV 0x400A


/** \defgroup hwlocality_opencl Interoperability with OpenCL
 *
 * This interface offers ways to retrieve topology information about
 * OpenCL devices.
 *
 * Only AMD and NVIDIA OpenCL implementations currently offer useful locality
 * information about their devices.
 *
 * @{
 */

/** \brief Return the domain, bus and device IDs of the OpenCL device \p device.
 *
 * Device \p device must match the local machine.
 */
static __hwloc_inline int
hwloc_opencl_get_device_pci_busid(cl_device_id device,
                               unsigned *domain, unsigned *bus, unsigned *dev, unsigned *func)
{
	hwloc_cl_device_topology_amd amdtopo;
	cl_uint nvbus, nvslot, nvdomain;
	cl_int clret;

	clret = clGetDeviceInfo(device, HWLOC_CL_DEVICE_TOPOLOGY_AMD, sizeof(amdtopo), &amdtopo, NULL);
	if (CL_SUCCESS == clret
	    && HWLOC_CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD == amdtopo.raw.type) {
		*domain = 0; /* can't do anything better */
		/* cl_device_topology_amd stores bus ID in cl_char, dont convert those signed char directly to unsigned int */
		*bus = (unsigned) (unsigned char) amdtopo.pcie.bus;
		*dev = (unsigned) (unsigned char) amdtopo.pcie.device;
		*func = (unsigned) (unsigned char) amdtopo.pcie.function;
		return 0;
	}

	clret = clGetDeviceInfo(device, HWLOC_CL_DEVICE_PCI_BUS_ID_NV, sizeof(nvbus), &nvbus, NULL);
	if (CL_SUCCESS == clret) {
		clret = clGetDeviceInfo(device, HWLOC_CL_DEVICE_PCI_SLOT_ID_NV, sizeof(nvslot), &nvslot, NULL);
		if (CL_SUCCESS == clret) {
			clret = clGetDeviceInfo(device, HWLOC_CL_DEVICE_PCI_DOMAIN_ID_NV, sizeof(nvdomain), &nvdomain, NULL);
			if (CL_SUCCESS == clret) { /* available since CUDA 10.2 */
				*domain = nvdomain;
			} else {
				*domain = 0;
			}
			*bus = nvbus & 0xff;
			/* non-documented but used in many other projects */
			*dev = nvslot >> 3;
			*func = nvslot & 0x7;
			return 0;
		}
	}

	return -1;
}

/** \brief Get the CPU set of processors that are physically
 * close to OpenCL device \p device.
 *
 * Store in \p set the CPU-set describing the locality of the OpenCL device \p device.
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
 * Linux with the AMD or NVIDIA OpenCL implementation; other systems will simply
 * get a full cpuset.
 */
static __hwloc_inline int
hwloc_opencl_get_device_cpuset(hwloc_topology_t topology __hwloc_attribute_unused,
			       cl_device_id device __hwloc_attribute_unused,
			       hwloc_cpuset_t set)
{
#if (defined HWLOC_LINUX_SYS)
	/* If we're on Linux, try AMD/NVIDIA extensions + the sysfs mechanism to get the local cpus */
#define HWLOC_OPENCL_DEVICE_SYSFS_PATH_MAX 128
	char path[HWLOC_OPENCL_DEVICE_SYSFS_PATH_MAX];
	unsigned pcidomain, pcibus, pcidev, pcifunc;

	if (!hwloc_topology_is_thissystem(topology)) {
		errno = EINVAL;
		return -1;
	}

	if (hwloc_opencl_get_device_pci_busid(device, &pcidomain, &pcibus, &pcidev, &pcifunc) < 0) {
		hwloc_bitmap_copy(set, hwloc_topology_get_complete_cpuset(topology));
		return 0;
	}

	sprintf(path, "/sys/bus/pci/devices/%04x:%02x:%02x.%01x/local_cpus", pcidomain, pcibus, pcidev, pcifunc);
	if (hwloc_linux_read_path_as_cpumask(path, set) < 0
	    || hwloc_bitmap_iszero(set))
		hwloc_bitmap_copy(set, hwloc_topology_get_complete_cpuset(topology));
#else
	/* Non-Linux systems simply get a full cpuset */
	hwloc_bitmap_copy(set, hwloc_topology_get_complete_cpuset(topology));
#endif
  return 0;
}

/** \brief Get the hwloc OS device object corresponding to the
 * OpenCL device for the given indexes.
 *
 * \return The hwloc OS device object describing the OpenCL device
 * whose platform index is \p platform_index,
 * and whose device index within this platform if \p device_index.
 * \return \c NULL if there is none.
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
 * \return The hwloc OS device object corresponding to the given OpenCL device \p device.
 * \return \c NULL if none could be found, for instance
 * if required OpenCL attributes are not available.
 *
 * This function currently only works on AMD and NVIDIA OpenCL devices that support
 * relevant OpenCL extensions. hwloc_opencl_get_device_osdev_by_index()
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
	hwloc_obj_t osdev;
	unsigned pcidomain, pcibus, pcidevice, pcifunc;

	if (hwloc_opencl_get_device_pci_busid(device, &pcidomain, &pcibus, &pcidevice, &pcifunc) < 0) {
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
		    && pcidev->attr->pcidev.domain == pcidomain
		    && pcidev->attr->pcidev.bus == pcibus
		    && pcidev->attr->pcidev.dev == pcidevice
		    && pcidev->attr->pcidev.func == pcifunc)
			return osdev;
		/* if PCI are filtered out, we need a info attr to match on */
	}

	return NULL;
}

/** @} */


#ifdef __cplusplus
} /* extern "C" */
#endif


#endif /* HWLOC_OPENCL_H */
