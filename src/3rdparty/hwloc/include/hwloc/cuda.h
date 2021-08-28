/*
 * Copyright © 2010-2021 Inria.  All rights reserved.
 * Copyright © 2010-2011 Université Bordeaux
 * Copyright © 2011 Cisco Systems, Inc.  All rights reserved.
 * See COPYING in top-level directory.
 */

/** \file
 * \brief Macros to help interaction between hwloc and the CUDA Driver API.
 *
 * Applications that use both hwloc and the CUDA Driver API may want to
 * include this file so as to get topology information for CUDA devices.
 *
 */

#ifndef HWLOC_CUDA_H
#define HWLOC_CUDA_H

#include "hwloc.h"
#include "hwloc/autogen/config.h"
#include "hwloc/helper.h"
#ifdef HWLOC_LINUX_SYS
#include "hwloc/linux.h"
#endif

#include <cuda.h>


#ifdef __cplusplus
extern "C" {
#endif


/** \defgroup hwlocality_cuda Interoperability with the CUDA Driver API
 *
 * This interface offers ways to retrieve topology information about
 * CUDA devices when using the CUDA Driver API.
 *
 * @{
 */

/** \brief Return the domain, bus and device IDs of the CUDA device \p cudevice.
 *
 * Device \p cudevice must match the local machine.
 */
static __hwloc_inline int
hwloc_cuda_get_device_pci_ids(hwloc_topology_t topology __hwloc_attribute_unused,
			      CUdevice cudevice, int *domain, int *bus, int *dev)
{
  CUresult cres;

#if CUDA_VERSION >= 4000
  cres = cuDeviceGetAttribute(domain, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, cudevice);
  if (cres != CUDA_SUCCESS) {
    errno = ENOSYS;
    return -1;
  }
#else
  *domain = 0;
#endif
  cres = cuDeviceGetAttribute(bus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, cudevice);
  if (cres != CUDA_SUCCESS) {
    errno = ENOSYS;
    return -1;
  }
  cres = cuDeviceGetAttribute(dev, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, cudevice);
  if (cres != CUDA_SUCCESS) {
    errno = ENOSYS;
    return -1;
  }

  return 0;
}

/** \brief Get the CPU set of processors that are physically
 * close to device \p cudevice.
 *
 * Store in \p set the CPU-set describing the locality of the CUDA device \p cudevice.
 *
 * Topology \p topology and device \p cudevice must match the local machine.
 * I/O devices detection and the CUDA component are not needed in the topology.
 *
 * The function only returns the locality of the device.
 * If more information about the device is needed, OS objects should
 * be used instead, see hwloc_cuda_get_device_osdev()
 * and hwloc_cuda_get_device_osdev_by_index().
 *
 * This function is currently only implemented in a meaningful way for
 * Linux; other systems will simply get a full cpuset.
 */
static __hwloc_inline int
hwloc_cuda_get_device_cpuset(hwloc_topology_t topology __hwloc_attribute_unused,
			     CUdevice cudevice, hwloc_cpuset_t set)
{
#ifdef HWLOC_LINUX_SYS
  /* If we're on Linux, use the sysfs mechanism to get the local cpus */
#define HWLOC_CUDA_DEVICE_SYSFS_PATH_MAX 128
  char path[HWLOC_CUDA_DEVICE_SYSFS_PATH_MAX];
  int domainid, busid, deviceid;

  if (hwloc_cuda_get_device_pci_ids(topology, cudevice, &domainid, &busid, &deviceid))
    return -1;

  if (!hwloc_topology_is_thissystem(topology)) {
    errno = EINVAL;
    return -1;
  }

  sprintf(path, "/sys/bus/pci/devices/%04x:%02x:%02x.0/local_cpus", domainid, busid, deviceid);
  if (hwloc_linux_read_path_as_cpumask(path, set) < 0
      || hwloc_bitmap_iszero(set))
    hwloc_bitmap_copy(set, hwloc_topology_get_complete_cpuset(topology));
#else
  /* Non-Linux systems simply get a full cpuset */
  hwloc_bitmap_copy(set, hwloc_topology_get_complete_cpuset(topology));
#endif
  return 0;
}

/** \brief Get the hwloc PCI device object corresponding to the
 * CUDA device \p cudevice.
 *
 * \return The hwloc PCI device object describing the CUDA device \p cudevice.
 * \return \c NULL if none could be found.
 *
 * Topology \p topology and device \p cudevice must match the local machine.
 * I/O devices detection must be enabled in topology \p topology.
 * The CUDA component is not needed in the topology.
 */
static __hwloc_inline hwloc_obj_t
hwloc_cuda_get_device_pcidev(hwloc_topology_t topology, CUdevice cudevice)
{
  int domain, bus, dev;

  if (hwloc_cuda_get_device_pci_ids(topology, cudevice, &domain, &bus, &dev))
    return NULL;

  return hwloc_get_pcidev_by_busid(topology, domain, bus, dev, 0);
}

/** \brief Get the hwloc OS device object corresponding to CUDA device \p cudevice.
 *
 * \return The hwloc OS device object that describes the given CUDA device \p cudevice.
 * \return \c NULL if none could be found.
 *
 * Topology \p topology and device \p cudevice must match the local machine.
 * I/O devices detection and the CUDA component must be enabled in the topology.
 * If not, the locality of the object may still be found using
 * hwloc_cuda_get_device_cpuset().
 *
 * \note This function cannot work if PCI devices are filtered out.
 *
 * \note The corresponding hwloc PCI device may be found by looking
 * at the result parent pointer (unless PCI devices are filtered out).
 */
static __hwloc_inline hwloc_obj_t
hwloc_cuda_get_device_osdev(hwloc_topology_t topology, CUdevice cudevice)
{
	hwloc_obj_t osdev = NULL;
	int domain, bus, dev;

	if (hwloc_cuda_get_device_pci_ids(topology, cudevice, &domain, &bus, &dev))
		return NULL;

	osdev = NULL;
	while ((osdev = hwloc_get_next_osdev(topology, osdev)) != NULL) {
		hwloc_obj_t pcidev = osdev->parent;
		if (strncmp(osdev->name, "cuda", 4))
			continue;
		if (pcidev
		    && pcidev->type == HWLOC_OBJ_PCI_DEVICE
		    && (int) pcidev->attr->pcidev.domain == domain
		    && (int) pcidev->attr->pcidev.bus == bus
		    && (int) pcidev->attr->pcidev.dev == dev
		    && pcidev->attr->pcidev.func == 0)
			return osdev;
		/* if PCI are filtered out, we need a info attr to match on */
	}

	return NULL;
}

/** \brief Get the hwloc OS device object corresponding to the
 * CUDA device whose index is \p idx.
 *
 * \return The hwloc OS device object describing the CUDA device whose index is \p idx.
 * \return \c NULL if none could be found.
 *
 * The topology \p topology does not necessarily have to match the current
 * machine. For instance the topology may be an XML import of a remote host.
 * I/O devices detection and the CUDA component must be enabled in the topology.
 *
 * \note The corresponding PCI device object can be obtained by looking
 * at the OS device parent object (unless PCI devices are filtered out).
 *
 * \note This function is identical to hwloc_cudart_get_device_osdev_by_index().
 */
static __hwloc_inline hwloc_obj_t
hwloc_cuda_get_device_osdev_by_index(hwloc_topology_t topology, unsigned idx)
{
	hwloc_obj_t osdev = NULL;
	while ((osdev = hwloc_get_next_osdev(topology, osdev)) != NULL) {
		if (HWLOC_OBJ_OSDEV_COPROC == osdev->attr->osdev.type
		    && osdev->name
		    && !strncmp("cuda", osdev->name, 4)
		    && atoi(osdev->name + 4) == (int) idx)
			return osdev;
	}
	return NULL;
}

/** @} */


#ifdef __cplusplus
} /* extern "C" */
#endif


#endif /* HWLOC_CUDA_H */
