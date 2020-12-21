/*
 * Copyright © 2009 CNRS
 * Copyright © 2009-2020 Inria.  All rights reserved.
 * Copyright © 2009-2010 Université Bordeaux
 * Copyright © 2009-2011 Cisco Systems, Inc.  All rights reserved.
 * See COPYING in top-level directory.
 */

/** \file
 * \brief Macros to help interaction between hwloc and OpenFabrics
 * verbs.
 *
 * Applications that use both hwloc and OpenFabrics verbs may want to
 * include this file so as to get topology information for OpenFabrics
 * hardware (InfiniBand, etc).
 *
 */

#ifndef HWLOC_OPENFABRICS_VERBS_H
#define HWLOC_OPENFABRICS_VERBS_H

#include "hwloc.h"
#include "hwloc/autogen/config.h"
#ifdef HWLOC_LINUX_SYS
#include "hwloc/linux.h"
#endif

#include <infiniband/verbs.h>


#ifdef __cplusplus
extern "C" {
#endif


/** \defgroup hwlocality_openfabrics Interoperability with OpenFabrics
 *
 * This interface offers ways to retrieve topology information about
 * OpenFabrics devices (InfiniBand, Omni-Path, usNIC, etc).
 *
 * @{
 */

/** \brief Get the CPU set of processors that are physically
 * close to device \p ibdev.
 *
 * Return the CPU set describing the locality of the OpenFabrics
 * device \p ibdev (InfiniBand, etc).
 *
 * Topology \p topology and device \p ibdev must match the local machine.
 * I/O devices detection is not needed in the topology.
 *
 * The function only returns the locality of the device.
 * If more information about the device is needed, OS objects should
 * be used instead, see hwloc_ibv_get_device_osdev()
 * and hwloc_ibv_get_device_osdev_by_name().
 *
 * This function is currently only implemented in a meaningful way for
 * Linux; other systems will simply get a full cpuset.
 */
static __hwloc_inline int
hwloc_ibv_get_device_cpuset(hwloc_topology_t topology __hwloc_attribute_unused,
			    struct ibv_device *ibdev, hwloc_cpuset_t set)
{
#ifdef HWLOC_LINUX_SYS
  /* If we're on Linux, use the verbs-provided sysfs mechanism to
     get the local cpus */
#define HWLOC_OPENFABRICS_VERBS_SYSFS_PATH_MAX 128
  char path[HWLOC_OPENFABRICS_VERBS_SYSFS_PATH_MAX];

  if (!hwloc_topology_is_thissystem(topology)) {
    errno = EINVAL;
    return -1;
  }

  sprintf(path, "/sys/class/infiniband/%s/device/local_cpus",
	  ibv_get_device_name(ibdev));
  if (hwloc_linux_read_path_as_cpumask(path, set) < 0
      || hwloc_bitmap_iszero(set))
    hwloc_bitmap_copy(set, hwloc_topology_get_complete_cpuset(topology));
#else
  /* Non-Linux systems simply get a full cpuset */
  hwloc_bitmap_copy(set, hwloc_topology_get_complete_cpuset(topology));
#endif
  return 0;
}

/** \brief Get the hwloc OS device object corresponding to the OpenFabrics
 * device named \p ibname.
 *
 * Return the OS device object describing the OpenFabrics device
 * (InfiniBand, Omni-Path, usNIC, etc) whose name is \p ibname
 * (mlx5_0, hfi1_0, usnic_0, qib0, etc).
 * Returns NULL if there is none.
 * The name \p ibname is usually obtained from ibv_get_device_name().
 *
 * The topology \p topology does not necessarily have to match the current
 * machine. For instance the topology may be an XML import of a remote host.
 * I/O devices detection must be enabled in the topology.
 *
 * \note The corresponding PCI device object can be obtained by looking
 * at the OS device parent object.
 */
static __hwloc_inline hwloc_obj_t
hwloc_ibv_get_device_osdev_by_name(hwloc_topology_t topology,
				   const char *ibname)
{
	hwloc_obj_t osdev = NULL;
	while ((osdev = hwloc_get_next_osdev(topology, osdev)) != NULL) {
		if (HWLOC_OBJ_OSDEV_OPENFABRICS == osdev->attr->osdev.type
		    && osdev->name && !strcmp(ibname, osdev->name))
			return osdev;
	}
	return NULL;
}

/** \brief Get the hwloc OS device object corresponding to the OpenFabrics
 * device \p ibdev.
 *
 * Return the OS device object describing the OpenFabrics device \p ibdev
 * (InfiniBand, etc). Returns NULL if there is none.
 *
 * Topology \p topology and device \p ibdev must match the local machine.
 * I/O devices detection must be enabled in the topology.
 * If not, the locality of the object may still be found using
 * hwloc_ibv_get_device_cpuset().
 *
 * \note The corresponding PCI device object can be obtained by looking
 * at the OS device parent object.
 */
static __hwloc_inline hwloc_obj_t
hwloc_ibv_get_device_osdev(hwloc_topology_t topology,
			   struct ibv_device *ibdev)
{
	if (!hwloc_topology_is_thissystem(topology)) {
		errno = EINVAL;
		return NULL;
	}
	return hwloc_ibv_get_device_osdev_by_name(topology, ibv_get_device_name(ibdev));
}

/** @} */


#ifdef __cplusplus
} /* extern "C" */
#endif


#endif /* HWLOC_OPENFABRICS_VERBS_H */
