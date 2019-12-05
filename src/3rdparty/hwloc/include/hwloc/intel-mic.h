/*
 * Copyright © 2013-2016 Inria.  All rights reserved.
 * See COPYING in top-level directory.
 */

/** \file
 * \brief Macros to help interaction between hwloc and Intel Xeon Phi (MIC).
 *
 * Applications that use both hwloc and Intel Xeon Phi (MIC) may want to
 * include this file so as to get topology information for MIC devices.
 */

#ifndef HWLOC_INTEL_MIC_H
#define HWLOC_INTEL_MIC_H

#include "hwloc.h"
#include "hwloc/autogen/config.h"
#include "hwloc/helper.h"

#ifdef HWLOC_LINUX_SYS
#include "hwloc/linux.h"

#include <dirent.h>
#include <string.h>
#endif

#include <stdio.h>
#include <stdlib.h>


#ifdef __cplusplus
extern "C" {
#endif


/** \defgroup hwlocality_intel_mic Interoperability with Intel Xeon Phi (MIC)
 *
 * This interface offers ways to retrieve topology information about
 * Intel Xeon Phi (MIC) devices.
 *
 * @{
 */

/** \brief Get the CPU set of logical processors that are physically
 * close to MIC device whose index is \p idx.
 *
 * Return the CPU set describing the locality of the MIC device whose index is \p idx.
 *
 * Topology \p topology and device index \p idx must match the local machine.
 * I/O devices detection is not needed in the topology.
 *
 * The function only returns the locality of the device.
 * If more information about the device is needed, OS objects should
 * be used instead, see hwloc_intel_mic_get_device_osdev_by_index().
 *
 * This function is currently only implemented in a meaningful way for
 * Linux; other systems will simply get a full cpuset.
 */
static __hwloc_inline int
hwloc_intel_mic_get_device_cpuset(hwloc_topology_t topology __hwloc_attribute_unused,
				  int idx __hwloc_attribute_unused,
				  hwloc_cpuset_t set)
{
#ifdef HWLOC_LINUX_SYS
	/* If we're on Linux, use the sysfs mechanism to get the local cpus */
#define HWLOC_INTEL_MIC_DEVICE_SYSFS_PATH_MAX 128
	char path[HWLOC_INTEL_MIC_DEVICE_SYSFS_PATH_MAX];
	DIR *sysdir = NULL;
	struct dirent *dirent;
	unsigned pcibus, pcidev, pcifunc;

	if (!hwloc_topology_is_thissystem(topology)) {
		errno = EINVAL;
		return -1;
	}

	sprintf(path, "/sys/class/mic/mic%d", idx);
	sysdir = opendir(path);
	if (!sysdir)
		return -1;

	while ((dirent = readdir(sysdir)) != NULL) {
		if (sscanf(dirent->d_name, "pci_%02x:%02x.%02x", &pcibus, &pcidev, &pcifunc) == 3) {
			sprintf(path, "/sys/class/mic/mic%d/pci_%02x:%02x.%02x/local_cpus", idx, pcibus, pcidev, pcifunc);
			if (hwloc_linux_read_path_as_cpumask(path, set) < 0
			    || hwloc_bitmap_iszero(set))
				hwloc_bitmap_copy(set, hwloc_topology_get_complete_cpuset(topology));
			break;
		}
	}

	closedir(sysdir);
#else
	/* Non-Linux systems simply get a full cpuset */
	hwloc_bitmap_copy(set, hwloc_topology_get_complete_cpuset(topology));
#endif
	return 0;
}

/** \brief Get the hwloc OS device object corresponding to the
 * MIC device for the given index.
 *
 * Return the OS device object describing the MIC device whose index is \p idx.
 * Return NULL if there is none.
 *
 * The topology \p topology does not necessarily have to match the current
 * machine. For instance the topology may be an XML import of a remote host.
 * I/O devices detection must be enabled in the topology.
 *
 * \note The corresponding PCI device object can be obtained by looking
 * at the OS device parent object.
 */
static __hwloc_inline hwloc_obj_t
hwloc_intel_mic_get_device_osdev_by_index(hwloc_topology_t topology,
					  unsigned idx)
{
	hwloc_obj_t osdev = NULL;
	while ((osdev = hwloc_get_next_osdev(topology, osdev)) != NULL) {
		if (HWLOC_OBJ_OSDEV_COPROC == osdev->attr->osdev.type
                    && osdev->name
		    && !strncmp("mic", osdev->name, 3)
		    && atoi(osdev->name + 3) == (int) idx)
                        return osdev;
        }
        return NULL;
}

/** @} */


#ifdef __cplusplus
} /* extern "C" */
#endif


#endif /* HWLOC_INTEL_MIC_H */
