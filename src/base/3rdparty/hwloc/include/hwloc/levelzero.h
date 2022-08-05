/*
 * Copyright © 2021 Inria.  All rights reserved.
 * See COPYING in top-level directory.
 */

/** \file
 * \brief Macros to help interaction between hwloc and the oneAPI Level Zero interface.
 *
 * Applications that use both hwloc and Level Zero may want to
 * include this file so as to get topology information for L0 devices.
 */

#ifndef HWLOC_LEVELZERO_H
#define HWLOC_LEVELZERO_H

#include "hwloc.h"
#include "hwloc/autogen/config.h"
#include "hwloc/helper.h"
#ifdef HWLOC_LINUX_SYS
#include "hwloc/linux.h"
#endif

#include <level_zero/ze_api.h>
#include <level_zero/zes_api.h>


#ifdef __cplusplus
extern "C" {
#endif


/** \defgroup hwlocality_levelzero Interoperability with the oneAPI Level Zero interface.
 *
 * This interface offers ways to retrieve topology information about
 * devices managed by the Level Zero API.
 *
 * @{
 */

/** \brief Get the CPU set of logical processors that are physically
 * close to the Level Zero device \p device
 *
 * Store in \p set the CPU-set describing the locality of
 * the Level Zero device \p device.
 *
 * Topology \p topology and device \p device must match the local machine.
 * The Level Zero must have been initialized with Sysman enabled
 * (ZES_ENABLE_SYSMAN=1 in the environment).
 * I/O devices detection and the Level Zero component are not needed in the
 * topology.
 *
 * The function only returns the locality of the device.
 * If more information about the device is needed, OS objects should
 * be used instead, see hwloc_levelzero_get_device_osdev().
 *
 * This function is currently only implemented in a meaningful way for
 * Linux; other systems will simply get a full cpuset.
 */
static __hwloc_inline int
hwloc_levelzero_get_device_cpuset(hwloc_topology_t topology __hwloc_attribute_unused,
                                  ze_device_handle_t device, hwloc_cpuset_t set)
{
#ifdef HWLOC_LINUX_SYS
  /* If we're on Linux, use the sysfs mechanism to get the local cpus */
#define HWLOC_LEVELZERO_DEVICE_SYSFS_PATH_MAX 128
  char path[HWLOC_LEVELZERO_DEVICE_SYSFS_PATH_MAX];
  zes_pci_properties_t pci;
  zes_device_handle_t sdevice = device;
  ze_result_t res;

  if (!hwloc_topology_is_thissystem(topology)) {
    errno = EINVAL;
    return -1;
  }

  res = zesDevicePciGetProperties(sdevice, &pci);
  if (res != ZE_RESULT_SUCCESS) {
    errno = EINVAL;
    return -1;
  }

  sprintf(path, "/sys/bus/pci/devices/%04x:%02x:%02x.%01x/local_cpus",
          pci.address.domain, pci.address.bus, pci.address.device, pci.address.function);
  if (hwloc_linux_read_path_as_cpumask(path, set) < 0
      || hwloc_bitmap_iszero(set))
    hwloc_bitmap_copy(set, hwloc_topology_get_complete_cpuset(topology));
#else
  /* Non-Linux systems simply get a full cpuset */
  hwloc_bitmap_copy(set, hwloc_topology_get_complete_cpuset(topology));
#endif
  return 0;
}

/** \brief Get the hwloc OS device object corresponding to Level Zero device
 * \p device.
 *
 * \return The hwloc OS device object that describes the given Level Zero device \p device.
 * \return \c NULL if none could be found.
 *
 * Topology \p topology and device \p dv_ind must match the local machine.
 * I/O devices detection and the Level Zero component must be enabled in the
 * topology. If not, the locality of the object may still be found using
 * hwloc_levelzero_get_device_cpuset().
 *
 * \note The corresponding hwloc PCI device may be found by looking
 * at the result parent pointer (unless PCI devices are filtered out).
 */
static __hwloc_inline hwloc_obj_t
hwloc_levelzero_get_device_osdev(hwloc_topology_t topology, ze_device_handle_t device)
{
  zes_device_handle_t sdevice = device;
  zes_pci_properties_t pci;
  ze_result_t res;
  hwloc_obj_t osdev;

  if (!hwloc_topology_is_thissystem(topology)) {
    errno = EINVAL;
    return NULL;
  }

  res = zesDevicePciGetProperties(sdevice, &pci);
  if (res != ZE_RESULT_SUCCESS) {
    /* L0 was likely initialized without sysman, don't bother */
    errno = EINVAL;
    return NULL;
  }

  osdev = NULL;
  while ((osdev = hwloc_get_next_osdev(topology, osdev)) != NULL) {
    hwloc_obj_t pcidev = osdev->parent;

    if (strncmp(osdev->name, "ze", 2))
      continue;

    if (pcidev
      && pcidev->type == HWLOC_OBJ_PCI_DEVICE
      && pcidev->attr->pcidev.domain == pci.address.domain
      && pcidev->attr->pcidev.bus == pci.address.bus
      && pcidev->attr->pcidev.dev == pci.address.device
      && pcidev->attr->pcidev.func == pci.address.function)
      return osdev;

    /* FIXME: when we'll have serialnumber, try it in case PCI is filtered-out */
  }

  return NULL;
}

/** @} */


#ifdef __cplusplus
} /* extern "C" */
#endif


#endif /* HWLOC_LEVELZERO_H */
