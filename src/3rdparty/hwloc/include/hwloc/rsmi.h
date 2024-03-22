/*
 * Copyright © 2012-2023 Inria.  All rights reserved.
 * Copyright (c) 2020, Advanced Micro Devices, Inc. All rights reserved.
 * Written by Advanced Micro Devices,
 * See COPYING in top-level directory.
 */

/** \file
 * \brief Macros to help interaction between hwloc and the ROCm SMI Management Library.
 *
 * Applications that use both hwloc and the ROCm SMI Management Library may want to
 * include this file so as to get topology information for AMD GPU devices.
 */

#ifndef HWLOC_RSMI_H
#define HWLOC_RSMI_H

#include "hwloc.h"
#include "hwloc/autogen/config.h"
#include "hwloc/helper.h"
#ifdef HWLOC_LINUX_SYS
#include "hwloc/linux.h"
#endif

#include <rocm_smi/rocm_smi.h>


#ifdef __cplusplus
extern "C" {
#endif


/** \defgroup hwlocality_rsmi Interoperability with the ROCm SMI Management Library
 *
 * This interface offers ways to retrieve topology information about
 * devices managed by the ROCm SMI Management Library.
 *
 * @{
 */

/** \brief Get the CPU set of logical processors that are physically
 * close to AMD GPU device whose index is \p dv_ind.
 *
 * Store in \p set the CPU-set describing the locality of the AMD GPU device
 * whose index is \p dv_ind.
 *
 * Topology \p topology and device \p dv_ind must match the local machine.
 * I/O devices detection and the ROCm SMI component are not needed in the
 * topology.
 *
 * The function only returns the locality of the device.
 * If more information about the device is needed, OS objects should
 * be used instead, see hwloc_rsmi_get_device_osdev()
 * and hwloc_rsmi_get_device_osdev_by_index().
 *
 * This function is currently only implemented in a meaningful way for
 * Linux; other systems will simply get a full cpuset.
 *
 * \return 0 on success.
 * \return -1 on error, for instance if device information could not be found.
 */
static __hwloc_inline int
hwloc_rsmi_get_device_cpuset(hwloc_topology_t topology __hwloc_attribute_unused,
                             uint32_t dv_ind, hwloc_cpuset_t set)
{
#ifdef HWLOC_LINUX_SYS
  /* If we're on Linux, use the sysfs mechanism to get the local cpus */
#define HWLOC_RSMI_DEVICE_SYSFS_PATH_MAX 128
  char path[HWLOC_RSMI_DEVICE_SYSFS_PATH_MAX];
  rsmi_status_t ret;
  uint64_t bdfid = 0;
  unsigned domain, device, bus;

  if (!hwloc_topology_is_thissystem(topology)) {
    errno = EINVAL;
    return -1;
  }

  ret = rsmi_dev_pci_id_get(dv_ind, &bdfid);
  if (RSMI_STATUS_SUCCESS != ret) {
    errno = EINVAL;
    return -1;
  }
  domain = (bdfid>>32) & 0xffffffff;
  bus = ((bdfid & 0xffff)>>8) & 0xff;
  device = ((bdfid & 0xff)>>3) & 0x1f;

  sprintf(path, "/sys/bus/pci/devices/%04x:%02x:%02x.0/local_cpus", domain, bus, device);
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
 * AMD GPU device whose index is \p dv_ind.
 *
 * \return The hwloc OS device object describing the AMD GPU device whose
 * index is \p dv_ind.
 * \return \c NULL if none could be found.
 *
 * The topology \p topology does not necessarily have to match the current
 * machine. For instance the topology may be an XML import of a remote host.
 * I/O devices detection and the ROCm SMI component must be enabled in the
 * topology.
 *
 * \note The corresponding PCI device object can be obtained by looking
 * at the OS device parent object (unless PCI devices are filtered out).
 */
static __hwloc_inline hwloc_obj_t
hwloc_rsmi_get_device_osdev_by_index(hwloc_topology_t topology, uint32_t dv_ind)
{
  hwloc_obj_t osdev = NULL;
  while ((osdev = hwloc_get_next_osdev(topology, osdev)) != NULL) {
    if (HWLOC_OBJ_OSDEV_GPU == osdev->attr->osdev.type
      && osdev->name
      && !strncmp("rsmi", osdev->name, 4)
      && atoi(osdev->name + 4) == (int) dv_ind)
      return osdev;
  }
  return NULL;
}

/** \brief Get the hwloc OS device object corresponding to AMD GPU device,
 * whose index is \p dv_ind.
 *
 * \return The hwloc OS device object that describes the given
 * AMD GPU, whose index is \p dv_ind.
 * \return \c NULL if none could be found.
 *
 * Topology \p topology and device \p dv_ind must match the local machine.
 * I/O devices detection and the ROCm SMI component must be enabled in the
 * topology. If not, the locality of the object may still be found using
 * hwloc_rsmi_get_device_cpuset().
 *
 * \note The corresponding hwloc PCI device may be found by looking
 * at the result parent pointer (unless PCI devices are filtered out).
 */
static __hwloc_inline hwloc_obj_t
hwloc_rsmi_get_device_osdev(hwloc_topology_t topology, uint32_t dv_ind)
{
  hwloc_obj_t osdev;
  rsmi_status_t ret;
  uint64_t bdfid = 0;
  unsigned domain, device, bus, func;
  uint64_t id;
  char uuid[64];

  if (!hwloc_topology_is_thissystem(topology)) {
    errno = EINVAL;
    return NULL;
  }

  ret = rsmi_dev_pci_id_get(dv_ind, &bdfid);
  if (RSMI_STATUS_SUCCESS != ret) {
    errno = EINVAL;
    return NULL;
  }
  domain = (bdfid>>32) & 0xffffffff;
  bus = ((bdfid & 0xffff)>>8) & 0xff;
  device = ((bdfid & 0xff)>>3) & 0x1f;
  func = bdfid & 0x7;

  ret = rsmi_dev_unique_id_get(dv_ind, &id);
  if (RSMI_STATUS_SUCCESS != ret)
    uuid[0] = '\0';
  else
    sprintf(uuid, "%lx", id);

  osdev = NULL;
  while ((osdev = hwloc_get_next_osdev(topology, osdev)) != NULL) {
    hwloc_obj_t pcidev = osdev->parent;
    const char *info;

    if (strncmp(osdev->name, "rsmi", 4))
      continue;

    if (pcidev
      && pcidev->type == HWLOC_OBJ_PCI_DEVICE
      && pcidev->attr->pcidev.domain == domain
      && pcidev->attr->pcidev.bus == bus
      && pcidev->attr->pcidev.dev == device
      && pcidev->attr->pcidev.func == func)
      return osdev;

    info = hwloc_obj_get_info_by_name(osdev, "AMDUUID");
    if (info && !strcmp(info, uuid))
      return osdev;
  }

  return NULL;
}

/** @} */


#ifdef __cplusplus
} /* extern "C" */
#endif


#endif /* HWLOC_RSMI_H */
