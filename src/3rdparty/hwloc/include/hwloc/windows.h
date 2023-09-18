/*
 * Copyright © 2021 Inria.  All rights reserved.
 * See COPYING in top-level directory.
 */

/** \file
 * \brief Macros to help interaction between hwloc and Windows.
 *
 * Applications that use hwloc on Windows may want to include this file
 * for Windows specific hwloc features.
 */

#ifndef HWLOC_WINDOWS_H
#define HWLOC_WINDOWS_H

#include "hwloc.h"


#ifdef __cplusplus
extern "C" {
#endif


/** \defgroup hwlocality_windows Windows-specific helpers
 *
 * These functions query Windows processor groups.
 * These groups partition the operating system into virtual sets
 * of up to 64 neighbor PUs.
 * Threads and processes may only be bound inside a single group.
 * Although Windows processor groups may be exposed in the hwloc
 * hierarchy as hwloc Groups, they are also often merged into
 * existing hwloc objects such as NUMA nodes or Packages.
 * This API provides explicit information about Windows processor
 * groups so that applications know whether binding to a large
 * set of PUs may fail because it spans over multiple Windows
 * processor groups.
 *
 * @{
 */


/** \brief Get the number of Windows processor groups
 *
 * \p flags must be 0 for now.
 *
 * \return at least \c 1 on success.
 * \return -1 on error, for instance if the topology does not match
 * the current system (e.g. loaded from another machine through XML).
 */
HWLOC_DECLSPEC int hwloc_windows_get_nr_processor_groups(hwloc_topology_t topology, unsigned long flags);

/** \brief Get the CPU-set of a Windows processor group.
 *
 * Get the set of PU included in the processor group specified
 * by \p pg_index.
 * \p pg_index must be between \c 0 and the value returned
 * by hwloc_windows_get_nr_processor_groups() minus 1.
 *
 * \p flags must be 0 for now.
 *
 * \return \c 0 on success.
 * \return \c -1 on error, for instance if \p pg_index is invalid,
 * or if the topology does not match the current system (e.g. loaded
 * from another machine through XML).
 */
HWLOC_DECLSPEC int hwloc_windows_get_processor_group_cpuset(hwloc_topology_t topology, unsigned pg_index, hwloc_cpuset_t cpuset, unsigned long flags);

/** @} */


#ifdef __cplusplus
} /* extern "C" */
#endif


#endif /* HWLOC_WINDOWS_H */
