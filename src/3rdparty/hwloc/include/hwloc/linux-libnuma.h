/*
 * Copyright © 2009 CNRS
 * Copyright © 2009-2023 Inria.  All rights reserved.
 * Copyright © 2009-2010, 2012 Université Bordeaux
 * See COPYING in top-level directory.
 */

/** \file
 * \brief Macros to help interaction between hwloc and Linux libnuma.
 *
 * Applications that use both Linux libnuma and hwloc may want to
 * include this file so as to ease conversion between their respective types.
*/

#ifndef HWLOC_LINUX_LIBNUMA_H
#define HWLOC_LINUX_LIBNUMA_H

#include "hwloc.h"

#include <numa.h>


#ifdef __cplusplus
extern "C" {
#endif


/** \defgroup hwlocality_linux_libnuma_ulongs Interoperability with Linux libnuma unsigned long masks
 *
 * This interface helps converting between Linux libnuma unsigned long masks
 * and hwloc cpusets and nodesets.
 *
 * \note Topology \p topology must match the current machine.
 *
 * \note The behavior of libnuma is undefined if the kernel is not NUMA-aware.
 * (when CONFIG_NUMA is not set in the kernel configuration).
 * This helper and libnuma may thus not be strictly compatible in this case,
 * which may be detected by checking whether numa_available() returns -1.
 *
 * @{
 */


/** \brief Convert hwloc CPU set \p cpuset into the array of unsigned long \p mask
 *
 * \p mask is the array of unsigned long that will be filled.
 * \p maxnode contains the maximal node number that may be stored in \p mask.
 * \p maxnode will be set to the maximal node number that was found, plus one.
 *
 * This function may be used before calling set_mempolicy, mbind, migrate_pages
 * or any other function that takes an array of unsigned long and a maximal
 * node number as input parameter.
 *
 * \return 0.
 */
static __hwloc_inline int
hwloc_cpuset_to_linux_libnuma_ulongs(hwloc_topology_t topology, hwloc_const_cpuset_t cpuset,
				    unsigned long *mask, unsigned long *maxnode)
{
  int depth = hwloc_get_type_depth(topology, HWLOC_OBJ_NUMANODE);
  unsigned long outmaxnode = -1;
  hwloc_obj_t node = NULL;

  /* round-up to the next ulong and clear all bytes */
  *maxnode = (*maxnode + 8*sizeof(*mask) - 1) & ~(8*sizeof(*mask) - 1);
  memset(mask, 0, *maxnode/8);

  while ((node = hwloc_get_next_obj_covering_cpuset_by_depth(topology, cpuset, depth, node)) != NULL) {
    if (node->os_index >= *maxnode)
      continue;
    mask[node->os_index/sizeof(*mask)/8] |= 1UL << (node->os_index % (sizeof(*mask)*8));
    if (outmaxnode == (unsigned long) -1 || outmaxnode < node->os_index)
      outmaxnode = node->os_index;
  }

  *maxnode = outmaxnode+1;
  return 0;
}

/** \brief Convert hwloc NUMA node set \p nodeset into the array of unsigned long \p mask
 *
 * \p mask is the array of unsigned long that will be filled.
 * \p maxnode contains the maximal node number that may be stored in \p mask.
 * \p maxnode will be set to the maximal node number that was found, plus one.
 *
 * This function may be used before calling set_mempolicy, mbind, migrate_pages
 * or any other function that takes an array of unsigned long and a maximal
 * node number as input parameter.
 *
 * \return 0.
 */
static __hwloc_inline int
hwloc_nodeset_to_linux_libnuma_ulongs(hwloc_topology_t topology, hwloc_const_nodeset_t nodeset,
				      unsigned long *mask, unsigned long *maxnode)
{
  int depth = hwloc_get_type_depth(topology, HWLOC_OBJ_NUMANODE);
  unsigned long outmaxnode = -1;
  hwloc_obj_t node = NULL;

  /* round-up to the next ulong and clear all bytes */
  *maxnode = (*maxnode + 8*sizeof(*mask) - 1) & ~(8*sizeof(*mask) - 1);
  memset(mask, 0, *maxnode/8);

  while ((node = hwloc_get_next_obj_by_depth(topology, depth, node)) != NULL) {
    if (node->os_index >= *maxnode)
      continue;
    if (!hwloc_bitmap_isset(nodeset, node->os_index))
      continue;
    mask[node->os_index/sizeof(*mask)/8] |= 1UL << (node->os_index % (sizeof(*mask)*8));
    if (outmaxnode == (unsigned long) -1 || outmaxnode < node->os_index)
      outmaxnode = node->os_index;
  }

  *maxnode = outmaxnode+1;
  return 0;
}

/** \brief Convert the array of unsigned long \p mask into hwloc CPU set
 *
 * \p mask is a array of unsigned long that will be read.
 * \p maxnode contains the maximal node number that may be read in \p mask.
 *
 * This function may be used after calling get_mempolicy or any other function
 * that takes an array of unsigned long as output parameter (and possibly
 * a maximal node number as input parameter).
 *
 * \return 0 on success.
 * \return -1 on error, for instance if failing an internal reallocation.
 */
static __hwloc_inline int
hwloc_cpuset_from_linux_libnuma_ulongs(hwloc_topology_t topology, hwloc_cpuset_t cpuset,
				      const unsigned long *mask, unsigned long maxnode)
{
  int depth = hwloc_get_type_depth(topology, HWLOC_OBJ_NUMANODE);
  hwloc_obj_t node = NULL;
  hwloc_bitmap_zero(cpuset);
  while ((node = hwloc_get_next_obj_by_depth(topology, depth, node)) != NULL)
    if (node->os_index < maxnode
	&& (mask[node->os_index/sizeof(*mask)/8] & (1UL << (node->os_index % (sizeof(*mask)*8)))))
      if (hwloc_bitmap_or(cpuset, cpuset, node->cpuset) < 0)
        return -1;
  return 0;
}

/** \brief Convert the array of unsigned long \p mask into hwloc NUMA node set
 *
 * \p mask is a array of unsigned long that will be read.
 * \p maxnode contains the maximal node number that may be read in \p mask.
 *
 * This function may be used after calling get_mempolicy or any other function
 * that takes an array of unsigned long as output parameter (and possibly
 * a maximal node number as input parameter).
 *
 * \return 0 on success.
 * \return -1 with errno set to \c ENOMEM if some internal reallocation failed.
 */
static __hwloc_inline int
hwloc_nodeset_from_linux_libnuma_ulongs(hwloc_topology_t topology, hwloc_nodeset_t nodeset,
					const unsigned long *mask, unsigned long maxnode)
{
  int depth = hwloc_get_type_depth(topology, HWLOC_OBJ_NUMANODE);
  hwloc_obj_t node = NULL;
  hwloc_bitmap_zero(nodeset);
  while ((node = hwloc_get_next_obj_by_depth(topology, depth, node)) != NULL)
    if (node->os_index < maxnode
	&& (mask[node->os_index/sizeof(*mask)/8] & (1UL << (node->os_index % (sizeof(*mask)*8)))))
      if (hwloc_bitmap_set(nodeset, node->os_index) < 0)
        return -1;
  return 0;
}

/** @} */



/** \defgroup hwlocality_linux_libnuma_bitmask Interoperability with Linux libnuma bitmask
 *
 * This interface helps converting between Linux libnuma bitmasks
 * and hwloc cpusets and nodesets.
 *
 * \note Topology \p topology must match the current machine.
 *
 * \note The behavior of libnuma is undefined if the kernel is not NUMA-aware.
 * (when CONFIG_NUMA is not set in the kernel configuration).
 * This helper and libnuma may thus not be strictly compatible in this case,
 * which may be detected by checking whether numa_available() returns -1.
 *
 * @{
 */


/** \brief Convert hwloc CPU set \p cpuset into the returned libnuma bitmask
 *
 * The returned bitmask should later be freed with numa_bitmask_free.
 *
 * This function may be used before calling many numa_ functions
 * that use a struct bitmask as an input parameter.
 *
 * \return newly allocated struct bitmask, or \c NULL on error.
 */
static __hwloc_inline struct bitmask *
hwloc_cpuset_to_linux_libnuma_bitmask(hwloc_topology_t topology, hwloc_const_cpuset_t cpuset) __hwloc_attribute_malloc;
static __hwloc_inline struct bitmask *
hwloc_cpuset_to_linux_libnuma_bitmask(hwloc_topology_t topology, hwloc_const_cpuset_t cpuset)
{
  int depth = hwloc_get_type_depth(topology, HWLOC_OBJ_NUMANODE);
  hwloc_obj_t node = NULL;
  struct bitmask *bitmask = numa_allocate_cpumask();
  if (!bitmask)
    return NULL;
  while ((node = hwloc_get_next_obj_covering_cpuset_by_depth(topology, cpuset, depth, node)) != NULL)
    if (node->attr->numanode.local_memory)
      numa_bitmask_setbit(bitmask, node->os_index);
  return bitmask;
}

/** \brief Convert hwloc NUMA node set \p nodeset into the returned libnuma bitmask
 *
 * The returned bitmask should later be freed with numa_bitmask_free.
 *
 * This function may be used before calling many numa_ functions
 * that use a struct bitmask as an input parameter.
 *
 * \return newly allocated struct bitmask, or \c NULL on error.
 */
static __hwloc_inline struct bitmask *
hwloc_nodeset_to_linux_libnuma_bitmask(hwloc_topology_t topology, hwloc_const_nodeset_t nodeset) __hwloc_attribute_malloc;
static __hwloc_inline struct bitmask *
hwloc_nodeset_to_linux_libnuma_bitmask(hwloc_topology_t topology, hwloc_const_nodeset_t nodeset)
{
  int depth = hwloc_get_type_depth(topology, HWLOC_OBJ_NUMANODE);
  hwloc_obj_t node = NULL;
  struct bitmask *bitmask = numa_allocate_cpumask();
  if (!bitmask)
    return NULL;
  while ((node = hwloc_get_next_obj_by_depth(topology, depth, node)) != NULL)
    if (hwloc_bitmap_isset(nodeset, node->os_index) && node->attr->numanode.local_memory)
      numa_bitmask_setbit(bitmask, node->os_index);
  return bitmask;
}

/** \brief Convert libnuma bitmask \p bitmask into hwloc CPU set \p cpuset
 *
 * This function may be used after calling many numa_ functions
 * that use a struct bitmask as an output parameter.
 *
 * \return 0 on success.
 * \return -1 with errno set to \c ENOMEM if some internal reallocation failed.
 */
static __hwloc_inline int
hwloc_cpuset_from_linux_libnuma_bitmask(hwloc_topology_t topology, hwloc_cpuset_t cpuset,
					const struct bitmask *bitmask)
{
  int depth = hwloc_get_type_depth(topology, HWLOC_OBJ_NUMANODE);
  hwloc_obj_t node = NULL;
  hwloc_bitmap_zero(cpuset);
  while ((node = hwloc_get_next_obj_by_depth(topology, depth, node)) != NULL)
    if (numa_bitmask_isbitset(bitmask, node->os_index))
      if (hwloc_bitmap_or(cpuset, cpuset, node->cpuset) < 0)
        return -1;
  return 0;
}

/** \brief Convert libnuma bitmask \p bitmask into hwloc NUMA node set \p nodeset
 *
 * This function may be used after calling many numa_ functions
 * that use a struct bitmask as an output parameter.
 *
 * \return 0 on success.
 * \return -1 with errno set to \c ENOMEM if some internal reallocation failed.
 */
static __hwloc_inline int
hwloc_nodeset_from_linux_libnuma_bitmask(hwloc_topology_t topology, hwloc_nodeset_t nodeset,
					 const struct bitmask *bitmask)
{
  int depth = hwloc_get_type_depth(topology, HWLOC_OBJ_NUMANODE);
  hwloc_obj_t node = NULL;
  hwloc_bitmap_zero(nodeset);
  while ((node = hwloc_get_next_obj_by_depth(topology, depth, node)) != NULL)
    if (numa_bitmask_isbitset(bitmask, node->os_index))
      if (hwloc_bitmap_set(nodeset, node->os_index) < 0)
        return -1;
  return 0;
}

/** @} */


#ifdef __cplusplus
} /* extern "C" */
#endif


#endif /* HWLOC_LINUX_NUMA_H */
