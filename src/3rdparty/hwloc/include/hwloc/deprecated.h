/*
 * Copyright © 2009 CNRS
 * Copyright © 2009-2022 Inria.  All rights reserved.
 * Copyright © 2009-2012 Université Bordeaux
 * Copyright © 2009-2010 Cisco Systems, Inc.  All rights reserved.
 * See COPYING in top-level directory.
 */

/**
 * This file contains the inline code of functions declared in hwloc.h
 */

#ifndef HWLOC_DEPRECATED_H
#define HWLOC_DEPRECATED_H

#ifndef HWLOC_H
#error Please include the main hwloc.h instead
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* backward compat with v2.0 before WHOLE_SYSTEM renaming */
#define HWLOC_TOPOLOGY_FLAG_WHOLE_SYSTEM HWLOC_TOPOLOGY_FLAG_INCLUDE_DISALLOWED
/* backward compat with v1.11 before System removal */
#define HWLOC_OBJ_SYSTEM HWLOC_OBJ_MACHINE
/* backward compat with v1.10 before Socket->Package renaming */
#define HWLOC_OBJ_SOCKET HWLOC_OBJ_PACKAGE
/* backward compat with v1.10 before Node->NUMANode clarification */
#define HWLOC_OBJ_NODE HWLOC_OBJ_NUMANODE

/** \brief Add a distances structure.
 *
 * Superseded by hwloc_distances_add_create()+hwloc_distances_add_values()+hwloc_distances_add_commit()
 * in v2.5.
 */
HWLOC_DECLSPEC int hwloc_distances_add(hwloc_topology_t topology,
				       unsigned nbobjs, hwloc_obj_t *objs, hwloc_uint64_t *values,
				       unsigned long kind, unsigned long flags) __hwloc_attribute_deprecated;

/** \brief Insert a misc object by parent.
 *
 * Identical to hwloc_topology_insert_misc_object().
 */
static __hwloc_inline hwloc_obj_t
hwloc_topology_insert_misc_object_by_parent(hwloc_topology_t topology, hwloc_obj_t parent, const char *name) __hwloc_attribute_deprecated;
static __hwloc_inline hwloc_obj_t
hwloc_topology_insert_misc_object_by_parent(hwloc_topology_t topology, hwloc_obj_t parent, const char *name)
{
  return hwloc_topology_insert_misc_object(topology, parent, name);
}

/** \brief Stringify the cpuset containing a set of objects.
 *
 * If \p size is 0, \p string may safely be \c NULL.
 *
 * \return the number of characters that were actually written if not truncating,
 * or that would have been written (not including the ending \\0).
 */
static __hwloc_inline int
hwloc_obj_cpuset_snprintf(char *str, size_t size, size_t nobj, struct hwloc_obj * const *objs) __hwloc_attribute_deprecated;
static __hwloc_inline int
hwloc_obj_cpuset_snprintf(char *str, size_t size, size_t nobj, struct hwloc_obj * const *objs)
{
  hwloc_bitmap_t set = hwloc_bitmap_alloc();
  int res;
  unsigned i;

  hwloc_bitmap_zero(set);
  for(i=0; i<nobj; i++)
    if (objs[i]->cpuset)
      hwloc_bitmap_or(set, set, objs[i]->cpuset);

  res = hwloc_bitmap_snprintf(str, size, set);
  hwloc_bitmap_free(set);
  return res;
}

/** \brief Convert a type string into a type and some attributes.
 *
 * Deprecated by hwloc_type_sscanf()
 */
static __hwloc_inline int
hwloc_obj_type_sscanf(const char *string, hwloc_obj_type_t *typep, int *depthattrp, void *typeattrp, size_t typeattrsize) __hwloc_attribute_deprecated;
static __hwloc_inline int
hwloc_obj_type_sscanf(const char *string, hwloc_obj_type_t *typep, int *depthattrp, void *typeattrp, size_t typeattrsize)
{
  union hwloc_obj_attr_u attr;
  int err = hwloc_type_sscanf(string, typep, &attr, sizeof(attr));
  if (err < 0)
    return err;
  if (hwloc_obj_type_is_cache(*typep)) {
    if (depthattrp)
      *depthattrp = (int) attr.cache.depth;
    if (typeattrp && typeattrsize >= sizeof(hwloc_obj_cache_type_t))
      memcpy(typeattrp, &attr.cache.type, sizeof(hwloc_obj_cache_type_t));
  } else if (*typep == HWLOC_OBJ_GROUP) {
    if (depthattrp)
      *depthattrp = (int) attr.group.depth;
  }
  return 0;
}

/** \brief Set the default memory binding policy of the current
 * process or thread to prefer the NUMA node(s) specified by physical \p nodeset
 */
static __hwloc_inline int
hwloc_set_membind_nodeset(hwloc_topology_t topology, hwloc_const_nodeset_t nodeset, hwloc_membind_policy_t policy, int flags) __hwloc_attribute_deprecated;
static __hwloc_inline int
hwloc_set_membind_nodeset(hwloc_topology_t topology, hwloc_const_nodeset_t nodeset, hwloc_membind_policy_t policy, int flags)
{
  return hwloc_set_membind(topology, nodeset, policy, flags | HWLOC_MEMBIND_BYNODESET);
}

/** \brief Query the default memory binding policy and physical locality of the
 * current process or thread.
 */
static __hwloc_inline int
hwloc_get_membind_nodeset(hwloc_topology_t topology, hwloc_nodeset_t nodeset, hwloc_membind_policy_t * policy, int flags) __hwloc_attribute_deprecated;
static __hwloc_inline int
hwloc_get_membind_nodeset(hwloc_topology_t topology, hwloc_nodeset_t nodeset, hwloc_membind_policy_t * policy, int flags)
{
  return hwloc_get_membind(topology, nodeset, policy, flags | HWLOC_MEMBIND_BYNODESET);
}

/** \brief Set the default memory binding policy of the specified
 * process to prefer the NUMA node(s) specified by physical \p nodeset
 */
static __hwloc_inline int
hwloc_set_proc_membind_nodeset(hwloc_topology_t topology, hwloc_pid_t pid, hwloc_const_nodeset_t nodeset, hwloc_membind_policy_t policy, int flags) __hwloc_attribute_deprecated;
static __hwloc_inline int
hwloc_set_proc_membind_nodeset(hwloc_topology_t topology, hwloc_pid_t pid, hwloc_const_nodeset_t nodeset, hwloc_membind_policy_t policy, int flags)
{
  return hwloc_set_proc_membind(topology, pid, nodeset, policy, flags | HWLOC_MEMBIND_BYNODESET);
}

/** \brief Query the default memory binding policy and physical locality of the
 * specified process.
 */
static __hwloc_inline int
hwloc_get_proc_membind_nodeset(hwloc_topology_t topology, hwloc_pid_t pid, hwloc_nodeset_t nodeset, hwloc_membind_policy_t * policy, int flags) __hwloc_attribute_deprecated;
static __hwloc_inline int
hwloc_get_proc_membind_nodeset(hwloc_topology_t topology, hwloc_pid_t pid, hwloc_nodeset_t nodeset, hwloc_membind_policy_t * policy, int flags)
{
  return hwloc_get_proc_membind(topology, pid, nodeset, policy, flags | HWLOC_MEMBIND_BYNODESET);
}

/** \brief Bind the already-allocated memory identified by (addr, len)
 * to the NUMA node(s) in physical \p nodeset.
 */
static __hwloc_inline int
hwloc_set_area_membind_nodeset(hwloc_topology_t topology, const void *addr, size_t len, hwloc_const_nodeset_t nodeset, hwloc_membind_policy_t policy, int flags) __hwloc_attribute_deprecated;
static __hwloc_inline int
hwloc_set_area_membind_nodeset(hwloc_topology_t topology, const void *addr, size_t len, hwloc_const_nodeset_t nodeset, hwloc_membind_policy_t policy, int flags)
{
  return hwloc_set_area_membind(topology, addr, len, nodeset, policy, flags | HWLOC_MEMBIND_BYNODESET);
}

/** \brief Query the physical NUMA node(s) and binding policy of the memory
 * identified by (\p addr, \p len ).
 */
static __hwloc_inline int
hwloc_get_area_membind_nodeset(hwloc_topology_t topology, const void *addr, size_t len, hwloc_nodeset_t nodeset, hwloc_membind_policy_t * policy, int flags) __hwloc_attribute_deprecated;
static __hwloc_inline int
hwloc_get_area_membind_nodeset(hwloc_topology_t topology, const void *addr, size_t len, hwloc_nodeset_t nodeset, hwloc_membind_policy_t * policy, int flags)
{
  return hwloc_get_area_membind(topology, addr, len, nodeset, policy, flags | HWLOC_MEMBIND_BYNODESET);
}

/** \brief Allocate some memory on the given physical nodeset \p nodeset
 */
static __hwloc_inline void *
hwloc_alloc_membind_nodeset(hwloc_topology_t topology, size_t len, hwloc_const_nodeset_t nodeset, hwloc_membind_policy_t policy, int flags) __hwloc_attribute_malloc __hwloc_attribute_deprecated;
static __hwloc_inline void *
hwloc_alloc_membind_nodeset(hwloc_topology_t topology, size_t len, hwloc_const_nodeset_t nodeset, hwloc_membind_policy_t policy, int flags)
{
  return hwloc_alloc_membind(topology, len, nodeset, policy, flags | HWLOC_MEMBIND_BYNODESET);
}

/** \brief Allocate some memory on the given nodeset \p nodeset.
 */
static __hwloc_inline void *
hwloc_alloc_membind_policy_nodeset(hwloc_topology_t topology, size_t len, hwloc_const_nodeset_t nodeset, hwloc_membind_policy_t policy, int flags) __hwloc_attribute_malloc __hwloc_attribute_deprecated;
static __hwloc_inline void *
hwloc_alloc_membind_policy_nodeset(hwloc_topology_t topology, size_t len, hwloc_const_nodeset_t nodeset, hwloc_membind_policy_t policy, int flags)
{
  return hwloc_alloc_membind_policy(topology, len, nodeset, policy, flags | HWLOC_MEMBIND_BYNODESET);
}

/** \brief Convert a CPU set into a NUMA node set and handle non-NUMA cases
 */
static __hwloc_inline void
hwloc_cpuset_to_nodeset_strict(hwloc_topology_t topology, hwloc_const_cpuset_t _cpuset, hwloc_nodeset_t nodeset) __hwloc_attribute_deprecated;
static __hwloc_inline void
hwloc_cpuset_to_nodeset_strict(hwloc_topology_t topology, hwloc_const_cpuset_t _cpuset, hwloc_nodeset_t nodeset)
{
  hwloc_cpuset_to_nodeset(topology, _cpuset, nodeset);
}

/** \brief Convert a NUMA node set into a CPU set and handle non-NUMA cases
 */
static __hwloc_inline void
hwloc_cpuset_from_nodeset_strict(hwloc_topology_t topology, hwloc_cpuset_t _cpuset, hwloc_const_nodeset_t nodeset) __hwloc_attribute_deprecated;
static __hwloc_inline void
hwloc_cpuset_from_nodeset_strict(hwloc_topology_t topology, hwloc_cpuset_t _cpuset, hwloc_const_nodeset_t nodeset)
{
  hwloc_cpuset_from_nodeset(topology, _cpuset, nodeset);
}


#ifdef __cplusplus
} /* extern "C" */
#endif


#endif /* HWLOC_DEPRECATED_H */
