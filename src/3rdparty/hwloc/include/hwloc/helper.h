/*
 * Copyright © 2009 CNRS
 * Copyright © 2009-2019 Inria.  All rights reserved.
 * Copyright © 2009-2012 Université Bordeaux
 * Copyright © 2009-2010 Cisco Systems, Inc.  All rights reserved.
 * See COPYING in top-level directory.
 */

/** \file
 * \brief High-level hwloc traversal helpers.
 */

#ifndef HWLOC_HELPER_H
#define HWLOC_HELPER_H

#ifndef HWLOC_H
#error Please include the main hwloc.h instead
#endif

#include <stdlib.h>
#include <errno.h>


#ifdef __cplusplus
extern "C" {
#endif


/** \defgroup hwlocality_helper_find_inside Finding Objects inside a CPU set
 * @{
 */

/** \brief Get the first largest object included in the given cpuset \p set.
 *
 * \return the first object that is included in \p set and whose parent is not.
 *
 * This is convenient for iterating over all largest objects within a CPU set
 * by doing a loop getting the first largest object and clearing its CPU set
 * from the remaining CPU set.
 */
static __hwloc_inline hwloc_obj_t
hwloc_get_first_largest_obj_inside_cpuset(hwloc_topology_t topology, hwloc_const_cpuset_t set)
{
  hwloc_obj_t obj = hwloc_get_root_obj(topology);
  if (!hwloc_bitmap_intersects(obj->cpuset, set))
    return NULL;
  while (!hwloc_bitmap_isincluded(obj->cpuset, set)) {
    /* while the object intersects without being included, look at its children */
    hwloc_obj_t child = obj->first_child;
    while (child) {
      if (hwloc_bitmap_intersects(child->cpuset, set))
	break;
      child = child->next_sibling;
    }
    if (!child)
      /* no child intersects, return their father */
      return obj;
    /* found one intersecting child, look at its children */
    obj = child;
  }
  /* obj is included, return it */
  return obj;
}

/** \brief Get the set of largest objects covering exactly a given cpuset \p set
 *
 * \return the number of objects returned in \p objs.
 */
HWLOC_DECLSPEC int hwloc_get_largest_objs_inside_cpuset (hwloc_topology_t topology, hwloc_const_cpuset_t set,
						 hwloc_obj_t * __hwloc_restrict objs, int max);

/** \brief Return the next object at depth \p depth included in CPU set \p set.
 *
 * If \p prev is \c NULL, return the first object at depth \p depth
 * included in \p set.  The next invokation should pass the previous
 * return value in \p prev so as to obtain the next object in \p set.
 *
 * \note Objects with empty CPU sets are ignored
 * (otherwise they would be considered included in any given set).
 *
 * \note This function cannot work if objects at the given depth do
 * not have CPU sets (I/O or Misc objects).
 */
static __hwloc_inline hwloc_obj_t
hwloc_get_next_obj_inside_cpuset_by_depth (hwloc_topology_t topology, hwloc_const_cpuset_t set,
					   int depth, hwloc_obj_t prev)
{
  hwloc_obj_t next = hwloc_get_next_obj_by_depth(topology, depth, prev);
  if (!next)
    return NULL;
  while (next && (hwloc_bitmap_iszero(next->cpuset) || !hwloc_bitmap_isincluded(next->cpuset, set)))
    next = next->next_cousin;
  return next;
}

/** \brief Return the next object of type \p type included in CPU set \p set.
 *
 * If there are multiple or no depth for given type, return \c NULL
 * and let the caller fallback to
 * hwloc_get_next_obj_inside_cpuset_by_depth().
 *
 * \note Objects with empty CPU sets are ignored
 * (otherwise they would be considered included in any given set).
 *
 * \note This function cannot work if objects of the given type do
 * not have CPU sets (I/O or Misc objects).
 */
static __hwloc_inline hwloc_obj_t
hwloc_get_next_obj_inside_cpuset_by_type (hwloc_topology_t topology, hwloc_const_cpuset_t set,
					  hwloc_obj_type_t type, hwloc_obj_t prev)
{
  int depth = hwloc_get_type_depth(topology, type);
  if (depth == HWLOC_TYPE_DEPTH_UNKNOWN || depth == HWLOC_TYPE_DEPTH_MULTIPLE)
    return NULL;
  return hwloc_get_next_obj_inside_cpuset_by_depth(topology, set, depth, prev);
}

/** \brief Return the (logically) \p idx -th object at depth \p depth included in CPU set \p set.
 *
 * \note Objects with empty CPU sets are ignored
 * (otherwise they would be considered included in any given set).
 *
 * \note This function cannot work if objects at the given depth do
 * not have CPU sets (I/O or Misc objects).
 */
static __hwloc_inline hwloc_obj_t
hwloc_get_obj_inside_cpuset_by_depth (hwloc_topology_t topology, hwloc_const_cpuset_t set,
				      int depth, unsigned idx) __hwloc_attribute_pure;
static __hwloc_inline hwloc_obj_t
hwloc_get_obj_inside_cpuset_by_depth (hwloc_topology_t topology, hwloc_const_cpuset_t set,
				      int depth, unsigned idx)
{
  hwloc_obj_t obj = hwloc_get_obj_by_depth (topology, depth, 0);
  unsigned count = 0;
  if (!obj)
    return NULL;
  while (obj) {
    if (!hwloc_bitmap_iszero(obj->cpuset) && hwloc_bitmap_isincluded(obj->cpuset, set)) {
      if (count == idx)
	return obj;
      count++;
    }
    obj = obj->next_cousin;
  }
  return NULL;
}

/** \brief Return the \p idx -th object of type \p type included in CPU set \p set.
 *
 * If there are multiple or no depth for given type, return \c NULL
 * and let the caller fallback to
 * hwloc_get_obj_inside_cpuset_by_depth().
 *
 * \note Objects with empty CPU sets are ignored
 * (otherwise they would be considered included in any given set).
 *
 * \note This function cannot work if objects of the given type do
 * not have CPU sets (I/O or Misc objects).
 */
static __hwloc_inline hwloc_obj_t
hwloc_get_obj_inside_cpuset_by_type (hwloc_topology_t topology, hwloc_const_cpuset_t set,
				     hwloc_obj_type_t type, unsigned idx) __hwloc_attribute_pure;
static __hwloc_inline hwloc_obj_t
hwloc_get_obj_inside_cpuset_by_type (hwloc_topology_t topology, hwloc_const_cpuset_t set,
				     hwloc_obj_type_t type, unsigned idx)
{
  int depth = hwloc_get_type_depth(topology, type);
  if (depth == HWLOC_TYPE_DEPTH_UNKNOWN || depth == HWLOC_TYPE_DEPTH_MULTIPLE)
    return NULL;
  return hwloc_get_obj_inside_cpuset_by_depth(topology, set, depth, idx);
}

/** \brief Return the number of objects at depth \p depth included in CPU set \p set.
 *
 * \note Objects with empty CPU sets are ignored
 * (otherwise they would be considered included in any given set).
 *
 * \note This function cannot work if objects at the given depth do
 * not have CPU sets (I/O or Misc objects).
 */
static __hwloc_inline unsigned
hwloc_get_nbobjs_inside_cpuset_by_depth (hwloc_topology_t topology, hwloc_const_cpuset_t set,
					 int depth) __hwloc_attribute_pure;
static __hwloc_inline unsigned
hwloc_get_nbobjs_inside_cpuset_by_depth (hwloc_topology_t topology, hwloc_const_cpuset_t set,
					 int depth)
{
  hwloc_obj_t obj = hwloc_get_obj_by_depth (topology, depth, 0);
  unsigned count = 0;
  if (!obj)
    return 0;
  while (obj) {
    if (!hwloc_bitmap_iszero(obj->cpuset) && hwloc_bitmap_isincluded(obj->cpuset, set))
      count++;
    obj = obj->next_cousin;
  }
  return count;
}

/** \brief Return the number of objects of type \p type included in CPU set \p set.
 *
 * If no object for that type exists inside CPU set \p set, 0 is
 * returned.  If there are several levels with objects of that type
 * inside CPU set \p set, -1 is returned.
 *
 * \note Objects with empty CPU sets are ignored
 * (otherwise they would be considered included in any given set).
 *
 * \note This function cannot work if objects of the given type do
 * not have CPU sets (I/O objects).
 */
static __hwloc_inline int
hwloc_get_nbobjs_inside_cpuset_by_type (hwloc_topology_t topology, hwloc_const_cpuset_t set,
					hwloc_obj_type_t type) __hwloc_attribute_pure;
static __hwloc_inline int
hwloc_get_nbobjs_inside_cpuset_by_type (hwloc_topology_t topology, hwloc_const_cpuset_t set,
					hwloc_obj_type_t type)
{
  int depth = hwloc_get_type_depth(topology, type);
  if (depth == HWLOC_TYPE_DEPTH_UNKNOWN)
    return 0;
  if (depth == HWLOC_TYPE_DEPTH_MULTIPLE)
    return -1; /* FIXME: agregate nbobjs from different levels? */
  return (int) hwloc_get_nbobjs_inside_cpuset_by_depth(topology, set, depth);
}

/** \brief Return the logical index among the objects included in CPU set \p set.
 *
 * Consult all objects in the same level as \p obj and inside CPU set \p set
 * in the logical order, and return the index of \p obj within them.
 * If \p set covers the entire topology, this is the logical index of \p obj.
 * Otherwise, this is similar to a logical index within the part of the topology
 * defined by CPU set \p set.
 *
 * \note Objects with empty CPU sets are ignored
 * (otherwise they would be considered included in any given set).
 *
 * \note This function cannot work if obj does not have CPU sets (I/O objects).
 */
static __hwloc_inline int
hwloc_get_obj_index_inside_cpuset (hwloc_topology_t topology __hwloc_attribute_unused, hwloc_const_cpuset_t set,
				   hwloc_obj_t obj) __hwloc_attribute_pure;
static __hwloc_inline int
hwloc_get_obj_index_inside_cpuset (hwloc_topology_t topology __hwloc_attribute_unused, hwloc_const_cpuset_t set,
				   hwloc_obj_t obj)
{
  int idx = 0;
  if (!hwloc_bitmap_isincluded(obj->cpuset, set))
    return -1;
  /* count how many objects are inside the cpuset on the way from us to the beginning of the level */
  while ((obj = obj->prev_cousin) != NULL)
    if (!hwloc_bitmap_iszero(obj->cpuset) && hwloc_bitmap_isincluded(obj->cpuset, set))
      idx++;
  return idx;
}

/** @} */



/** \defgroup hwlocality_helper_find_covering Finding Objects covering at least CPU set
 * @{
 */

/** \brief Get the child covering at least CPU set \p set.
 *
 * \return \c NULL if no child matches or if \p set is empty.
 *
 * \note This function cannot work if parent does not have a CPU set (I/O or Misc objects).
 */
static __hwloc_inline hwloc_obj_t
hwloc_get_child_covering_cpuset (hwloc_topology_t topology __hwloc_attribute_unused, hwloc_const_cpuset_t set,
				hwloc_obj_t parent) __hwloc_attribute_pure;
static __hwloc_inline hwloc_obj_t
hwloc_get_child_covering_cpuset (hwloc_topology_t topology __hwloc_attribute_unused, hwloc_const_cpuset_t set,
				hwloc_obj_t parent)
{
  hwloc_obj_t child;
  if (hwloc_bitmap_iszero(set))
    return NULL;
  child = parent->first_child;
  while (child) {
    if (child->cpuset && hwloc_bitmap_isincluded(set, child->cpuset))
      return child;
    child = child->next_sibling;
  }
  return NULL;
}

/** \brief Get the lowest object covering at least CPU set \p set
 *
 * \return \c NULL if no object matches or if \p set is empty.
 */
static __hwloc_inline hwloc_obj_t
hwloc_get_obj_covering_cpuset (hwloc_topology_t topology, hwloc_const_cpuset_t set) __hwloc_attribute_pure;
static __hwloc_inline hwloc_obj_t
hwloc_get_obj_covering_cpuset (hwloc_topology_t topology, hwloc_const_cpuset_t set)
{
  struct hwloc_obj *current = hwloc_get_root_obj(topology);
  if (hwloc_bitmap_iszero(set) || !hwloc_bitmap_isincluded(set, current->cpuset))
    return NULL;
  while (1) {
    hwloc_obj_t child = hwloc_get_child_covering_cpuset(topology, set, current);
    if (!child)
      return current;
    current = child;
  }
}

/** \brief Iterate through same-depth objects covering at least CPU set \p set
 *
 * If object \p prev is \c NULL, return the first object at depth \p
 * depth covering at least part of CPU set \p set.  The next
 * invokation should pass the previous return value in \p prev so as
 * to obtain the next object covering at least another part of \p set.
 *
 * \note This function cannot work if objects at the given depth do
 * not have CPU sets (I/O or Misc objects).
 */
static __hwloc_inline hwloc_obj_t
hwloc_get_next_obj_covering_cpuset_by_depth(hwloc_topology_t topology, hwloc_const_cpuset_t set,
					    int depth, hwloc_obj_t prev)
{
  hwloc_obj_t next = hwloc_get_next_obj_by_depth(topology, depth, prev);
  if (!next)
    return NULL;
  while (next && !hwloc_bitmap_intersects(set, next->cpuset))
    next = next->next_cousin;
  return next;
}

/** \brief Iterate through same-type objects covering at least CPU set \p set
 *
 * If object \p prev is \c NULL, return the first object of type \p
 * type covering at least part of CPU set \p set.  The next invokation
 * should pass the previous return value in \p prev so as to obtain
 * the next object of type \p type covering at least another part of
 * \p set.
 *
 * If there are no or multiple depths for type \p type, \c NULL is returned.
 * The caller may fallback to hwloc_get_next_obj_covering_cpuset_by_depth()
 * for each depth.
 *
 * \note This function cannot work if objects of the given type do
 * not have CPU sets (I/O or Misc objects).
 */
static __hwloc_inline hwloc_obj_t
hwloc_get_next_obj_covering_cpuset_by_type(hwloc_topology_t topology, hwloc_const_cpuset_t set,
					   hwloc_obj_type_t type, hwloc_obj_t prev)
{
  int depth = hwloc_get_type_depth(topology, type);
  if (depth == HWLOC_TYPE_DEPTH_UNKNOWN || depth == HWLOC_TYPE_DEPTH_MULTIPLE)
    return NULL;
  return hwloc_get_next_obj_covering_cpuset_by_depth(topology, set, depth, prev);
}

/** @} */



/** \defgroup hwlocality_helper_ancestors Looking at Ancestor and Child Objects
 * @{
 *
 * Be sure to see the figure in \ref termsanddefs that shows a
 * complete topology tree, including depths, child/sibling/cousin
 * relationships, and an example of an asymmetric topology where one
 * package has fewer caches than its peers.
 */

/** \brief Returns the ancestor object of \p obj at depth \p depth.
 *
 * \note \p depth should not be the depth of PU or NUMA objects
 * since they are ancestors of no objects (except Misc or I/O).
 * This function rather expects an intermediate level depth,
 * such as the depth of Packages, Cores, or Caches.
 */
static __hwloc_inline hwloc_obj_t
hwloc_get_ancestor_obj_by_depth (hwloc_topology_t topology __hwloc_attribute_unused, int depth, hwloc_obj_t obj) __hwloc_attribute_pure;
static __hwloc_inline hwloc_obj_t
hwloc_get_ancestor_obj_by_depth (hwloc_topology_t topology __hwloc_attribute_unused, int depth, hwloc_obj_t obj)
{
  hwloc_obj_t ancestor = obj;
  if (obj->depth < depth)
    return NULL;
  while (ancestor && ancestor->depth > depth)
    ancestor = ancestor->parent;
  return ancestor;
}

/** \brief Returns the ancestor object of \p obj with type \p type.
 *
 * \note \p type should not be ::HWLOC_OBJ_PU or ::HWLOC_OBJ_NUMANODE
 * since these objects are ancestors of no objects (except Misc or I/O).
 * This function rather expects an intermediate object type,
 * such as ::HWLOC_OBJ_PACKAGE, ::HWLOC_OBJ_CORE, etc.
 */
static __hwloc_inline hwloc_obj_t
hwloc_get_ancestor_obj_by_type (hwloc_topology_t topology __hwloc_attribute_unused, hwloc_obj_type_t type, hwloc_obj_t obj) __hwloc_attribute_pure;
static __hwloc_inline hwloc_obj_t
hwloc_get_ancestor_obj_by_type (hwloc_topology_t topology __hwloc_attribute_unused, hwloc_obj_type_t type, hwloc_obj_t obj)
{
  hwloc_obj_t ancestor = obj->parent;
  while (ancestor && ancestor->type != type)
    ancestor = ancestor->parent;
  return ancestor;
}

/** \brief Returns the common parent object to objects \p obj1 and \p obj2 */
static __hwloc_inline hwloc_obj_t
hwloc_get_common_ancestor_obj (hwloc_topology_t topology __hwloc_attribute_unused, hwloc_obj_t obj1, hwloc_obj_t obj2) __hwloc_attribute_pure;
static __hwloc_inline hwloc_obj_t
hwloc_get_common_ancestor_obj (hwloc_topology_t topology __hwloc_attribute_unused, hwloc_obj_t obj1, hwloc_obj_t obj2)
{
  /* the loop isn't so easy since intermediate ancestors may have
   * different depth, causing us to alternate between using obj1->parent
   * and obj2->parent. Also, even if at some point we find ancestors of
   * of the same depth, their ancestors may have different depth again.
   */
  while (obj1 != obj2) {
    while (obj1->depth > obj2->depth)
      obj1 = obj1->parent;
    while (obj2->depth > obj1->depth)
      obj2 = obj2->parent;
    if (obj1 != obj2 && obj1->depth == obj2->depth) {
      obj1 = obj1->parent;
      obj2 = obj2->parent;
    }
  }
  return obj1;
}

/** \brief Returns true if \p obj is inside the subtree beginning with ancestor object \p subtree_root.
 *
 * \note This function cannot work if \p obj and \p subtree_root objects do
 * not have CPU sets (I/O or Misc objects).
 */
static __hwloc_inline int
hwloc_obj_is_in_subtree (hwloc_topology_t topology __hwloc_attribute_unused, hwloc_obj_t obj, hwloc_obj_t subtree_root) __hwloc_attribute_pure;
static __hwloc_inline int
hwloc_obj_is_in_subtree (hwloc_topology_t topology __hwloc_attribute_unused, hwloc_obj_t obj, hwloc_obj_t subtree_root)
{
  return obj->cpuset && subtree_root->cpuset && hwloc_bitmap_isincluded(obj->cpuset, subtree_root->cpuset);
}

/** \brief Return the next child.
 *
 * Return the next child among the normal children list,
 * then among the memory children list, then among the I/O
 * children list, then among the Misc children list.
 *
 * If \p prev is \c NULL, return the first child.
 *
 * Return \c NULL when there is no next child.
 */
static __hwloc_inline hwloc_obj_t
hwloc_get_next_child (hwloc_topology_t topology __hwloc_attribute_unused, hwloc_obj_t parent, hwloc_obj_t prev)
{
  hwloc_obj_t obj;
  int state = 0;
  if (prev) {
    if (prev->type == HWLOC_OBJ_MISC)
      state = 3;
    else if (prev->type == HWLOC_OBJ_BRIDGE || prev->type == HWLOC_OBJ_PCI_DEVICE || prev->type == HWLOC_OBJ_OS_DEVICE)
      state = 2;
    else if (prev->type == HWLOC_OBJ_NUMANODE)
      state = 1;
    obj = prev->next_sibling;
  } else {
    obj = parent->first_child;
  }
  if (!obj && state == 0) {
    obj = parent->memory_first_child;
    state = 1;
  }
  if (!obj && state == 1) {
    obj = parent->io_first_child;
    state = 2;
  }
  if (!obj && state == 2) {
    obj = parent->misc_first_child;
    state = 3;
  }
  return obj;
}

/** @} */



/** \defgroup hwlocality_helper_types Kinds of object Type
 * @{
 *
 * Each object type is
 * either Normal (i.e. hwloc_obj_type_is_normal() returns 1),
 * or Memory (i.e. hwloc_obj_type_is_memory() returns 1)
 * or I/O (i.e. hwloc_obj_type_is_io() returns 1)
 * or Misc (i.e. equal to ::HWLOC_OBJ_MISC).
 * It cannot be of more than one of these kinds.
 */

/** \brief Check whether an object type is Normal.
 *
 * Normal objects are objects of the main CPU hierarchy
 * (Machine, Package, Core, PU, CPU caches, etc.),
 * but they are not NUMA nodes, I/O devices or Misc objects.
 *
 * They are attached to parent as Normal children,
 * not as Memory, I/O or Misc children.
 *
 * \return 1 if an object of type \p type is a Normal object, 0 otherwise.
 */
HWLOC_DECLSPEC int
hwloc_obj_type_is_normal(hwloc_obj_type_t type);

/** \brief Check whether an object type is I/O.
 *
 * I/O objects are objects attached to their parents
 * in the I/O children list.
 * This current includes Bridges, PCI and OS devices.
 *
 * \return 1 if an object of type \p type is a I/O object, 0 otherwise.
 */
HWLOC_DECLSPEC int
hwloc_obj_type_is_io(hwloc_obj_type_t type);

/** \brief Check whether an object type is Memory.
 *
 * Memory objects are objects attached to their parents
 * in the Memory children list.
 * This current includes NUMA nodes and Memory-side caches.
 *
 * \return 1 if an object of type \p type is a Memory object, 0 otherwise.
 */
HWLOC_DECLSPEC int
hwloc_obj_type_is_memory(hwloc_obj_type_t type);

/** \brief Check whether an object type is a CPU Cache (Data, Unified or Instruction).
 *
 * Memory-side caches are not CPU caches.
 *
 * \return 1 if an object of type \p type is a Cache, 0 otherwise.
 */
HWLOC_DECLSPEC int
hwloc_obj_type_is_cache(hwloc_obj_type_t type);

/** \brief Check whether an object type is a CPU Data or Unified Cache.
 *
 * Memory-side caches are not CPU caches.
 *
 * \return 1 if an object of type \p type is a CPU Data or Unified Cache, 0 otherwise.
 */
HWLOC_DECLSPEC int
hwloc_obj_type_is_dcache(hwloc_obj_type_t type);

/** \brief Check whether an object type is a CPU Instruction Cache,
 *
 * Memory-side caches are not CPU caches.
 *
 * \return 1 if an object of type \p type is a CPU Instruction Cache, 0 otherwise.
 */
HWLOC_DECLSPEC int
hwloc_obj_type_is_icache(hwloc_obj_type_t type);

/** @} */



/** \defgroup hwlocality_helper_find_cache Looking at Cache Objects
 * @{
 */

/** \brief Find the depth of cache objects matching cache level and type.
 *
 * Return the depth of the topology level that contains cache objects
 * whose attributes match \p cachelevel and \p cachetype.

 * This function is identical to calling hwloc_get_type_depth() with the
 * corresponding type such as ::HWLOC_OBJ_L1ICACHE, except that it may
 * also return a Unified cache when looking for an instruction cache.
 *
 * If no cache level matches, ::HWLOC_TYPE_DEPTH_UNKNOWN is returned.
 *
 * If \p cachetype is ::HWLOC_OBJ_CACHE_UNIFIED, the depth of the
 * unique matching unified cache level is returned.
 *
 * If \p cachetype is ::HWLOC_OBJ_CACHE_DATA or ::HWLOC_OBJ_CACHE_INSTRUCTION,
 * either a matching cache, or a unified cache is returned.
 *
 * If \p cachetype is \c -1, it is ignored and multiple levels may
 * match. The function returns either the depth of a uniquely matching
 * level or ::HWLOC_TYPE_DEPTH_MULTIPLE.
 */
static __hwloc_inline int
hwloc_get_cache_type_depth (hwloc_topology_t topology,
			    unsigned cachelevel, hwloc_obj_cache_type_t cachetype)
{
  int depth;
  int found = HWLOC_TYPE_DEPTH_UNKNOWN;
  for (depth=0; ; depth++) {
    hwloc_obj_t obj = hwloc_get_obj_by_depth(topology, depth, 0);
    if (!obj)
      break;
    if (!hwloc_obj_type_is_dcache(obj->type) || obj->attr->cache.depth != cachelevel)
      /* doesn't match, try next depth */
      continue;
    if (cachetype == (hwloc_obj_cache_type_t) -1) {
      if (found != HWLOC_TYPE_DEPTH_UNKNOWN) {
	/* second match, return MULTIPLE */
        return HWLOC_TYPE_DEPTH_MULTIPLE;
      }
      /* first match, mark it as found */
      found = depth;
      continue;
    }
    if (obj->attr->cache.type == cachetype || obj->attr->cache.type == HWLOC_OBJ_CACHE_UNIFIED)
      /* exact match (either unified is alone, or we match instruction or data), return immediately */
      return depth;
  }
  /* went to the bottom, return what we found */
  return found;
}

/** \brief Get the first data (or unified) cache covering a cpuset \p set
 *
 * \return \c NULL if no cache matches.
 */
static __hwloc_inline hwloc_obj_t
hwloc_get_cache_covering_cpuset (hwloc_topology_t topology, hwloc_const_cpuset_t set) __hwloc_attribute_pure;
static __hwloc_inline hwloc_obj_t
hwloc_get_cache_covering_cpuset (hwloc_topology_t topology, hwloc_const_cpuset_t set)
{
  hwloc_obj_t current = hwloc_get_obj_covering_cpuset(topology, set);
  while (current) {
    if (hwloc_obj_type_is_dcache(current->type))
      return current;
    current = current->parent;
  }
  return NULL;
}

/** \brief Get the first data (or unified) cache shared between an object and somebody else.
 *
 * \return \c NULL if no cache matches or if an invalid object is given.
 */
static __hwloc_inline hwloc_obj_t
hwloc_get_shared_cache_covering_obj (hwloc_topology_t topology __hwloc_attribute_unused, hwloc_obj_t obj) __hwloc_attribute_pure;
static __hwloc_inline hwloc_obj_t
hwloc_get_shared_cache_covering_obj (hwloc_topology_t topology __hwloc_attribute_unused, hwloc_obj_t obj)
{
  hwloc_obj_t current = obj->parent;
  if (!obj->cpuset)
    return NULL;
  while (current) {
    if (!hwloc_bitmap_isequal(current->cpuset, obj->cpuset)
        && hwloc_obj_type_is_dcache(current->type))
      return current;
    current = current->parent;
  }
  return NULL;
}

/** @} */



/** \defgroup hwlocality_helper_find_misc Finding objects, miscellaneous helpers
 * @{
 *
 * Be sure to see the figure in \ref termsanddefs that shows a
 * complete topology tree, including depths, child/sibling/cousin
 * relationships, and an example of an asymmetric topology where one
 * package has fewer caches than its peers.
 */

/** \brief Returns the object of type ::HWLOC_OBJ_PU with \p os_index.
 *
 * This function is useful for converting a CPU set into the PU
 * objects it contains.
 * When retrieving the current binding (e.g. with hwloc_get_cpubind()),
 * one may iterate over the bits of the resulting CPU set with
 * hwloc_bitmap_foreach_begin(), and find the corresponding PUs
 * with this function.
 */
static __hwloc_inline hwloc_obj_t
hwloc_get_pu_obj_by_os_index(hwloc_topology_t topology, unsigned os_index) __hwloc_attribute_pure;
static __hwloc_inline hwloc_obj_t
hwloc_get_pu_obj_by_os_index(hwloc_topology_t topology, unsigned os_index)
{
  hwloc_obj_t obj = NULL;
  while ((obj = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_PU, obj)) != NULL)
    if (obj->os_index == os_index)
      return obj;
  return NULL;
}

/** \brief Returns the object of type ::HWLOC_OBJ_NUMANODE with \p os_index.
 *
 * This function is useful for converting a nodeset into the NUMA node
 * objects it contains.
 * When retrieving the current binding (e.g. with hwloc_get_membind() with HWLOC_MEMBIND_BYNODESET),
 * one may iterate over the bits of the resulting nodeset with
 * hwloc_bitmap_foreach_begin(), and find the corresponding NUMA nodes
 * with this function.
 */
static __hwloc_inline hwloc_obj_t
hwloc_get_numanode_obj_by_os_index(hwloc_topology_t topology, unsigned os_index) __hwloc_attribute_pure;
static __hwloc_inline hwloc_obj_t
hwloc_get_numanode_obj_by_os_index(hwloc_topology_t topology, unsigned os_index)
{
  hwloc_obj_t obj = NULL;
  while ((obj = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_NUMANODE, obj)) != NULL)
    if (obj->os_index == os_index)
      return obj;
  return NULL;
}

/** \brief Do a depth-first traversal of the topology to find and sort
 *
 * all objects that are at the same depth than \p src.
 * Report in \p objs up to \p max physically closest ones to \p src.
 *
 * \return the number of objects returned in \p objs.
 *
 * \return 0 if \p src is an I/O object.
 *
 * \note This function requires the \p src object to have a CPU set.
 */
/* TODO: rather provide an iterator? Provide a way to know how much should be allocated? By returning the total number of objects instead? */
HWLOC_DECLSPEC unsigned hwloc_get_closest_objs (hwloc_topology_t topology, hwloc_obj_t src, hwloc_obj_t * __hwloc_restrict objs, unsigned max);

/** \brief Find an object below another object, both specified by types and indexes.
 *
 * Start from the top system object and find object of type \p type1
 * and logical index \p idx1.  Then look below this object and find another
 * object of type \p type2 and logical index \p idx2.  Indexes are specified
 * within the parent, not withing the entire system.
 *
 * For instance, if type1 is PACKAGE, idx1 is 2, type2 is CORE and idx2
 * is 3, return the fourth core object below the third package.
 *
 * \note This function requires these objects to have a CPU set.
 */
static __hwloc_inline hwloc_obj_t
hwloc_get_obj_below_by_type (hwloc_topology_t topology,
			     hwloc_obj_type_t type1, unsigned idx1,
			     hwloc_obj_type_t type2, unsigned idx2) __hwloc_attribute_pure;
static __hwloc_inline hwloc_obj_t
hwloc_get_obj_below_by_type (hwloc_topology_t topology,
			     hwloc_obj_type_t type1, unsigned idx1,
			     hwloc_obj_type_t type2, unsigned idx2)
{
  hwloc_obj_t obj;
  obj = hwloc_get_obj_by_type (topology, type1, idx1);
  if (!obj)
    return NULL;
  return hwloc_get_obj_inside_cpuset_by_type(topology, obj->cpuset, type2, idx2);
}

/** \brief Find an object below a chain of objects specified by types and indexes.
 *
 * This is a generalized version of hwloc_get_obj_below_by_type().
 *
 * Arrays \p typev and \p idxv must contain \p nr types and indexes.
 *
 * Start from the top system object and walk the arrays \p typev and \p idxv.
 * For each type and logical index couple in the arrays, look under the previously found
 * object to find the index-th object of the given type.
 * Indexes are specified within the parent, not withing the entire system.
 *
 * For instance, if nr is 3, typev contains NODE, PACKAGE and CORE,
 * and idxv contains 0, 1 and 2, return the third core object below
 * the second package below the first NUMA node.
 *
 * \note This function requires all these objects and the root object
 * to have a CPU set.
 */
static __hwloc_inline hwloc_obj_t
hwloc_get_obj_below_array_by_type (hwloc_topology_t topology, int nr, hwloc_obj_type_t *typev, unsigned *idxv) __hwloc_attribute_pure;
static __hwloc_inline hwloc_obj_t
hwloc_get_obj_below_array_by_type (hwloc_topology_t topology, int nr, hwloc_obj_type_t *typev, unsigned *idxv)
{
  hwloc_obj_t obj = hwloc_get_root_obj(topology);
  int i;
  for(i=0; i<nr; i++) {
    if (!obj)
      return NULL;
    obj = hwloc_get_obj_inside_cpuset_by_type(topology, obj->cpuset, typev[i], idxv[i]);
  }
  return obj;
}

/** @} */



/** \defgroup hwlocality_helper_distribute Distributing items over a topology
 * @{
 */

/** \brief Flags to be given to hwloc_distrib().
 */
enum hwloc_distrib_flags_e {
  /** \brief Distrib in reverse order, starting from the last objects.
   * \hideinitializer
   */
  HWLOC_DISTRIB_FLAG_REVERSE = (1UL<<0)
};

/** \brief Distribute \p n items over the topology under \p roots
 *
 * Array \p set will be filled with \p n cpusets recursively distributed
 * linearly over the topology under objects \p roots, down to depth \p until
 * (which can be INT_MAX to distribute down to the finest level).
 *
 * \p n_roots is usually 1 and \p roots only contains the topology root object
 * so as to distribute over the entire topology.
 *
 * This is typically useful when an application wants to distribute \p n
 * threads over a machine, giving each of them as much private cache as
 * possible and keeping them locally in number order.
 *
 * The caller may typically want to also call hwloc_bitmap_singlify()
 * before binding a thread so that it does not move at all.
 *
 * \p flags should be 0 or a OR'ed set of ::hwloc_distrib_flags_e.
 *
 * \note This function requires the \p roots objects to have a CPU set.
 *
 * \note This function replaces the now deprecated hwloc_distribute()
 * and hwloc_distributev() functions.
 */
static __hwloc_inline int
hwloc_distrib(hwloc_topology_t topology,
	      hwloc_obj_t *roots, unsigned n_roots,
	      hwloc_cpuset_t *set,
	      unsigned n,
	      int until, unsigned long flags)
{
  unsigned i;
  unsigned tot_weight;
  unsigned given, givenweight;
  hwloc_cpuset_t *cpusetp = set;

  if (flags & ~HWLOC_DISTRIB_FLAG_REVERSE) {
    errno = EINVAL;
    return -1;
  }

  tot_weight = 0;
  for (i = 0; i < n_roots; i++)
    tot_weight += (unsigned) hwloc_bitmap_weight(roots[i]->cpuset);

  for (i = 0, given = 0, givenweight = 0; i < n_roots; i++) {
    unsigned chunk, weight;
    hwloc_obj_t root = roots[flags & HWLOC_DISTRIB_FLAG_REVERSE ? n_roots-1-i : i];
    hwloc_cpuset_t cpuset = root->cpuset;
    if (root->type == HWLOC_OBJ_NUMANODE)
      /* NUMANodes have same cpuset as their parent, but we need normal objects below */
      root = root->parent;
    weight = (unsigned) hwloc_bitmap_weight(cpuset);
    if (!weight)
      continue;
    /* Give to root a chunk proportional to its weight.
     * If previous chunks got rounded-up, we may get a bit less. */
    chunk = (( (givenweight+weight) * n  + tot_weight-1) / tot_weight)
          - ((  givenweight         * n  + tot_weight-1) / tot_weight);
    if (!root->arity || chunk <= 1 || root->depth >= until) {
      /* We can't split any more, put everything there.  */
      if (chunk) {
	/* Fill cpusets with ours */
	unsigned j;
	for (j=0; j < chunk; j++)
	  cpusetp[j] = hwloc_bitmap_dup(cpuset);
      } else {
	/* We got no chunk, just merge our cpuset to a previous one
	 * (the first chunk cannot be empty)
	 * so that this root doesn't get ignored.
	 */
	assert(given);
	hwloc_bitmap_or(cpusetp[-1], cpusetp[-1], cpuset);
      }
    } else {
      /* Still more to distribute, recurse into children */
      hwloc_distrib(topology, root->children, root->arity, cpusetp, chunk, until, flags);
    }
    cpusetp += chunk;
    given += chunk;
    givenweight += weight;
  }

  return 0;
}

/** @} */



/** \defgroup hwlocality_helper_topology_sets CPU and node sets of entire topologies
 * @{
 */

/** \brief Get complete CPU set
 *
 * \return the complete CPU set of logical processors of the system.
 *
 * \note The returned cpuset is not newly allocated and should thus not be
 * changed or freed; hwloc_bitmap_dup() must be used to obtain a local copy.
 *
 * \note This is equivalent to retrieving the root object complete CPU-set.
 */
HWLOC_DECLSPEC hwloc_const_cpuset_t
hwloc_topology_get_complete_cpuset(hwloc_topology_t topology) __hwloc_attribute_pure;

/** \brief Get topology CPU set
 *
 * \return the CPU set of logical processors of the system for which hwloc
 * provides topology information. This is equivalent to the cpuset of the
 * system object.
 *
 * \note The returned cpuset is not newly allocated and should thus not be
 * changed or freed; hwloc_bitmap_dup() must be used to obtain a local copy.
 *
 * \note This is equivalent to retrieving the root object CPU-set.
 */
HWLOC_DECLSPEC hwloc_const_cpuset_t
hwloc_topology_get_topology_cpuset(hwloc_topology_t topology) __hwloc_attribute_pure;

/** \brief Get allowed CPU set
 *
 * \return the CPU set of allowed logical processors of the system.
 *
 * \note If the topology flag ::HWLOC_TOPOLOGY_FLAG_INCLUDE_DISALLOWED was not set,
 * this is identical to hwloc_topology_get_topology_cpuset(), which means
 * all PUs are allowed.
 *
 * \note If ::HWLOC_TOPOLOGY_FLAG_INCLUDE_DISALLOWED was set, applying
 * hwloc_bitmap_intersects() on the result of this function and on an object
 * cpuset checks whether there are allowed PUs inside that object.
 * Applying hwloc_bitmap_and() returns the list of these allowed PUs.
 *
 * \note The returned cpuset is not newly allocated and should thus not be
 * changed or freed, hwloc_bitmap_dup() must be used to obtain a local copy.
 */
HWLOC_DECLSPEC hwloc_const_cpuset_t
hwloc_topology_get_allowed_cpuset(hwloc_topology_t topology) __hwloc_attribute_pure;

/** \brief Get complete node set
 *
 * \return the complete node set of memory of the system.
 *
 * \note The returned nodeset is not newly allocated and should thus not be
 * changed or freed; hwloc_bitmap_dup() must be used to obtain a local copy.
 *
 * \note This is equivalent to retrieving the root object complete nodeset.
 */
HWLOC_DECLSPEC hwloc_const_nodeset_t
hwloc_topology_get_complete_nodeset(hwloc_topology_t topology) __hwloc_attribute_pure;

/** \brief Get topology node set
 *
 * \return the node set of memory of the system for which hwloc
 * provides topology information. This is equivalent to the nodeset of the
 * system object.
 *
 * \note The returned nodeset is not newly allocated and should thus not be
 * changed or freed; hwloc_bitmap_dup() must be used to obtain a local copy.
 *
 * \note This is equivalent to retrieving the root object nodeset.
 */
HWLOC_DECLSPEC hwloc_const_nodeset_t
hwloc_topology_get_topology_nodeset(hwloc_topology_t topology) __hwloc_attribute_pure;

/** \brief Get allowed node set
 *
 * \return the node set of allowed memory of the system.
 *
 * \note If the topology flag ::HWLOC_TOPOLOGY_FLAG_INCLUDE_DISALLOWED was not set,
 * this is identical to hwloc_topology_get_topology_nodeset(), which means
 * all NUMA nodes are allowed.
 *
 * \note If ::HWLOC_TOPOLOGY_FLAG_INCLUDE_DISALLOWED was set, applying
 * hwloc_bitmap_intersects() on the result of this function and on an object
 * nodeset checks whether there are allowed NUMA nodes inside that object.
 * Applying hwloc_bitmap_and() returns the list of these allowed NUMA nodes.
 *
 * \note The returned nodeset is not newly allocated and should thus not be
 * changed or freed, hwloc_bitmap_dup() must be used to obtain a local copy.
 */
HWLOC_DECLSPEC hwloc_const_nodeset_t
hwloc_topology_get_allowed_nodeset(hwloc_topology_t topology) __hwloc_attribute_pure;

/** @} */



/** \defgroup hwlocality_helper_nodeset_convert Converting between CPU sets and node sets
 *
 * @{
 */

/** \brief Convert a CPU set into a NUMA node set and handle non-NUMA cases
 *
 * If some NUMA nodes have no CPUs at all, this function never sets their
 * indexes in the output node set, even if a full CPU set is given in input.
 *
 * If the topology contains no NUMA nodes, the machine is considered
 * as a single memory node, and the following behavior is used:
 * If \p cpuset is empty, \p nodeset will be emptied as well.
 * Otherwise \p nodeset will be entirely filled.
 */
static __hwloc_inline int
hwloc_cpuset_to_nodeset(hwloc_topology_t topology, hwloc_const_cpuset_t _cpuset, hwloc_nodeset_t nodeset)
{
	int depth = hwloc_get_type_depth(topology, HWLOC_OBJ_NUMANODE);
	hwloc_obj_t obj = NULL;
	assert(depth != HWLOC_TYPE_DEPTH_UNKNOWN);
	hwloc_bitmap_zero(nodeset);
	while ((obj = hwloc_get_next_obj_covering_cpuset_by_depth(topology, _cpuset, depth, obj)) != NULL)
		if (hwloc_bitmap_set(nodeset, obj->os_index) < 0)
			return -1;
	return 0;
}

/** \brief Convert a NUMA node set into a CPU set and handle non-NUMA cases
 *
 * If the topology contains no NUMA nodes, the machine is considered
 * as a single memory node, and the following behavior is used:
 * If \p nodeset is empty, \p cpuset will be emptied as well.
 * Otherwise \p cpuset will be entirely filled.
 * This is useful for manipulating memory binding sets.
 */
static __hwloc_inline int
hwloc_cpuset_from_nodeset(hwloc_topology_t topology, hwloc_cpuset_t _cpuset, hwloc_const_nodeset_t nodeset)
{
	int depth = hwloc_get_type_depth(topology, HWLOC_OBJ_NUMANODE);
	hwloc_obj_t obj = NULL;
	assert(depth != HWLOC_TYPE_DEPTH_UNKNOWN);
	hwloc_bitmap_zero(_cpuset);
	while ((obj = hwloc_get_next_obj_by_depth(topology, depth, obj)) != NULL) {
		if (hwloc_bitmap_isset(nodeset, obj->os_index))
			/* no need to check obj->cpuset because objects in levels always have a cpuset */
			if (hwloc_bitmap_or(_cpuset, _cpuset, obj->cpuset) < 0)
				return -1;
	}
	return 0;
}

/** @} */



/** \defgroup hwlocality_advanced_io Finding I/O objects
 * @{
 */

/** \brief Get the first non-I/O ancestor object.
 *
 * Given the I/O object \p ioobj, find the smallest non-I/O ancestor
 * object. This object (normal or memory) may then be used for binding
 * because it has non-NULL CPU and node sets
 * and because its locality is the same as \p ioobj.
 *
 * \note The resulting object is usually a normal object but it could also
 * be a memory object (e.g. NUMA node) in future platforms if I/O objects
 * ever get attached to memory instead of CPUs.
 */
static __hwloc_inline hwloc_obj_t
hwloc_get_non_io_ancestor_obj(hwloc_topology_t topology __hwloc_attribute_unused,
			      hwloc_obj_t ioobj)
{
  hwloc_obj_t obj = ioobj;
  while (obj && !obj->cpuset) {
    obj = obj->parent;
  }
  return obj;
}

/** \brief Get the next PCI device in the system.
 *
 * \return the first PCI device if \p prev is \c NULL.
 */
static __hwloc_inline hwloc_obj_t
hwloc_get_next_pcidev(hwloc_topology_t topology, hwloc_obj_t prev)
{
  return hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_PCI_DEVICE, prev);
}

/** \brief Find the PCI device object matching the PCI bus id
 * given domain, bus device and function PCI bus id.
 */
static __hwloc_inline hwloc_obj_t
hwloc_get_pcidev_by_busid(hwloc_topology_t topology,
			  unsigned domain, unsigned bus, unsigned dev, unsigned func)
{
  hwloc_obj_t obj = NULL;
  while ((obj = hwloc_get_next_pcidev(topology, obj)) != NULL) {
    if (obj->attr->pcidev.domain == domain
	&& obj->attr->pcidev.bus == bus
	&& obj->attr->pcidev.dev == dev
	&& obj->attr->pcidev.func == func)
      return obj;
  }
  return NULL;
}

/** \brief Find the PCI device object matching the PCI bus id
 * given as a string xxxx:yy:zz.t or yy:zz.t.
 */
static __hwloc_inline hwloc_obj_t
hwloc_get_pcidev_by_busidstring(hwloc_topology_t topology, const char *busid)
{
  unsigned domain = 0; /* default */
  unsigned bus, dev, func;

  if (sscanf(busid, "%x:%x.%x", &bus, &dev, &func) != 3
      && sscanf(busid, "%x:%x:%x.%x", &domain, &bus, &dev, &func) != 4) {
    errno = EINVAL;
    return NULL;
  }

  return hwloc_get_pcidev_by_busid(topology, domain, bus, dev, func);
}

/** \brief Get the next OS device in the system.
 *
 * \return the first OS device if \p prev is \c NULL.
 */
static __hwloc_inline hwloc_obj_t
hwloc_get_next_osdev(hwloc_topology_t topology, hwloc_obj_t prev)
{
  return hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_OS_DEVICE, prev);
}

/** \brief Get the next bridge in the system.
 *
 * \return the first bridge if \p prev is \c NULL.
 */
static __hwloc_inline hwloc_obj_t
hwloc_get_next_bridge(hwloc_topology_t topology, hwloc_obj_t prev)
{
  return hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_BRIDGE, prev);
}

/* \brief Checks whether a given bridge covers a given PCI bus.
 */
static __hwloc_inline int
hwloc_bridge_covers_pcibus(hwloc_obj_t bridge,
			   unsigned domain, unsigned bus)
{
  return bridge->type == HWLOC_OBJ_BRIDGE
    && bridge->attr->bridge.downstream_type == HWLOC_OBJ_BRIDGE_PCI
    && bridge->attr->bridge.downstream.pci.domain == domain
    && bridge->attr->bridge.downstream.pci.secondary_bus <= bus
    && bridge->attr->bridge.downstream.pci.subordinate_bus >= bus;
}

/** @} */



#ifdef __cplusplus
} /* extern "C" */
#endif


#endif /* HWLOC_HELPER_H */
