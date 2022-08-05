/*
 * Copyright © 2009 CNRS
 * Copyright © 2009-2018 Inria.  All rights reserved.
 * Copyright © 2009-2012 Université Bordeaux
 * Copyright © 2009-2010 Cisco Systems, Inc.  All rights reserved.
 * See COPYING in top-level directory.
 */

/**
 * This file contains the inline code of functions declared in hwloc.h
 */

#ifndef HWLOC_INLINES_H
#define HWLOC_INLINES_H

#ifndef HWLOC_H
#error Please include the main hwloc.h instead
#endif

#include <stdlib.h>
#include <errno.h>


#ifdef __cplusplus
extern "C" {
#endif

static __hwloc_inline int
hwloc_get_type_or_below_depth (hwloc_topology_t topology, hwloc_obj_type_t type)
{
  int depth = hwloc_get_type_depth(topology, type);

  if (depth != HWLOC_TYPE_DEPTH_UNKNOWN)
    return depth;

  /* find the highest existing level with type order >= */
  for(depth = hwloc_get_type_depth(topology, HWLOC_OBJ_PU); ; depth--)
    if (hwloc_compare_types(hwloc_get_depth_type(topology, depth), type) < 0)
      return depth+1;

  /* Shouldn't ever happen, as there is always a Machine level with lower order and known depth.  */
  /* abort(); */
}

static __hwloc_inline int
hwloc_get_type_or_above_depth (hwloc_topology_t topology, hwloc_obj_type_t type)
{
  int depth = hwloc_get_type_depth(topology, type);

  if (depth != HWLOC_TYPE_DEPTH_UNKNOWN)
    return depth;

  /* find the lowest existing level with type order <= */
  for(depth = 0; ; depth++)
    if (hwloc_compare_types(hwloc_get_depth_type(topology, depth), type) > 0)
      return depth-1;

  /* Shouldn't ever happen, as there is always a PU level with higher order and known depth.  */
  /* abort(); */
}

static __hwloc_inline int
hwloc_get_nbobjs_by_type (hwloc_topology_t topology, hwloc_obj_type_t type)
{
  int depth = hwloc_get_type_depth(topology, type);
  if (depth == HWLOC_TYPE_DEPTH_UNKNOWN)
    return 0;
  if (depth == HWLOC_TYPE_DEPTH_MULTIPLE)
    return -1; /* FIXME: agregate nbobjs from different levels? */
  return (int) hwloc_get_nbobjs_by_depth(topology, depth);
}

static __hwloc_inline hwloc_obj_t
hwloc_get_obj_by_type (hwloc_topology_t topology, hwloc_obj_type_t type, unsigned idx)
{
  int depth = hwloc_get_type_depth(topology, type);
  if (depth == HWLOC_TYPE_DEPTH_UNKNOWN)
    return NULL;
  if (depth == HWLOC_TYPE_DEPTH_MULTIPLE)
    return NULL;
  return hwloc_get_obj_by_depth(topology, depth, idx);
}

static __hwloc_inline hwloc_obj_t
hwloc_get_next_obj_by_depth (hwloc_topology_t topology, int depth, hwloc_obj_t prev)
{
  if (!prev)
    return hwloc_get_obj_by_depth (topology, depth, 0);
  if (prev->depth != depth)
    return NULL;
  return prev->next_cousin;
}

static __hwloc_inline hwloc_obj_t
hwloc_get_next_obj_by_type (hwloc_topology_t topology, hwloc_obj_type_t type,
			    hwloc_obj_t prev)
{
  int depth = hwloc_get_type_depth(topology, type);
  if (depth == HWLOC_TYPE_DEPTH_UNKNOWN || depth == HWLOC_TYPE_DEPTH_MULTIPLE)
    return NULL;
  return hwloc_get_next_obj_by_depth (topology, depth, prev);
}

static __hwloc_inline hwloc_obj_t
hwloc_get_root_obj (hwloc_topology_t topology)
{
  return hwloc_get_obj_by_depth (topology, 0, 0);
}

static __hwloc_inline const char *
hwloc_obj_get_info_by_name(hwloc_obj_t obj, const char *name)
{
  unsigned i;
  for(i=0; i<obj->infos_count; i++) {
    struct hwloc_info_s *info = &obj->infos[i];
    if (!strcmp(info->name, name))
      return info->value;
  }
  return NULL;
}

static __hwloc_inline void *
hwloc_alloc_membind_policy(hwloc_topology_t topology, size_t len, hwloc_const_cpuset_t set, hwloc_membind_policy_t policy, int flags)
{
  void *p = hwloc_alloc_membind(topology, len, set, policy, flags);
  if (p)
    return p;

  if (hwloc_set_membind(topology, set, policy, flags) < 0)
    /* hwloc_set_membind() takes care of ignoring errors if non-STRICT */
    return NULL;

  p = hwloc_alloc(topology, len);
  if (p && policy != HWLOC_MEMBIND_FIRSTTOUCH)
    /* Enforce the binding by touching the data */
    memset(p, 0, len);
  return p;
}


#ifdef __cplusplus
} /* extern "C" */
#endif


#endif /* HWLOC_INLINES_H */
