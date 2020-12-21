/*
 * Copyright © 2009 CNRS
 * Copyright © 2009-2020 Inria.  All rights reserved.
 * Copyright © 2009-2010, 2020 Université Bordeaux
 * Copyright © 2009-2011 Cisco Systems, Inc.  All rights reserved.
 * See COPYING in top-level directory.
 */

#include "private/autogen/config.h"
#include "hwloc.h"
#include "private/private.h"
#include "private/misc.h"
#include "private/debug.h"

#ifdef HAVE_STRINGS_H
#include <strings.h>
#endif /* HAVE_STRINGS_H */

int
hwloc_get_type_depth (struct hwloc_topology *topology, hwloc_obj_type_t type)
{
  HWLOC_BUILD_ASSERT(HWLOC_OBJ_TYPE_MIN == 0);
  if ((unsigned) type >= HWLOC_OBJ_TYPE_MAX)
    return HWLOC_TYPE_DEPTH_UNKNOWN;
  else
    return topology->type_depth[type];
}

hwloc_obj_type_t
hwloc_get_depth_type (hwloc_topology_t topology, int depth)
{
  if ((unsigned)depth >= topology->nb_levels)
    switch (depth) {
    case HWLOC_TYPE_DEPTH_NUMANODE:
      return HWLOC_OBJ_NUMANODE;
    case HWLOC_TYPE_DEPTH_BRIDGE:
      return HWLOC_OBJ_BRIDGE;
    case HWLOC_TYPE_DEPTH_PCI_DEVICE:
      return HWLOC_OBJ_PCI_DEVICE;
    case HWLOC_TYPE_DEPTH_OS_DEVICE:
      return HWLOC_OBJ_OS_DEVICE;
    case HWLOC_TYPE_DEPTH_MISC:
      return HWLOC_OBJ_MISC;
    case HWLOC_TYPE_DEPTH_MEMCACHE:
      return HWLOC_OBJ_MEMCACHE;
    default:
      return HWLOC_OBJ_TYPE_NONE;
    }
  return topology->levels[depth][0]->type;
}

int
hwloc_get_memory_parents_depth (hwloc_topology_t topology)
{
  int depth = HWLOC_TYPE_DEPTH_UNKNOWN;
  /* memory leaves are always NUMA nodes for now, no need to check parents of other memory types */
  hwloc_obj_t numa = hwloc_get_obj_by_depth(topology, HWLOC_TYPE_DEPTH_NUMANODE, 0);
  assert(numa);
  while (numa) {
    hwloc_obj_t parent = numa->parent;
    /* walk-up the memory hierarchy */
    while (hwloc__obj_type_is_memory(parent->type))
      parent = parent->parent;

    if (depth == HWLOC_TYPE_DEPTH_UNKNOWN)
      depth = parent->depth;
    else if (depth != parent->depth)
      return HWLOC_TYPE_DEPTH_MULTIPLE;

    numa = numa->next_cousin;
  }

  assert(depth >= 0);
  return depth;
}

unsigned
hwloc_get_nbobjs_by_depth (struct hwloc_topology *topology, int depth)
{
  if ((unsigned)depth >= topology->nb_levels) {
    unsigned l = HWLOC_SLEVEL_FROM_DEPTH(depth);
    if (l < HWLOC_NR_SLEVELS)
      return topology->slevels[l].nbobjs;
    else
      return 0;
  }
  return topology->level_nbobjects[depth];
}

struct hwloc_obj *
hwloc_get_obj_by_depth (struct hwloc_topology *topology, int depth, unsigned idx)
{
  if ((unsigned)depth >= topology->nb_levels) {
    unsigned l = HWLOC_SLEVEL_FROM_DEPTH(depth);
    if (l < HWLOC_NR_SLEVELS)
      return idx < topology->slevels[l].nbobjs ? topology->slevels[l].objs[idx] : NULL;
    else
      return NULL;
  }
  if (idx >= topology->level_nbobjects[depth])
    return NULL;
  return topology->levels[depth][idx];
}

int
hwloc_obj_type_is_normal(hwloc_obj_type_t type)
{
  return hwloc__obj_type_is_normal(type);
}

int
hwloc_obj_type_is_memory(hwloc_obj_type_t type)
{
  return hwloc__obj_type_is_memory(type);
}

int
hwloc_obj_type_is_io(hwloc_obj_type_t type)
{
  return hwloc__obj_type_is_io(type);
}

int
hwloc_obj_type_is_cache(hwloc_obj_type_t type)
{
  return hwloc__obj_type_is_cache(type);
}

int
hwloc_obj_type_is_dcache(hwloc_obj_type_t type)
{
  return hwloc__obj_type_is_dcache(type);
}

int
hwloc_obj_type_is_icache(hwloc_obj_type_t type)
{
  return hwloc__obj_type_is_icache(type);
}

static hwloc_obj_t hwloc_get_obj_by_depth_and_gp_index(hwloc_topology_t topology, unsigned depth, uint64_t gp_index)
{
  hwloc_obj_t obj = hwloc_get_obj_by_depth(topology, depth, 0);
  while (obj) {
    if (obj->gp_index == gp_index)
      return obj;
    obj = obj->next_cousin;
  }
  return NULL;
}

hwloc_obj_t hwloc_get_obj_by_type_and_gp_index(hwloc_topology_t topology, hwloc_obj_type_t type, uint64_t gp_index)
{
  int depth = hwloc_get_type_depth(topology, type);
  if (depth == HWLOC_TYPE_DEPTH_UNKNOWN)
    return NULL;
  if (depth == HWLOC_TYPE_DEPTH_MULTIPLE) {
    for(depth=1 /* no multiple machine levels */;
	(unsigned) depth < topology->nb_levels-1 /* no multiple PU levels */;
	depth++) {
      if (hwloc_get_depth_type(topology, depth) == type) {
	hwloc_obj_t obj = hwloc_get_obj_by_depth_and_gp_index(topology, depth, gp_index);
	if (obj)
	  return obj;
      }
    }
    return NULL;
  }
  return hwloc_get_obj_by_depth_and_gp_index(topology, depth, gp_index);
}

unsigned hwloc_get_closest_objs (struct hwloc_topology *topology, struct hwloc_obj *src, struct hwloc_obj **objs, unsigned max)
{
  struct hwloc_obj *parent, *nextparent, **src_objs;
  unsigned i,src_nbobjects;
  unsigned stored = 0;

  if (!src->cpuset)
    return 0;

  src_nbobjects = topology->level_nbobjects[src->depth];
  src_objs = topology->levels[src->depth];

  parent = src;
  while (stored < max) {
    while (1) {
      nextparent = parent->parent;
      if (!nextparent)
	goto out;
      if (!hwloc_bitmap_isequal(parent->cpuset, nextparent->cpuset))
	break;
      parent = nextparent;
    }

    /* traverse src's objects and find those that are in nextparent and were not in parent */
    for(i=0; i<src_nbobjects; i++) {
      if (hwloc_bitmap_isincluded(src_objs[i]->cpuset, nextparent->cpuset)
	  && !hwloc_bitmap_isincluded(src_objs[i]->cpuset, parent->cpuset)) {
	objs[stored++] = src_objs[i];
	if (stored == max)
	  goto out;
      }
    }
    parent = nextparent;
  }

 out:
  return stored;
}

static int
hwloc__get_largest_objs_inside_cpuset (struct hwloc_obj *current, hwloc_const_bitmap_t set,
				       struct hwloc_obj ***res, int *max)
{
  int gotten = 0;
  unsigned i;

  /* the caller must ensure this */
  if (*max <= 0)
    return 0;

  if (hwloc_bitmap_isequal(current->cpuset, set)) {
    **res = current;
    (*res)++;
    (*max)--;
    return 1;
  }

  for (i=0; i<current->arity; i++) {
    hwloc_bitmap_t subset;
    int ret;

    /* split out the cpuset part corresponding to this child and see if there's anything to do */
    if (!hwloc_bitmap_intersects(set,current->children[i]->cpuset))
      continue;

    subset = hwloc_bitmap_dup(set);
    hwloc_bitmap_and(subset, subset, current->children[i]->cpuset);
    ret = hwloc__get_largest_objs_inside_cpuset (current->children[i], subset, res, max);
    gotten += ret;
    hwloc_bitmap_free(subset);

    /* if no more room to store remaining objects, return what we got so far */
    if (!*max)
      break;
  }

  return gotten;
}

int
hwloc_get_largest_objs_inside_cpuset (struct hwloc_topology *topology, hwloc_const_bitmap_t set,
				      struct hwloc_obj **objs, int max)
{
  struct hwloc_obj *current = topology->levels[0][0];

  if (!hwloc_bitmap_isincluded(set, current->cpuset))
    return -1;

  if (max <= 0)
    return 0;

  return hwloc__get_largest_objs_inside_cpuset (current, set, &objs, &max);
}

const char *
hwloc_obj_type_string (hwloc_obj_type_t obj)
{
  switch (obj)
    {
    case HWLOC_OBJ_MACHINE: return "Machine";
    case HWLOC_OBJ_MISC: return "Misc";
    case HWLOC_OBJ_GROUP: return "Group";
    case HWLOC_OBJ_MEMCACHE: return "MemCache";
    case HWLOC_OBJ_NUMANODE: return "NUMANode";
    case HWLOC_OBJ_PACKAGE: return "Package";
    case HWLOC_OBJ_DIE: return "Die";
    case HWLOC_OBJ_L1CACHE: return "L1Cache";
    case HWLOC_OBJ_L2CACHE: return "L2Cache";
    case HWLOC_OBJ_L3CACHE: return "L3Cache";
    case HWLOC_OBJ_L4CACHE: return "L4Cache";
    case HWLOC_OBJ_L5CACHE: return "L5Cache";
    case HWLOC_OBJ_L1ICACHE: return "L1iCache";
    case HWLOC_OBJ_L2ICACHE: return "L2iCache";
    case HWLOC_OBJ_L3ICACHE: return "L3iCache";
    case HWLOC_OBJ_CORE: return "Core";
    case HWLOC_OBJ_BRIDGE: return "Bridge";
    case HWLOC_OBJ_PCI_DEVICE: return "PCIDev";
    case HWLOC_OBJ_OS_DEVICE: return "OSDev";
    case HWLOC_OBJ_PU: return "PU";
    default: return "Unknown";
    }
}

/* Check if string matches the given type at least on minmatch chars.
 * On success, return the address of where matching stop, either pointing to \0 or to a suffix (digits, colon, etc)
 * On error, return NULL;
 */
static __hwloc_inline const char *
hwloc__type_match(const char *string,
		  const char *type, /* type must be lowercase */
		  size_t minmatch)
{
  const char *s, *t;
  unsigned i;
  for(i=0, s=string, t=type; ; i++, s++, t++) {
    if (!*s) {
      /* string ends before type */
      if (i<minmatch)
	return NULL;
      else
	return s;
    }
    if (*s != *t && *s != *t + 'A' - 'a') {
      /* string is different */
      if ((*s >= 'a' && *s <= 'z') || (*s >= 'A' && *s <= 'Z') || *s == '-')
	/* valid character that doesn't match */
	return NULL;
      /* invalid character, we reached the end of the type namein string, stop matching here */
      if (i<minmatch)
	return NULL;
      else
	return s;
    }
  }

  return NULL;
}

int
hwloc_type_sscanf(const char *string, hwloc_obj_type_t *typep,
		  union hwloc_obj_attr_u *attrp, size_t attrsize)
{
  hwloc_obj_type_t type = (hwloc_obj_type_t) -1;
  unsigned depthattr = (unsigned) -1;
  hwloc_obj_cache_type_t cachetypeattr = (hwloc_obj_cache_type_t) -1; /* unspecified */
  hwloc_obj_bridge_type_t ubtype = (hwloc_obj_bridge_type_t) -1;
  hwloc_obj_osdev_type_t ostype = (hwloc_obj_osdev_type_t) -1;
  char *end;

  /* Never match the ending \0 since we want to match things like core:2 too.
   * We'll only compare the beginning substring only made of letters and dash.
   */

  /* types without a custom depth */

  /* osdev subtype first to avoid conflicts coproc/core etc */
  if (hwloc__type_match(string, "osdev", 2)) {
    type = HWLOC_OBJ_OS_DEVICE;
  } else if (hwloc__type_match(string, "block", 4)) {
    type = HWLOC_OBJ_OS_DEVICE;
    ostype = HWLOC_OBJ_OSDEV_BLOCK;
  } else if (hwloc__type_match(string, "network", 3)) {
    type = HWLOC_OBJ_OS_DEVICE;
    ostype = HWLOC_OBJ_OSDEV_NETWORK;
  } else if (hwloc__type_match(string, "openfabrics", 7)) {
    type = HWLOC_OBJ_OS_DEVICE;
    ostype = HWLOC_OBJ_OSDEV_OPENFABRICS;
  } else if (hwloc__type_match(string, "dma", 3)) {
    type = HWLOC_OBJ_OS_DEVICE;
    ostype = HWLOC_OBJ_OSDEV_DMA;
  } else if (hwloc__type_match(string, "gpu", 3)) {
    type = HWLOC_OBJ_OS_DEVICE;
    ostype = HWLOC_OBJ_OSDEV_GPU;
  } else if (hwloc__type_match(string, "coproc", 5)
	     || hwloc__type_match(string, "co-processor", 6)) {
    type = HWLOC_OBJ_OS_DEVICE;
    ostype = HWLOC_OBJ_OSDEV_COPROC;

  } else if (hwloc__type_match(string, "machine", 2)) {
    type = HWLOC_OBJ_MACHINE;
  } else if (hwloc__type_match(string, "numanode", 2)
	     || hwloc__type_match(string, "node", 2)) { /* for convenience */
    type = HWLOC_OBJ_NUMANODE;
  } else if (hwloc__type_match(string, "memcache", 5)
	     || hwloc__type_match(string, "memory-side cache", 8)) {
    type = HWLOC_OBJ_MEMCACHE;
  } else if (hwloc__type_match(string, "package", 2)
	     || hwloc__type_match(string, "socket", 2)) { /* backward compat with v1.10 */
    type = HWLOC_OBJ_PACKAGE;
  } else if (hwloc__type_match(string, "die", 2)) {
    type = HWLOC_OBJ_DIE;
  } else if (hwloc__type_match(string, "core", 2)) {
    type = HWLOC_OBJ_CORE;
  } else if (hwloc__type_match(string, "pu", 2)) {
    type = HWLOC_OBJ_PU;
  } else if (hwloc__type_match(string, "misc", 4)) {
    type = HWLOC_OBJ_MISC;

  } else if (hwloc__type_match(string, "bridge", 4)) {
    type = HWLOC_OBJ_BRIDGE;
  } else if (hwloc__type_match(string, "hostbridge", 6)) {
    type = HWLOC_OBJ_BRIDGE;
    ubtype = HWLOC_OBJ_BRIDGE_HOST;
  } else if (hwloc__type_match(string, "pcibridge", 5)) {
    type = HWLOC_OBJ_BRIDGE;
    ubtype = HWLOC_OBJ_BRIDGE_PCI;

  } else if (hwloc__type_match(string, "pcidev", 3)) {
    type = HWLOC_OBJ_PCI_DEVICE;

  /* types with depthattr */
  } else if ((string[0] == 'l' || string[0] == 'L') && string[1] >= '0' && string[1] <= '9') {
    char *suffix;
    depthattr = strtol(string+1, &end, 10);
    if (*end == 'i' || *end == 'I') {
      if (depthattr >= 1 && depthattr <= 3) {
	type = HWLOC_OBJ_L1ICACHE + depthattr-1;
	cachetypeattr = HWLOC_OBJ_CACHE_INSTRUCTION;
	suffix = end+1;
      } else
	return -1;
    } else {
      if (depthattr >= 1 && depthattr <= 5) {
	type = HWLOC_OBJ_L1CACHE + depthattr-1;
	if (*end == 'd' || *end == 'D') {
	  cachetypeattr = HWLOC_OBJ_CACHE_DATA;
	  suffix = end+1;
	} else if (*end == 'u' || *end == 'U') {
	  cachetypeattr = HWLOC_OBJ_CACHE_UNIFIED;
	  suffix = end+1;
	} else {
	  cachetypeattr = HWLOC_OBJ_CACHE_UNIFIED;
	  suffix = end;
	}
      } else
	return -1;
    }
    /* check whether the optional suffix matches "cache" */
    if (!hwloc__type_match(suffix, "cache", 0))
      return -1;

  } else if ((end = (char *) hwloc__type_match(string, "group", 2)) != NULL) {
    type = HWLOC_OBJ_GROUP;
    if (*end >= '0' && *end <= '9') {
      depthattr = strtol(end, &end, 10);
    }

  } else
    return -1;

  *typep = type;
  if (attrp) {
    if (hwloc__obj_type_is_cache(type) && attrsize >= sizeof(attrp->cache)) {
      attrp->cache.depth = depthattr;
      attrp->cache.type = cachetypeattr;
    } else if (type == HWLOC_OBJ_GROUP && attrsize >= sizeof(attrp->group)) {
      attrp->group.depth = depthattr;
    } else if (type == HWLOC_OBJ_BRIDGE && attrsize >= sizeof(attrp->bridge)) {
      attrp->bridge.upstream_type = ubtype;
      attrp->bridge.downstream_type = HWLOC_OBJ_BRIDGE_PCI; /* nothing else so far */
    } else if (type == HWLOC_OBJ_OS_DEVICE && attrsize >= sizeof(attrp->osdev)) {
      attrp->osdev.type = ostype;
    }
  }
  return 0;
}

int
hwloc_type_sscanf_as_depth(const char *string, hwloc_obj_type_t *typep,
			   hwloc_topology_t topology, int *depthp)
{
  union hwloc_obj_attr_u attr;
  hwloc_obj_type_t type;
  int depth;
  int err;

  err = hwloc_type_sscanf(string, &type, &attr, sizeof(attr));
  if (err < 0)
    return err;

  depth = hwloc_get_type_depth(topology, type);
  if (type == HWLOC_OBJ_GROUP
      && depth == HWLOC_TYPE_DEPTH_MULTIPLE
      && attr.group.depth != (unsigned)-1) {
    unsigned l;
    depth = HWLOC_TYPE_DEPTH_UNKNOWN;
    for(l=0; l<topology->nb_levels; l++) {
      if (topology->levels[l][0]->type == HWLOC_OBJ_GROUP
	  && topology->levels[l][0]->attr->group.depth == attr.group.depth) {
	depth = (int)l;
	break;
      }
    }
  }

  if (typep)
    *typep = type;
  *depthp = depth;
  return 0;
}

static const char* hwloc_obj_cache_type_letter(hwloc_obj_cache_type_t type)
{
  switch (type) {
  case HWLOC_OBJ_CACHE_UNIFIED: return "";
  case HWLOC_OBJ_CACHE_DATA: return "d";
  case HWLOC_OBJ_CACHE_INSTRUCTION: return "i";
  default: return "unknown";
  }
}

int
hwloc_obj_type_snprintf(char * __hwloc_restrict string, size_t size, hwloc_obj_t obj, int verbose)
{
  hwloc_obj_type_t type = obj->type;
  switch (type) {
  case HWLOC_OBJ_MISC:
  case HWLOC_OBJ_MACHINE:
  case HWLOC_OBJ_NUMANODE:
  case HWLOC_OBJ_MEMCACHE:
  case HWLOC_OBJ_PACKAGE:
  case HWLOC_OBJ_DIE:
  case HWLOC_OBJ_CORE:
  case HWLOC_OBJ_PU:
    return hwloc_snprintf(string, size, "%s", hwloc_obj_type_string(type));
  case HWLOC_OBJ_L1CACHE:
  case HWLOC_OBJ_L2CACHE:
  case HWLOC_OBJ_L3CACHE:
  case HWLOC_OBJ_L4CACHE:
  case HWLOC_OBJ_L5CACHE:
  case HWLOC_OBJ_L1ICACHE:
  case HWLOC_OBJ_L2ICACHE:
  case HWLOC_OBJ_L3ICACHE:
    return hwloc_snprintf(string, size, "L%u%s%s", obj->attr->cache.depth,
			  hwloc_obj_cache_type_letter(obj->attr->cache.type),
			  verbose ? "Cache" : "");
  case HWLOC_OBJ_GROUP:
    if (obj->attr->group.depth != (unsigned) -1)
      return hwloc_snprintf(string, size, "%s%u", hwloc_obj_type_string(type), obj->attr->group.depth);
    else
      return hwloc_snprintf(string, size, "%s", hwloc_obj_type_string(type));
  case HWLOC_OBJ_BRIDGE:
    return hwloc_snprintf(string, size, obj->attr->bridge.upstream_type == HWLOC_OBJ_BRIDGE_PCI ? "PCIBridge" : "HostBridge");
  case HWLOC_OBJ_PCI_DEVICE:
    return hwloc_snprintf(string, size, "PCI");
  case HWLOC_OBJ_OS_DEVICE:
    switch (obj->attr->osdev.type) {
    case HWLOC_OBJ_OSDEV_BLOCK: return hwloc_snprintf(string, size, "Block");
    case HWLOC_OBJ_OSDEV_NETWORK: return hwloc_snprintf(string, size, verbose ? "Network" : "Net");
    case HWLOC_OBJ_OSDEV_OPENFABRICS: return hwloc_snprintf(string, size, "OpenFabrics");
    case HWLOC_OBJ_OSDEV_DMA: return hwloc_snprintf(string, size, "DMA");
    case HWLOC_OBJ_OSDEV_GPU: return hwloc_snprintf(string, size, "GPU");
    case HWLOC_OBJ_OSDEV_COPROC: return hwloc_snprintf(string, size, verbose ? "Co-Processor" : "CoProc");
    default:
      if (size > 0)
	*string = '\0';
      return 0;
    }
    break;
  default:
    if (size > 0)
      *string = '\0';
    return 0;
  }
}

int
hwloc_obj_attr_snprintf(char * __hwloc_restrict string, size_t size, hwloc_obj_t obj, const char * separator, int verbose)
{
  const char *prefix = "";
  char *tmp = string;
  ssize_t tmplen = size;
  int ret = 0;
  int res;

  /* make sure we output at least an empty string */
  if (size)
    *string = '\0';

  /* print memory attributes */
  res = 0;
  if (verbose) {
    if (obj->type == HWLOC_OBJ_NUMANODE && obj->attr->numanode.local_memory)
      res = hwloc_snprintf(tmp, tmplen, "%slocal=%lu%s%stotal=%lu%s",
			   prefix,
			   (unsigned long) hwloc_memory_size_printf_value(obj->attr->numanode.local_memory, verbose),
			   hwloc_memory_size_printf_unit(obj->attr->numanode.local_memory, verbose),
			   separator,
			   (unsigned long) hwloc_memory_size_printf_value(obj->total_memory, verbose),
			   hwloc_memory_size_printf_unit(obj->total_memory, verbose));
    else if (obj->total_memory)
      res = hwloc_snprintf(tmp, tmplen, "%stotal=%lu%s",
			   prefix,
			   (unsigned long) hwloc_memory_size_printf_value(obj->total_memory, verbose),
			   hwloc_memory_size_printf_unit(obj->total_memory, verbose));
  } else {
    if (obj->type == HWLOC_OBJ_NUMANODE && obj->attr->numanode.local_memory)
      res = hwloc_snprintf(tmp, tmplen, "%s%lu%s",
			   prefix,
			   (unsigned long) hwloc_memory_size_printf_value(obj->attr->numanode.local_memory, verbose),
			   hwloc_memory_size_printf_unit(obj->attr->numanode.local_memory, verbose));
  }
  if (res < 0)
    return -1;
  ret += res;
  if (ret > 0)
    prefix = separator;
  if (res >= tmplen)
    res = tmplen>0 ? (int)tmplen - 1 : 0;
  tmp += res;
  tmplen -= res;

  /* printf type-specific attributes */
  res = 0;
  switch (obj->type) {
  case HWLOC_OBJ_L1CACHE:
  case HWLOC_OBJ_L2CACHE:
  case HWLOC_OBJ_L3CACHE:
  case HWLOC_OBJ_L4CACHE:
  case HWLOC_OBJ_L5CACHE:
  case HWLOC_OBJ_L1ICACHE:
  case HWLOC_OBJ_L2ICACHE:
  case HWLOC_OBJ_L3ICACHE:
  case HWLOC_OBJ_MEMCACHE:
    if (verbose) {
      char assoc[32];
      if (obj->attr->cache.associativity == -1)
	snprintf(assoc, sizeof(assoc), "%sfully-associative", separator);
      else if (obj->attr->cache.associativity == 0)
	*assoc = '\0';
      else
	snprintf(assoc, sizeof(assoc), "%sways=%d", separator, obj->attr->cache.associativity);
      res = hwloc_snprintf(tmp, tmplen, "%ssize=%lu%s%slinesize=%u%s",
			   prefix,
			   (unsigned long) hwloc_memory_size_printf_value(obj->attr->cache.size, verbose),
			   hwloc_memory_size_printf_unit(obj->attr->cache.size, verbose),
			   separator, obj->attr->cache.linesize,
			   assoc);
    } else
      res = hwloc_snprintf(tmp, tmplen, "%s%lu%s",
			   prefix,
			   (unsigned long) hwloc_memory_size_printf_value(obj->attr->cache.size, verbose),
			   hwloc_memory_size_printf_unit(obj->attr->cache.size, verbose));
    break;
  case HWLOC_OBJ_BRIDGE:
    if (verbose) {
      char up[128], down[64];
      /* upstream is PCI or HOST */
      if (obj->attr->bridge.upstream_type == HWLOC_OBJ_BRIDGE_PCI) {
        char linkspeed[64]= "";
        if (obj->attr->pcidev.linkspeed)
          snprintf(linkspeed, sizeof(linkspeed), "%slink=%.2fGB/s", separator, obj->attr->pcidev.linkspeed);
	snprintf(up, sizeof(up), "busid=%04x:%02x:%02x.%01x%sid=%04x:%04x%sclass=%04x(%s)%s",
		 obj->attr->pcidev.domain, obj->attr->pcidev.bus, obj->attr->pcidev.dev, obj->attr->pcidev.func, separator,
		 obj->attr->pcidev.vendor_id, obj->attr->pcidev.device_id, separator,
		 obj->attr->pcidev.class_id, hwloc_pci_class_string(obj->attr->pcidev.class_id), linkspeed);
      } else
        *up = '\0';
      /* downstream is_PCI */
      snprintf(down, sizeof(down), "buses=%04x:[%02x-%02x]",
	       obj->attr->bridge.downstream.pci.domain, obj->attr->bridge.downstream.pci.secondary_bus, obj->attr->bridge.downstream.pci.subordinate_bus);
      if (*up)
	res = hwloc_snprintf(string, size, "%s%s%s", up, separator, down);
      else
	res = hwloc_snprintf(string, size, "%s", down);
    }
    break;
  case HWLOC_OBJ_PCI_DEVICE:
    if (verbose) {
      char linkspeed[64]= "";
      if (obj->attr->pcidev.linkspeed)
        snprintf(linkspeed, sizeof(linkspeed), "%slink=%.2fGB/s", separator, obj->attr->pcidev.linkspeed);
      res = hwloc_snprintf(string, size, "busid=%04x:%02x:%02x.%01x%sid=%04x:%04x%sclass=%04x(%s)%s",
			   obj->attr->pcidev.domain, obj->attr->pcidev.bus, obj->attr->pcidev.dev, obj->attr->pcidev.func, separator,
			   obj->attr->pcidev.vendor_id, obj->attr->pcidev.device_id, separator,
			   obj->attr->pcidev.class_id, hwloc_pci_class_string(obj->attr->pcidev.class_id), linkspeed);
    }
    break;
  default:
    break;
  }
  if (res < 0)
    return -1;
  ret += res;
  if (ret > 0)
    prefix = separator;
  if (res >= tmplen)
    res = tmplen>0 ? (int)tmplen - 1 : 0;
  tmp += res;
  tmplen -= res;

  /* printf infos */
  if (verbose) {
    unsigned i;
    for(i=0; i<obj->infos_count; i++) {
      struct hwloc_info_s *info = &obj->infos[i];
      const char *quote;
      if (strchr(info->value, ' '))
        quote = "\"";
      else
        quote = "";
      res = hwloc_snprintf(tmp, tmplen, "%s%s=%s%s%s",
			     prefix,
			     info->name,
			     quote, info->value, quote);
      if (res < 0)
        return -1;
      ret += res;
      if (res >= tmplen)
        res = tmplen>0 ? (int)tmplen - 1 : 0;
      tmp += res;
      tmplen -= res;
      if (ret > 0)
        prefix = separator;
    }
  }

  return ret;
}

int hwloc_bitmap_singlify_per_core(hwloc_topology_t topology, hwloc_bitmap_t cpuset, unsigned which)
{
  hwloc_obj_t core = NULL;
  while ((core = hwloc_get_next_obj_covering_cpuset_by_type(topology, cpuset, HWLOC_OBJ_CORE, core)) != NULL) {
    /* this core has some PUs in the cpuset, find the index-th one */
    unsigned i = 0;
    int pu = -1;
    do {
      pu = hwloc_bitmap_next(core->cpuset, pu);
      if (pu == -1) {
	/* no which-th PU in cpuset and core, remove the entire core */
	hwloc_bitmap_andnot(cpuset, cpuset, core->cpuset);
	break;
      }
      if (hwloc_bitmap_isset(cpuset, pu)) {
	if (i == which) {
	  /* remove the entire core except that exact pu */
	  hwloc_bitmap_andnot(cpuset, cpuset, core->cpuset);
	  hwloc_bitmap_set(cpuset, pu);
	  break;
	}
	i++;
      }
    } while (1);
  }
  return 0;
}
