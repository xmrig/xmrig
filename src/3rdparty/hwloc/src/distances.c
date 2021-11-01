/*
 * Copyright © 2010-2021 Inria.  All rights reserved.
 * Copyright © 2011-2012 Université Bordeaux
 * Copyright © 2011 Cisco Systems, Inc.  All rights reserved.
 * See COPYING in top-level directory.
 */

#include "private/autogen/config.h"
#include "hwloc.h"
#include "private/private.h"
#include "private/debug.h"
#include "private/misc.h"

#include <float.h>
#include <math.h>

static struct hwloc_internal_distances_s *
hwloc__internal_distances_from_public(hwloc_topology_t topology, struct hwloc_distances_s *distances);

static void
hwloc__groups_by_distances(struct hwloc_topology *topology, unsigned nbobjs, struct hwloc_obj **objs, uint64_t *values, unsigned long kind, unsigned nbaccuracies, float *accuracies, int needcheck);

static void
hwloc_internal_distances_restrict(hwloc_obj_t *objs,
				  uint64_t *indexes,
				  hwloc_obj_type_t *different_types,
				  uint64_t *values,
				  unsigned nbobjs, unsigned disappeared);

static void
hwloc_internal_distances_print_matrix(struct hwloc_internal_distances_s *dist)
{
  unsigned nbobjs = dist->nbobjs;
  hwloc_obj_t *objs = dist->objs;
  hwloc_uint64_t *values = dist->values;
  int gp = !HWLOC_DIST_TYPE_USE_OS_INDEX(dist->unique_type);
  unsigned i, j;

  fprintf(stderr, "%s", gp ? "gp_index" : "os_index");
  for(j=0; j<nbobjs; j++)
    fprintf(stderr, " % 5d", (int)(gp ? objs[j]->gp_index : objs[j]->os_index));
  fprintf(stderr, "\n");
  for(i=0; i<nbobjs; i++) {
    fprintf(stderr, "  % 5d", (int)(gp ? objs[i]->gp_index : objs[i]->os_index));
    for(j=0; j<nbobjs; j++)
      fprintf(stderr, " % 5lld", (long long) values[i*nbobjs + j]);
    fprintf(stderr, "\n");
  }
}

/******************************************************
 * Global init, prepare, destroy, dup
 */

/* called during topology init() */
void hwloc_internal_distances_init(struct hwloc_topology *topology)
{
  topology->first_dist = topology->last_dist = NULL;
  topology->next_dist_id = 0;
}

/* called at the beginning of load() */
void hwloc_internal_distances_prepare(struct hwloc_topology *topology)
{
  char *env;
  hwloc_localeswitch_declare;

  topology->grouping = 1;
  if (topology->type_filter[HWLOC_OBJ_GROUP] == HWLOC_TYPE_FILTER_KEEP_NONE)
    topology->grouping = 0;
  env = getenv("HWLOC_GROUPING");
  if (env && !atoi(env))
    topology->grouping = 0;

  if (topology->grouping) {
    topology->grouping_next_subkind = 0;

    HWLOC_BUILD_ASSERT(sizeof(topology->grouping_accuracies)/sizeof(*topology->grouping_accuracies) == 5);
    topology->grouping_accuracies[0] = 0.0f;
    topology->grouping_accuracies[1] = 0.01f;
    topology->grouping_accuracies[2] = 0.02f;
    topology->grouping_accuracies[3] = 0.05f;
    topology->grouping_accuracies[4] = 0.1f;
    topology->grouping_nbaccuracies = 5;

    hwloc_localeswitch_init();
    env = getenv("HWLOC_GROUPING_ACCURACY");
    if (!env) {
      /* only use 0.0 */
      topology->grouping_nbaccuracies = 1;
    } else if (strcmp(env, "try")) {
      /* use the given value */
      topology->grouping_nbaccuracies = 1;
      topology->grouping_accuracies[0] = (float) atof(env);
    } /* otherwise try all values */
    hwloc_localeswitch_fini();

    topology->grouping_verbose = 0;
    env = getenv("HWLOC_GROUPING_VERBOSE");
    if (env)
      topology->grouping_verbose = atoi(env);
  }
}

static void hwloc_internal_distances_free(struct hwloc_internal_distances_s *dist)
{
  free(dist->name);
  free(dist->different_types);
  free(dist->indexes);
  free(dist->objs);
  free(dist->values);
  free(dist);
}

/* called during topology destroy */
void hwloc_internal_distances_destroy(struct hwloc_topology * topology)
{
  struct hwloc_internal_distances_s *dist, *next = topology->first_dist;
  while ((dist = next) != NULL) {
    next = dist->next;
    hwloc_internal_distances_free(dist);
  }
  topology->first_dist = topology->last_dist = NULL;
}

static int hwloc_internal_distances_dup_one(struct hwloc_topology *new, struct hwloc_internal_distances_s *olddist)
{
  struct hwloc_tma *tma = new->tma;
  struct hwloc_internal_distances_s *newdist;
  unsigned nbobjs = olddist->nbobjs;

  newdist = hwloc_tma_malloc(tma, sizeof(*newdist));
  if (!newdist)
    return -1;
  if (olddist->name) {
    newdist->name = hwloc_tma_strdup(tma, olddist->name);
    if (!newdist->name) {
      assert(!tma || !tma->dontfree); /* this tma cannot fail to allocate */
      hwloc_internal_distances_free(newdist);
      return -1;
    }
  } else {
    newdist->name = NULL;
  }

  if (olddist->different_types) {
    newdist->different_types = hwloc_tma_malloc(tma, nbobjs * sizeof(*newdist->different_types));
    if (!newdist->different_types) {
      assert(!tma || !tma->dontfree); /* this tma cannot fail to allocate */
      hwloc_internal_distances_free(newdist);
      return -1;
    }
    memcpy(newdist->different_types, olddist->different_types, nbobjs * sizeof(*newdist->different_types));
  } else
    newdist->different_types = NULL;
  newdist->unique_type = olddist->unique_type;
  newdist->nbobjs = nbobjs;
  newdist->kind = olddist->kind;
  newdist->id = olddist->id;

  newdist->indexes = hwloc_tma_malloc(tma, nbobjs * sizeof(*newdist->indexes));
  newdist->objs = hwloc_tma_calloc(tma, nbobjs * sizeof(*newdist->objs));
  newdist->iflags = olddist->iflags & ~HWLOC_INTERNAL_DIST_FLAG_OBJS_VALID; /* must be revalidated after dup() */
  newdist->values = hwloc_tma_malloc(tma, nbobjs*nbobjs * sizeof(*newdist->values));
  if (!newdist->indexes || !newdist->objs || !newdist->values) {
    assert(!tma || !tma->dontfree); /* this tma cannot fail to allocate */
    hwloc_internal_distances_free(newdist);
    return -1;
  }

  memcpy(newdist->indexes, olddist->indexes, nbobjs * sizeof(*newdist->indexes));
  memcpy(newdist->values, olddist->values, nbobjs*nbobjs * sizeof(*newdist->values));

  newdist->next = NULL;
  newdist->prev = new->last_dist;
  if (new->last_dist)
    new->last_dist->next = newdist;
  else
    new->first_dist = newdist;
  new->last_dist = newdist;

  return 0;
}

/* This function may be called with topology->tma set, it cannot free() or realloc() */
int hwloc_internal_distances_dup(struct hwloc_topology *new, struct hwloc_topology *old)
{
  struct hwloc_internal_distances_s *olddist;
  int err;
  new->next_dist_id = old->next_dist_id;
  for(olddist = old->first_dist; olddist; olddist = olddist->next) {
    err = hwloc_internal_distances_dup_one(new, olddist);
    if (err < 0)
      return err;
  }
  return 0;
}

/******************************************************
 * Remove distances from the topology
 */

int hwloc_distances_remove(hwloc_topology_t topology)
{
  if (!topology->is_loaded) {
    errno = EINVAL;
    return -1;
  }
  if (topology->adopted_shmem_addr) {
    errno = EPERM;
    return -1;
  }
  hwloc_internal_distances_destroy(topology);
  return 0;
}

int hwloc_distances_remove_by_depth(hwloc_topology_t topology, int depth)
{
  struct hwloc_internal_distances_s *dist, *next;
  hwloc_obj_type_t type;

  if (!topology->is_loaded) {
    errno = EINVAL;
    return -1;
  }
  if (topology->adopted_shmem_addr) {
    errno = EPERM;
    return -1;
  }

  /* switch back to types since we don't support groups for now */
  type = hwloc_get_depth_type(topology, depth);
  if (type == (hwloc_obj_type_t)-1) {
    errno = EINVAL;
    return -1;
  }

  next = topology->first_dist;
  while ((dist = next) != NULL) {
    next = dist->next;
    if (dist->unique_type == type) {
      if (next)
	next->prev = dist->prev;
      else
	topology->last_dist = dist->prev;
      if (dist->prev)
	dist->prev->next = dist->next;
      else
	topology->first_dist = dist->next;
      hwloc_internal_distances_free(dist);
    }
  }

  return 0;
}

int hwloc_distances_release_remove(hwloc_topology_t topology,
				   struct hwloc_distances_s *distances)
{
  struct hwloc_internal_distances_s *dist = hwloc__internal_distances_from_public(topology, distances);
  if (!dist) {
    errno = EINVAL;
    return -1;
  }
  if (dist->prev)
    dist->prev->next = dist->next;
  else
    topology->first_dist = dist->next;
  if (dist->next)
    dist->next->prev = dist->prev;
  else
    topology->last_dist = dist->prev;
  hwloc_internal_distances_free(dist);
  hwloc_distances_release(topology, distances);
  return 0;
}

/*********************************************************
 * Backend functions for adding distances to the topology
 */

/* cancel a distances handle. only needed internally for now */
static void
hwloc_backend_distances_add__cancel(struct hwloc_internal_distances_s *dist)
{
  /* everything is set to NULL in hwloc_backend_distances_add_create() */
  free(dist->name);
  free(dist->indexes);
  free(dist->objs);
  free(dist->different_types);
  free(dist->values);
  free(dist);
}

/* prepare a distances handle for later commit in the topology.
 * we duplicate the caller's name.
 */
hwloc_backend_distances_add_handle_t
hwloc_backend_distances_add_create(hwloc_topology_t topology,
                                   const char *name, unsigned long kind, unsigned long flags)
{
  struct hwloc_internal_distances_s *dist;

  if (flags) {
    errno = EINVAL;
    goto err;
  }

  dist = calloc(1, sizeof(*dist));
  if (!dist)
    goto err;

  if (name) {
    dist->name = strdup(name); /* ignore failure */
    if (!dist->name)
      goto err_with_dist;
  }

  dist->kind = kind;
  dist->iflags = HWLOC_INTERNAL_DIST_FLAG_NOT_COMMITTED;

  dist->unique_type = HWLOC_OBJ_TYPE_NONE;
  dist->different_types = NULL;
  dist->nbobjs = 0;
  dist->indexes = NULL;
  dist->objs = NULL;
  dist->values = NULL;

  dist->id = topology->next_dist_id++;
  return dist;

 err_with_dist:
  hwloc_backend_distances_add__cancel(dist);
 err:
  return NULL;
}

/* attach objects and values to a distances handle.
 * on success, objs and values arrays are attached and will be freed with the distances.
 * on failure, the handle is freed.
 */
int
hwloc_backend_distances_add_values(hwloc_topology_t topology __hwloc_attribute_unused,
                                   hwloc_backend_distances_add_handle_t handle,
                                   unsigned nbobjs, hwloc_obj_t *objs,
                                   hwloc_uint64_t *values,
                                   unsigned long flags)
{
  struct hwloc_internal_distances_s *dist = handle;
  hwloc_obj_type_t unique_type, *different_types = NULL;
  hwloc_uint64_t *indexes = NULL;
  unsigned i, disappeared = 0;

  if (dist->nbobjs || !(dist->iflags & HWLOC_INTERNAL_DIST_FLAG_NOT_COMMITTED)) {
    /* target distances is already set */
    errno = EINVAL;
    goto err;
  }

  if (flags || nbobjs < 2 || !objs || !values) {
    errno = EINVAL;
    goto err;
  }

  /* is there any NULL object? (useful in case of problem during insert in backends) */
  for(i=0; i<nbobjs; i++)
    if (!objs[i])
      disappeared++;
  if (disappeared) {
    /* some objects are NULL */
    if (disappeared == nbobjs) {
      /* nothing left, drop the matrix */
      errno = ENOENT;
      goto err;
    }
    /* restrict the matrix */
    hwloc_internal_distances_restrict(objs, NULL, NULL, values, nbobjs, disappeared);
    nbobjs -= disappeared;
  }

  indexes = malloc(nbobjs * sizeof(*indexes));
  if (!indexes)
    goto err;

  unique_type = objs[0]->type;
  for(i=1; i<nbobjs; i++)
    if (objs[i]->type != unique_type) {
      unique_type = HWLOC_OBJ_TYPE_NONE;
      break;
    }
  if (unique_type == HWLOC_OBJ_TYPE_NONE) {
    /* heterogeneous types */
    different_types = malloc(nbobjs * sizeof(*different_types));
    if (!different_types)
      goto err_with_indexes;
    for(i=0; i<nbobjs; i++)
      different_types[i] = objs[i]->type;
  }

  dist->nbobjs = nbobjs;
  dist->objs = objs;
  dist->iflags |= HWLOC_INTERNAL_DIST_FLAG_OBJS_VALID;
  dist->indexes = indexes;
  dist->unique_type = unique_type;
  dist->different_types = different_types;
  dist->values = values;

  if (different_types)
    dist->kind |= HWLOC_DISTANCES_KIND_HETEROGENEOUS_TYPES;

  if (HWLOC_DIST_TYPE_USE_OS_INDEX(dist->unique_type)) {
      for(i=0; i<nbobjs; i++)
	dist->indexes[i] = objs[i]->os_index;
    } else {
      for(i=0; i<nbobjs; i++)
	dist->indexes[i] = objs[i]->gp_index;
    }

  return 0;

 err_with_indexes:
  free(indexes);
 err:
  hwloc_backend_distances_add__cancel(dist);
  return -1;
}

/* attach objects and values to a distance handle.
 * on success, objs and values arrays are attached and will be freed with the distances.
 * on failure, the handle is freed.
 */
static int
hwloc_backend_distances_add_values_by_index(hwloc_topology_t topology __hwloc_attribute_unused,
                                            hwloc_backend_distances_add_handle_t handle,
                                            unsigned nbobjs, hwloc_obj_type_t unique_type, hwloc_obj_type_t *different_types, hwloc_uint64_t *indexes,
                                            hwloc_uint64_t *values)
{
  struct hwloc_internal_distances_s *dist = handle;
  hwloc_obj_t *objs;

  if (dist->nbobjs || !(dist->iflags & HWLOC_INTERNAL_DIST_FLAG_NOT_COMMITTED)) {
    /* target distances is already set */
    errno = EINVAL;
    goto err;
  }
  if (nbobjs < 2 || !indexes || !values || (unique_type == HWLOC_OBJ_TYPE_NONE && !different_types)) {
    errno = EINVAL;
    goto err;
  }

  objs = malloc(nbobjs * sizeof(*objs));
  if (!objs)
    goto err;

  dist->nbobjs = nbobjs;
  dist->objs = objs;
  dist->indexes = indexes;
  dist->unique_type = unique_type;
  dist->different_types = different_types;
  dist->values = values;

  if (different_types)
    dist->kind |= HWLOC_DISTANCES_KIND_HETEROGENEOUS_TYPES;

  return 0;

 err:
  hwloc_backend_distances_add__cancel(dist);
  return -1;
}

/* commit a distances handle.
 * on failure, the handle is freed with its objects and values arrays.
 */
int
hwloc_backend_distances_add_commit(hwloc_topology_t topology,
                                   hwloc_backend_distances_add_handle_t handle,
                                   unsigned long flags)
{
  struct hwloc_internal_distances_s *dist = handle;

  if (!dist->nbobjs || !(dist->iflags & HWLOC_INTERNAL_DIST_FLAG_NOT_COMMITTED)) {
    /* target distances not ready for commit */
    errno = EINVAL;
    goto err;
  }

  if ((flags & HWLOC_DISTANCES_ADD_FLAG_GROUP) && !dist->objs) {
    /* cannot group without objects,
     * and we don't group from XML anyway since the hwloc that generated the XML should have grouped already.
     */
    errno = EINVAL;
    goto err;
  }

  if (topology->grouping && (flags & HWLOC_DISTANCES_ADD_FLAG_GROUP) && !dist->different_types) {
    float full_accuracy = 0.f;
    float *accuracies;
    unsigned nbaccuracies;

    if (flags & HWLOC_DISTANCES_ADD_FLAG_GROUP_INACCURATE) {
      accuracies = topology->grouping_accuracies;
      nbaccuracies = topology->grouping_nbaccuracies;
    } else {
      accuracies = &full_accuracy;
      nbaccuracies = 1;
    }

    if (topology->grouping_verbose) {
      fprintf(stderr, "Trying to group objects using distance matrix:\n");
      hwloc_internal_distances_print_matrix(dist);
    }

    hwloc__groups_by_distances(topology, dist->nbobjs, dist->objs, dist->values,
			       dist->kind, nbaccuracies, accuracies, 1 /* check the first matrix */);
  }

  if (topology->last_dist)
    topology->last_dist->next = dist;
  else
    topology->first_dist = dist;
  dist->prev = topology->last_dist;
  dist->next = NULL;
  topology->last_dist = dist;

  dist->iflags &= ~HWLOC_INTERNAL_DIST_FLAG_NOT_COMMITTED;
  return 0;

 err:
  hwloc_backend_distances_add__cancel(dist);
  return -1;
}

/* all-in-one backend function not exported to plugins, only used by XML for now */
int hwloc_internal_distances_add_by_index(hwloc_topology_t topology, const char *name,
                                          hwloc_obj_type_t unique_type, hwloc_obj_type_t *different_types, unsigned nbobjs, uint64_t *indexes, uint64_t *values,
                                          unsigned long kind, unsigned long flags)
{
  hwloc_backend_distances_add_handle_t handle;
  int err;

  handle = hwloc_backend_distances_add_create(topology, name, kind, 0);
  if (!handle)
    goto err;

  err = hwloc_backend_distances_add_values_by_index(topology, handle,
                                                    nbobjs, unique_type, different_types, indexes,
                                                    values);
  if (err < 0)
    goto err;

  /* arrays are now attached to the handle */
  indexes = NULL;
  different_types = NULL;
  values = NULL;

  err = hwloc_backend_distances_add_commit(topology, handle, flags);
  if (err < 0)
    goto err;

  return 0;

 err:
  free(indexes);
  free(different_types);
  free(values);
  return -1;
}

/* all-in-one backend function not exported to plugins, used by OS backends */
int hwloc_internal_distances_add(hwloc_topology_t topology, const char *name,
                                 unsigned nbobjs, hwloc_obj_t *objs, uint64_t *values,
                                 unsigned long kind, unsigned long flags)
{
  hwloc_backend_distances_add_handle_t handle;
  int err;

  handle = hwloc_backend_distances_add_create(topology, name, kind, 0);
  if (!handle)
    goto err;

  err = hwloc_backend_distances_add_values(topology, handle,
                                           nbobjs, objs,
                                           values,
                                           0);
  if (err < 0)
    goto err;

  /* arrays are now attached to the handle */
  objs = NULL;
  values = NULL;

  err = hwloc_backend_distances_add_commit(topology, handle, flags);
  if (err < 0)
    goto err;

  return 0;

 err:
  free(objs);
  free(values);
  return -1;
}

/********************************
 * User API for adding distances
 */

#define HWLOC_DISTANCES_KIND_FROM_ALL (HWLOC_DISTANCES_KIND_FROM_OS|HWLOC_DISTANCES_KIND_FROM_USER)
#define HWLOC_DISTANCES_KIND_MEANS_ALL (HWLOC_DISTANCES_KIND_MEANS_LATENCY|HWLOC_DISTANCES_KIND_MEANS_BANDWIDTH)
#define HWLOC_DISTANCES_KIND_ALL (HWLOC_DISTANCES_KIND_FROM_ALL|HWLOC_DISTANCES_KIND_MEANS_ALL|HWLOC_DISTANCES_KIND_HETEROGENEOUS_TYPES)
#define HWLOC_DISTANCES_ADD_FLAG_ALL (HWLOC_DISTANCES_ADD_FLAG_GROUP|HWLOC_DISTANCES_ADD_FLAG_GROUP_INACCURATE)

void * hwloc_distances_add_create(hwloc_topology_t topology,
                                  const char *name, unsigned long kind,
                                  unsigned long flags)
{
  if (!topology->is_loaded) {
    errno = EINVAL;
    return NULL;
  }
  if (topology->adopted_shmem_addr) {
    errno = EPERM;
    return NULL;
  }
  if ((kind & ~HWLOC_DISTANCES_KIND_ALL)
      || hwloc_weight_long(kind & HWLOC_DISTANCES_KIND_FROM_ALL) != 1
      || hwloc_weight_long(kind & HWLOC_DISTANCES_KIND_MEANS_ALL) != 1) {
    errno = EINVAL;
    return NULL;
  }

  return hwloc_backend_distances_add_create(topology, name, kind, flags);
}

int hwloc_distances_add_values(hwloc_topology_t topology,
                               void *handle,
                               unsigned nbobjs, hwloc_obj_t *objs,
                               hwloc_uint64_t *values,
                               unsigned long flags)
{
  unsigned i;
  uint64_t *_values;
  hwloc_obj_t *_objs;
  int err;

  /* no strict need to check for duplicates, things shouldn't break */

  for(i=1; i<nbobjs; i++)
    if (!objs[i]) {
      errno = EINVAL;
      goto out;
    }

  /* copy the input arrays and give them to the topology */
  _objs = malloc(nbobjs*sizeof(hwloc_obj_t));
  _values = malloc(nbobjs*nbobjs*sizeof(*_values));
  if (!_objs || !_values)
    goto out_with_arrays;

  memcpy(_objs, objs, nbobjs*sizeof(hwloc_obj_t));
  memcpy(_values, values, nbobjs*nbobjs*sizeof(*_values));

  err = hwloc_backend_distances_add_values(topology, handle, nbobjs, _objs, _values, flags);
  if (err < 0) {
    /* handle was canceled inside hwloc_backend_distances_add_values */
    handle = NULL;
    goto out_with_arrays;
  }

  return 0;

 out_with_arrays:
  free(_objs);
  free(_values);
 out:
  if (handle)
    hwloc_backend_distances_add__cancel(handle);
  return -1;
}

int
hwloc_distances_add_commit(hwloc_topology_t topology,
                           void *handle,
                           unsigned long flags)
{
  int err;

  if (flags & ~HWLOC_DISTANCES_ADD_FLAG_ALL) {
    errno = EINVAL;
    goto out;
  }

  err = hwloc_backend_distances_add_commit(topology, handle, flags);
  if (err < 0) {
    /* handle was canceled inside hwloc_backend_distances_add_commit */
    handle = NULL;
    goto out;
  }

  /* in case we added some groups, see if we need to reconnect */
  hwloc_topology_reconnect(topology, 0);

  return 0;

 out:
  if (handle)
    hwloc_backend_distances_add__cancel(handle);
  return -1;
}

/* deprecated all-in-one user function */
int hwloc_distances_add(hwloc_topology_t topology,
			unsigned nbobjs, hwloc_obj_t *objs, hwloc_uint64_t *values,
			unsigned long kind, unsigned long flags)
{
  void *handle;
  int err;

  handle = hwloc_distances_add_create(topology, NULL, kind, 0);
  if (!handle)
    return -1;

  err = hwloc_distances_add_values(topology, handle, nbobjs, objs, values, 0);
  if (err < 0)
    return -1;

  err = hwloc_distances_add_commit(topology, handle, flags);
  if (err < 0)
    return -1;

  return 0;
}

/******************************************************
 * Refresh objects in distances
 */

static void
hwloc_internal_distances_restrict(hwloc_obj_t *objs,
				  uint64_t *indexes,
                                  hwloc_obj_type_t *different_types,
				  uint64_t *values,
				  unsigned nbobjs, unsigned disappeared)
{
  unsigned i, newi;
  unsigned j, newj;

  for(i=0, newi=0; i<nbobjs; i++)
    if (objs[i]) {
      for(j=0, newj=0; j<nbobjs; j++)
	if (objs[j]) {
	  values[newi*(nbobjs-disappeared)+newj] = values[i*nbobjs+j];
	  newj++;
	}
      newi++;
    }

  for(i=0, newi=0; i<nbobjs; i++)
    if (objs[i]) {
      objs[newi] = objs[i];
      if (indexes)
	indexes[newi] = indexes[i];
      if (different_types)
        different_types[newi] = different_types[i];
      newi++;
    }
}

static int
hwloc_internal_distances_refresh_one(hwloc_topology_t topology,
				     struct hwloc_internal_distances_s *dist)
{
  hwloc_obj_type_t unique_type = dist->unique_type;
  hwloc_obj_type_t *different_types = dist->different_types;
  unsigned nbobjs = dist->nbobjs;
  hwloc_obj_t *objs = dist->objs;
  uint64_t *indexes = dist->indexes;
  unsigned disappeared = 0;
  unsigned i;

  if (dist->iflags & HWLOC_INTERNAL_DIST_FLAG_OBJS_VALID)
    return 0;

  for(i=0; i<nbobjs; i++) {
    hwloc_obj_t obj;
    /* TODO use cpuset/nodeset to find pus/numas from the root?
     * faster than traversing the entire level?
     */
    if (HWLOC_DIST_TYPE_USE_OS_INDEX(unique_type)) {
      if (unique_type == HWLOC_OBJ_PU)
	obj = hwloc_get_pu_obj_by_os_index(topology, (unsigned) indexes[i]);
      else if (unique_type == HWLOC_OBJ_NUMANODE)
	obj = hwloc_get_numanode_obj_by_os_index(topology, (unsigned) indexes[i]);
      else
	abort();
    } else {
      obj = hwloc_get_obj_by_type_and_gp_index(topology, different_types ? different_types[i] : unique_type, indexes[i]);
    }
    objs[i] = obj;
    if (!obj)
      disappeared++;
  }

  if (nbobjs-disappeared < 2)
    /* became useless, drop */
    return -1;

  if (disappeared) {
    hwloc_internal_distances_restrict(objs, dist->indexes, dist->different_types, dist->values, nbobjs, disappeared);
    dist->nbobjs -= disappeared;
  }

  dist->iflags |= HWLOC_INTERNAL_DIST_FLAG_OBJS_VALID;
  return 0;
}

/* This function may be called with topology->tma set, it cannot free() or realloc() */
void
hwloc_internal_distances_refresh(hwloc_topology_t topology)
{
  struct hwloc_internal_distances_s *dist, *next;

  for(dist = topology->first_dist; dist; dist = next) {
    next = dist->next;

    if (hwloc_internal_distances_refresh_one(topology, dist) < 0) {
      assert(!topology->tma || !topology->tma->dontfree); /* this tma cannot fail to allocate */
      if (dist->prev)
	dist->prev->next = next;
      else
	topology->first_dist = next;
      if (next)
	next->prev = dist->prev;
      else
	topology->last_dist = dist->prev;
      hwloc_internal_distances_free(dist);
      continue;
    }
  }
}

void
hwloc_internal_distances_invalidate_cached_objs(hwloc_topology_t topology)
{
  struct hwloc_internal_distances_s *dist;
  for(dist = topology->first_dist; dist; dist = dist->next)
    dist->iflags &= ~HWLOC_INTERNAL_DIST_FLAG_OBJS_VALID;
}

/******************************************************
 * User API for getting distances
 */

/* what we actually allocate for user queries, even if we only
 * return the distances part of it.
 */
struct hwloc_distances_container_s {
  unsigned id;
  struct hwloc_distances_s distances;
};

#define HWLOC_DISTANCES_CONTAINER_OFFSET ((char*)&((struct hwloc_distances_container_s*)NULL)->distances - (char*)NULL)
#define HWLOC_DISTANCES_CONTAINER(_d) (struct hwloc_distances_container_s *) ( ((char*)_d) - HWLOC_DISTANCES_CONTAINER_OFFSET )

static struct hwloc_internal_distances_s *
hwloc__internal_distances_from_public(hwloc_topology_t topology, struct hwloc_distances_s *distances)
{
  struct hwloc_distances_container_s *cont = HWLOC_DISTANCES_CONTAINER(distances);
  struct hwloc_internal_distances_s *dist;
  for(dist = topology->first_dist; dist; dist = dist->next)
    if (dist->id == cont->id)
      return dist;
  return NULL;
}

void
hwloc_distances_release(hwloc_topology_t topology __hwloc_attribute_unused,
			struct hwloc_distances_s *distances)
{
  struct hwloc_distances_container_s *cont = HWLOC_DISTANCES_CONTAINER(distances);
  free(distances->values);
  free(distances->objs);
  free(cont);
}

const char *
hwloc_distances_get_name(hwloc_topology_t topology, struct hwloc_distances_s *distances)
{
  struct hwloc_internal_distances_s *dist = hwloc__internal_distances_from_public(topology, distances);
  return dist ? dist->name : NULL;
}

static struct hwloc_distances_s *
hwloc_distances_get_one(hwloc_topology_t topology __hwloc_attribute_unused,
			struct hwloc_internal_distances_s *dist)
{
  struct hwloc_distances_container_s *cont;
  struct hwloc_distances_s *distances;
  unsigned nbobjs;

  cont = malloc(sizeof(*cont));
  if (!cont)
    return NULL;
  distances = &cont->distances;

  nbobjs = distances->nbobjs = dist->nbobjs;

  distances->objs = malloc(nbobjs * sizeof(hwloc_obj_t));
  if (!distances->objs)
    goto out;
  memcpy(distances->objs, dist->objs, nbobjs * sizeof(hwloc_obj_t));

  distances->values = malloc(nbobjs * nbobjs * sizeof(*distances->values));
  if (!distances->values)
    goto out_with_objs;
  memcpy(distances->values, dist->values, nbobjs*nbobjs*sizeof(*distances->values));

  distances->kind = dist->kind;

  cont->id = dist->id;
  return distances;

 out_with_objs:
  free(distances->objs);
 out:
  free(cont);
  return NULL;
}

static int
hwloc__distances_get(hwloc_topology_t topology,
		     const char *name, hwloc_obj_type_t type,
		     unsigned *nrp, struct hwloc_distances_s **distancesp,
		     unsigned long kind, unsigned long flags __hwloc_attribute_unused)
{
  struct hwloc_internal_distances_s *dist;
  unsigned nr = 0, i;

  /* We could return the internal arrays (as const),
   * but it would require to prevent removing distances between get() and free().
   * Not performance critical anyway.
   */

  if (flags) {
    errno = EINVAL;
    return -1;
  }

  /* we could refresh only the distances that match, but we won't have many distances anyway,
   * so performance is totally negligible.
   *
   * This is also useful in multithreaded apps that modify the topology.
   * They can call any valid hwloc_distances_get() to force a refresh after
   * changing the topology, so that future concurrent get() won't cause
   * concurrent refresh().
   */
  hwloc_internal_distances_refresh(topology);

  for(dist = topology->first_dist; dist; dist = dist->next) {
    unsigned long kind_from = kind & HWLOC_DISTANCES_KIND_FROM_ALL;
    unsigned long kind_means = kind & HWLOC_DISTANCES_KIND_MEANS_ALL;

    if (name && (!dist->name || strcmp(name, dist->name)))
      continue;

    if (type != HWLOC_OBJ_TYPE_NONE && type != dist->unique_type)
      continue;

    if (kind_from && !(kind_from & dist->kind))
      continue;
    if (kind_means && !(kind_means & dist->kind))
      continue;

    if (nr < *nrp) {
      struct hwloc_distances_s *distances = hwloc_distances_get_one(topology, dist);
      if (!distances)
	goto error;
      distancesp[nr] = distances;
    }
    nr++;
  }

  for(i=nr; i<*nrp; i++)
    distancesp[i] = NULL;
  *nrp = nr;
  return 0;

 error:
  for(i=0; i<nr; i++)
    hwloc_distances_release(topology, distancesp[i]);
  return -1;
}

int
hwloc_distances_get(hwloc_topology_t topology,
		    unsigned *nrp, struct hwloc_distances_s **distancesp,
		    unsigned long kind, unsigned long flags)
{
  if (flags || !topology->is_loaded) {
    errno = EINVAL;
    return -1;
  }

  return hwloc__distances_get(topology, NULL, HWLOC_OBJ_TYPE_NONE, nrp, distancesp, kind, flags);
}

int
hwloc_distances_get_by_depth(hwloc_topology_t topology, int depth,
			     unsigned *nrp, struct hwloc_distances_s **distancesp,
			     unsigned long kind, unsigned long flags)
{
  hwloc_obj_type_t type;

  if (flags || !topology->is_loaded) {
    errno = EINVAL;
    return -1;
  }

  /* FIXME: passing the depth of a group level may return group distances at a different depth */
  type = hwloc_get_depth_type(topology, depth);
  if (type == (hwloc_obj_type_t)-1) {
    errno = EINVAL;
    return -1;
  }

  return hwloc__distances_get(topology, NULL, type, nrp, distancesp, kind, flags);
}

int
hwloc_distances_get_by_name(hwloc_topology_t topology, const char *name,
			    unsigned *nrp, struct hwloc_distances_s **distancesp,
			    unsigned long flags)
{
  if (flags || !topology->is_loaded) {
    errno = EINVAL;
    return -1;
  }

  return hwloc__distances_get(topology, name, HWLOC_OBJ_TYPE_NONE, nrp, distancesp, HWLOC_DISTANCES_KIND_ALL, flags);
}

int
hwloc_distances_get_by_type(hwloc_topology_t topology, hwloc_obj_type_t type,
			    unsigned *nrp, struct hwloc_distances_s **distancesp,
			    unsigned long kind, unsigned long flags)
{
  if (flags || !topology->is_loaded) {
    errno = EINVAL;
    return -1;
  }

  return hwloc__distances_get(topology, NULL, type, nrp, distancesp, kind, flags);
}

/******************************************************
 * Grouping objects according to distances
 */

static int hwloc_compare_values(uint64_t a, uint64_t b, float accuracy)
{
  if (accuracy != 0.0f && fabsf((float)a-(float)b) < (float)a * accuracy)
    return 0;
  return a < b ? -1 : a == b ? 0 : 1;
}

/*
 * Place objects in groups if they are in a transitive graph of minimal values.
 * Return how many groups were created, or 0 if some incomplete distance graphs were found.
 */
static unsigned
hwloc__find_groups_by_min_distance(unsigned nbobjs,
				   uint64_t *_values,
				   float accuracy,
				   unsigned *groupids,
				   int verbose)
{
  uint64_t min_distance = UINT64_MAX;
  unsigned groupid = 1;
  unsigned i,j,k;
  unsigned skipped = 0;

#define VALUE(i, j) _values[(i) * nbobjs + (j)]

  memset(groupids, 0, nbobjs*sizeof(*groupids));

  /* find the minimal distance */
  for(i=0; i<nbobjs; i++)
    for(j=0; j<nbobjs; j++) /* check the entire matrix, it may not be perfectly symmetric depending on the accuracy */
      if (i != j && VALUE(i, j) < min_distance) /* no accuracy here, we want the real minimal */
        min_distance = VALUE(i, j);
  hwloc_debug("  found minimal distance %llu between objects\n", (unsigned long long) min_distance);

  if (min_distance == UINT64_MAX)
    return 0;

  /* build groups of objects connected with this distance */
  for(i=0; i<nbobjs; i++) {
    unsigned size;
    unsigned firstfound;

    /* if already grouped, skip */
    if (groupids[i])
      continue;

    /* start a new group */
    groupids[i] = groupid;
    size = 1;
    firstfound = i;

    while (firstfound != (unsigned)-1) {
      /* we added new objects to the group, the first one was firstfound.
       * rescan all connections from these new objects (starting at first found) to any other objects,
       * so as to find new objects minimally-connected by transivity.
       */
      unsigned newfirstfound = (unsigned)-1;
      for(j=firstfound; j<nbobjs; j++)
	if (groupids[j] == groupid)
	  for(k=0; k<nbobjs; k++)
              if (!groupids[k] && !hwloc_compare_values(VALUE(j, k), min_distance, accuracy)) {
	      groupids[k] = groupid;
	      size++;
	      if (newfirstfound == (unsigned)-1)
		newfirstfound = k;
	      if (i == j)
		hwloc_debug("  object %u is minimally connected to %u\n", k, i);
	      else
	        hwloc_debug("  object %u is minimally connected to %u through %u\n", k, i, j);
	    }
      firstfound = newfirstfound;
    }

    if (size == 1) {
      /* cancel this useless group, ignore this object and try from the next one */
      groupids[i] = 0;
      skipped++;
      continue;
    }

    /* valid this group */
    groupid++;
    if (verbose)
      fprintf(stderr, " Found transitive graph with %u objects with minimal distance %llu accuracy %f\n",
	      size, (unsigned long long) min_distance, accuracy);
  }

  if (groupid == 2 && !skipped)
    /* we created a single group containing all objects, ignore it */
    return 0;

  /* return the last id, since it's also the number of used group ids */
  return groupid-1;
}

/* check that the matrix is ok */
static int
hwloc__check_grouping_matrix(unsigned nbobjs, uint64_t *_values, float accuracy, int verbose)
{
  unsigned i,j;
  for(i=0; i<nbobjs; i++) {
    for(j=i+1; j<nbobjs; j++) {
      /* should be symmetric */
      if (hwloc_compare_values(VALUE(i, j), VALUE(j, i), accuracy)) {
	if (verbose)
	  fprintf(stderr, " Distance matrix asymmetric ([%u,%u]=%llu != [%u,%u]=%llu), aborting\n",
		  i, j, (unsigned long long) VALUE(i, j), j, i, (unsigned long long) VALUE(j, i));
	return -1;
      }
      /* diagonal is smaller than everything else */
      if (hwloc_compare_values(VALUE(i, j), VALUE(i, i), accuracy) <= 0) {
	if (verbose)
	  fprintf(stderr, " Distance to self not strictly minimal ([%u,%u]=%llu <= [%u,%u]=%llu), aborting\n",
		  i, j, (unsigned long long) VALUE(i, j), i, i, (unsigned long long) VALUE(i, i));
	return -1;
      }
    }
  }
  return 0;
}

/*
 * Look at object physical distances to group them.
 */
static void
hwloc__groups_by_distances(struct hwloc_topology *topology,
			   unsigned nbobjs,
			   struct hwloc_obj **objs,
			   uint64_t *_values,
			   unsigned long kind,
			   unsigned nbaccuracies,
			   float *accuracies,
			   int needcheck)
{
  unsigned *groupids;
  unsigned nbgroups = 0;
  unsigned i,j;
  int verbose = topology->grouping_verbose;
  hwloc_obj_t *groupobjs;
  unsigned * groupsizes;
  uint64_t *groupvalues;
  unsigned failed = 0;

  if (nbobjs <= 2)
      return;

  if (!(kind & HWLOC_DISTANCES_KIND_MEANS_LATENCY))
    /* don't know use to use those for grouping */
    /* TODO hwloc__find_groups_by_max_distance() for bandwidth */
    return;

  groupids = malloc(nbobjs * sizeof(*groupids));
  if (!groupids)
    return;

  for(i=0; i<nbaccuracies; i++) {
    if (verbose)
      fprintf(stderr, "Trying to group %u %s objects according to physical distances with accuracy %f\n",
	      nbobjs, hwloc_obj_type_string(objs[0]->type), accuracies[i]);
    if (needcheck && hwloc__check_grouping_matrix(nbobjs, _values, accuracies[i], verbose) < 0)
      continue;
    nbgroups = hwloc__find_groups_by_min_distance(nbobjs, _values, accuracies[i], groupids, verbose);
    if (nbgroups)
      break;
  }
  if (!nbgroups)
    goto out_with_groupids;

  groupobjs = malloc(nbgroups * sizeof(*groupobjs));
  groupsizes = malloc(nbgroups * sizeof(*groupsizes));
  groupvalues = malloc(nbgroups * nbgroups * sizeof(*groupvalues));
  if (!groupobjs || !groupsizes || !groupvalues)
    goto out_with_groups;

      /* create new Group objects and record their size */
      memset(&(groupsizes[0]), 0, sizeof(groupsizes[0]) * nbgroups);
      for(i=0; i<nbgroups; i++) {
          /* create the Group object */
          hwloc_obj_t group_obj, res_obj;
          group_obj = hwloc_alloc_setup_object(topology, HWLOC_OBJ_GROUP, HWLOC_UNKNOWN_INDEX);
          group_obj->cpuset = hwloc_bitmap_alloc();
          group_obj->attr->group.kind = HWLOC_GROUP_KIND_DISTANCE;
          group_obj->attr->group.subkind = topology->grouping_next_subkind;
          for (j=0; j<nbobjs; j++)
	    if (groupids[j] == i+1) {
	      /* assemble the group sets */
	      hwloc_obj_add_other_obj_sets(group_obj, objs[j]);
              groupsizes[i]++;
            }
          hwloc_debug_1arg_bitmap("adding Group object with %u objects and cpuset %s\n",
                                  groupsizes[i], group_obj->cpuset);
          res_obj = hwloc__insert_object_by_cpuset(topology, NULL, group_obj,
                                                   (kind & HWLOC_DISTANCES_KIND_FROM_USER) ? "distances:fromuser:group" : "distances:group");
	  /* res_obj may be NULL on failure to insert. */
	  if (!res_obj)
	    failed++;
	  /* or it may be different from groupobjs if we got groups from XML import before grouping */
          groupobjs[i] = res_obj;
      }
      topology->grouping_next_subkind++;

      if (failed)
	/* don't try to group above if we got a NULL group here, just keep this incomplete level */
	goto out_with_groups;

      /* factorize values */
      memset(&(groupvalues[0]), 0, sizeof(groupvalues[0]) * nbgroups * nbgroups);
#undef VALUE
#define VALUE(i, j) _values[(i) * nbobjs + (j)]
#define GROUP_VALUE(i, j) groupvalues[(i) * nbgroups + (j)]
      for(i=0; i<nbobjs; i++)
	if (groupids[i])
	  for(j=0; j<nbobjs; j++)
	    if (groupids[j])
                GROUP_VALUE(groupids[i]-1, groupids[j]-1) += VALUE(i, j);
      for(i=0; i<nbgroups; i++)
          for(j=0; j<nbgroups; j++) {
              unsigned groupsize = groupsizes[i]*groupsizes[j];
              GROUP_VALUE(i, j) /= groupsize;
          }
#ifdef HWLOC_DEBUG
      hwloc_debug("%s", "generated new distance matrix between groups:\n");
      hwloc_debug("%s", "  index");
      for(j=0; j<nbgroups; j++)
	hwloc_debug(" % 5d", (int) j); /* print index because os_index is -1 for Groups */
      hwloc_debug("%s", "\n");
      for(i=0; i<nbgroups; i++) {
	hwloc_debug("  % 5d", (int) i);
	for(j=0; j<nbgroups; j++)
	  hwloc_debug(" %llu", (unsigned long long) GROUP_VALUE(i, j));
	hwloc_debug("%s", "\n");
      }
#endif

      hwloc__groups_by_distances(topology, nbgroups, groupobjs, groupvalues, kind, nbaccuracies, accuracies, 0 /* no need to check generated matrix */);

 out_with_groups:
  free(groupobjs);
  free(groupsizes);
  free(groupvalues);
 out_with_groupids:
  free(groupids);
}

static int
hwloc__distances_transform_remove_null(struct hwloc_distances_s *distances)
{
  hwloc_uint64_t *values = distances->values;
  hwloc_obj_t *objs = distances->objs;
  unsigned i, nb, nbobjs = distances->nbobjs;
  hwloc_obj_type_t unique_type;

  for(i=0, nb=0; i<nbobjs; i++)
    if (objs[i])
      nb++;

  if (nb < 2) {
    errno = EINVAL;
    return -1;
  }

  if (nb == nbobjs)
    return 0;

  hwloc_internal_distances_restrict(objs, NULL, NULL, values, nbobjs, nbobjs-nb);
  distances->nbobjs = nb;

  /* update HWLOC_DISTANCES_KIND_HETEROGENEOUS_TYPES for convenience */
  unique_type = objs[0]->type;
  for(i=1; i<nb; i++)
    if (objs[i]->type != unique_type) {
      unique_type = HWLOC_OBJ_TYPE_NONE;
      break;
    }
  if (unique_type == HWLOC_OBJ_TYPE_NONE)
    distances->kind |= HWLOC_DISTANCES_KIND_HETEROGENEOUS_TYPES;
  else
    distances->kind &= ~HWLOC_DISTANCES_KIND_HETEROGENEOUS_TYPES;

  return 0;
}

static int
hwloc__distances_transform_links(struct hwloc_distances_s *distances)
{
  /* FIXME: we should look for the greatest common denominator
   * but we just use the smallest positive value, that's enough for current use-cases.
   * We'll return -1 in other cases.
   */
  hwloc_uint64_t divider, *values = distances->values;
  unsigned i, nbobjs = distances->nbobjs;

  if (!(distances->kind & HWLOC_DISTANCES_KIND_MEANS_BANDWIDTH)) {
    errno = EINVAL;
    return -1;
  }

  for(i=0; i<nbobjs; i++)
    values[i*nbobjs+i] = 0;

  /* find the smallest positive value */
  divider = 0;
  for(i=0; i<nbobjs*nbobjs; i++)
    if (values[i] && (!divider || values[i] < divider))
      divider = values[i];

  if (!divider)
    /* only zeroes? do nothing */
    return 0;

  /* check it divides all values */
  for(i=0; i<nbobjs*nbobjs; i++)
    if (values[i]%divider) {
      errno = ENOENT;
      return -1;
    }

  /* ok, now divide for real */
  for(i=0; i<nbobjs*nbobjs; i++)
    values[i] /= divider;

  return 0;
}

static __hwloc_inline int is_nvswitch(hwloc_obj_t obj)
{
  return obj && obj->subtype && !strcmp(obj->subtype, "NVSwitch");
}

static int
hwloc__distances_transform_merge_switch_ports(hwloc_topology_t topology,
                                              struct hwloc_distances_s *distances)
{
  struct hwloc_internal_distances_s *dist = hwloc__internal_distances_from_public(topology, distances);
  hwloc_obj_t *objs = distances->objs;
  hwloc_uint64_t *values = distances->values;
  unsigned first, i, j, nbobjs = distances->nbobjs;

  if (strcmp(dist->name, "NVLinkBandwidth")) {
    errno = EINVAL;
    return -1;
  }

  /* find the first port */
  first = (unsigned) -1;
  for(i=0; i<nbobjs; i++)
    if (is_nvswitch(objs[i])) {
      first = i;
      break;
    }
  if (first == (unsigned)-1) {
    errno = ENOENT;
    return -1;
  }

  for(j=i+1; j<nbobjs; j++) {
    if (is_nvswitch(objs[j])) {
      /* another port, merge it */
      unsigned k;
      for(k=0; k<nbobjs; k++) {
        if (k==i || k==j)
          continue;
        values[k*nbobjs+i] += values[k*nbobjs+j];
        values[k*nbobjs+j] = 0;
        values[i*nbobjs+k] += values[j*nbobjs+k];
        values[j*nbobjs+k] = 0;
      }
      values[i*nbobjs+i] += values[j*nbobjs+j];
      values[j*nbobjs+j] = 0;
    }
    /* the caller will also call REMOVE_NULL to remove other ports */
    objs[j] = NULL;
  }

  return 0;
}

static int
hwloc__distances_transform_transitive_closure(hwloc_topology_t topology,
                                              struct hwloc_distances_s *distances)
{
  struct hwloc_internal_distances_s *dist = hwloc__internal_distances_from_public(topology, distances);
  hwloc_obj_t *objs = distances->objs;
  hwloc_uint64_t *values = distances->values;
  unsigned nbobjs = distances->nbobjs;
  unsigned i, j, k;

  if (strcmp(dist->name, "NVLinkBandwidth")) {
    errno = EINVAL;
    return -1;
  }

  for(i=0; i<nbobjs; i++) {
    hwloc_uint64_t bw_i2sw = 0;
    if (is_nvswitch(objs[i]))
      continue;
    /* count our BW to the switch */
    for(k=0; k<nbobjs; k++)
      if (is_nvswitch(objs[k]))
        bw_i2sw += values[i*nbobjs+k];

    for(j=0; j<nbobjs; j++) {
      hwloc_uint64_t bw_sw2j = 0;
      if (i == j || is_nvswitch(objs[j]))
        continue;
      /* count our BW from the switch */
      for(k=0; k<nbobjs; k++)
        if (is_nvswitch(objs[k]))
          bw_sw2j += values[k*nbobjs+j];

      /* bandwidth from i to j is now min(i2sw,sw2j) */
      values[i*nbobjs+j] = bw_i2sw > bw_sw2j ? bw_sw2j : bw_i2sw;
    }
  }

  return 0;
}

int
hwloc_distances_transform(hwloc_topology_t topology,
                          struct hwloc_distances_s *distances,
                          enum hwloc_distances_transform_e transform,
                          void *transform_attr,
                          unsigned long flags)
{
  if (flags || transform_attr) {
    errno = EINVAL;
    return -1;
  }

  switch (transform) {
  case HWLOC_DISTANCES_TRANSFORM_REMOVE_NULL:
    return hwloc__distances_transform_remove_null(distances);
  case HWLOC_DISTANCES_TRANSFORM_LINKS:
    return hwloc__distances_transform_links(distances);
  case HWLOC_DISTANCES_TRANSFORM_MERGE_SWITCH_PORTS:
  {
    int err;
    err = hwloc__distances_transform_merge_switch_ports(topology, distances);
    if (!err)
      err = hwloc__distances_transform_remove_null(distances);
    return err;
  }
  case HWLOC_DISTANCES_TRANSFORM_TRANSITIVE_CLOSURE:
    return hwloc__distances_transform_transitive_closure(topology, distances);
  default:
    errno = EINVAL;
    return -1;
  }
}
