/*
 * Copyright © 2010-2018 Inria.  All rights reserved.
 * Copyright © 2011-2012 Université Bordeaux
 * Copyright © 2011 Cisco Systems, Inc.  All rights reserved.
 * See COPYING in top-level directory.
 */

#include <private/autogen/config.h>
#include <hwloc.h>
#include <private/private.h>
#include <private/debug.h>
#include <private/misc.h>

#include <float.h>
#include <math.h>

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

  newdist->type = olddist->type;
  newdist->nbobjs = nbobjs;
  newdist->kind = olddist->kind;
  newdist->id = olddist->id;

  newdist->indexes = hwloc_tma_malloc(tma, nbobjs * sizeof(*newdist->indexes));
  newdist->objs = hwloc_tma_calloc(tma, nbobjs * sizeof(*newdist->objs));
  newdist->objs_are_valid = 0;
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

  /* switch back to types since we don't support groups for now */
  type = hwloc_get_depth_type(topology, depth);
  if (type == (hwloc_obj_type_t)-1) {
    errno = EINVAL;
    return -1;
  }

  next = topology->first_dist;
  while ((dist = next) != NULL) {
    next = dist->next;
    if (dist->type == type) {
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

/******************************************************
 * Add distances to the topology
 */

static void
hwloc__groups_by_distances(struct hwloc_topology *topology, unsigned nbobjs, struct hwloc_obj **objs, uint64_t *values, unsigned long kind, unsigned nbaccuracies, float *accuracies, int needcheck);

/* insert a distance matrix in the topology.
 * the caller gives us the distances and objs pointers, we'll free them later.
 */
static int
hwloc_internal_distances__add(hwloc_topology_t topology,
			      hwloc_obj_type_t type, unsigned nbobjs, hwloc_obj_t *objs, uint64_t *indexes, uint64_t *values,
			      unsigned long kind)
{
  struct hwloc_internal_distances_s *dist = calloc(1, sizeof(*dist));
  if (!dist)
    goto err;

  dist->type = type;
  dist->nbobjs = nbobjs;
  dist->kind = kind;

  if (!objs) {
    assert(indexes);
    /* we only have indexes, we'll refresh objs from there */
    dist->indexes = indexes;
    dist->objs = calloc(nbobjs, sizeof(hwloc_obj_t));
    if (!dist->objs)
      goto err_with_dist;
    dist->objs_are_valid = 0;

  } else {
    unsigned i;
    assert(!indexes);
    /* we only have objs, generate the indexes arrays so that we can refresh objs later */
    dist->objs = objs;
    dist->objs_are_valid = 1;
    dist->indexes = malloc(nbobjs * sizeof(*dist->indexes));
    if (!dist->indexes)
      goto err_with_dist;
    if (dist->type == HWLOC_OBJ_PU || dist->type == HWLOC_OBJ_NUMANODE) {
      for(i=0; i<nbobjs; i++)
	dist->indexes[i] = objs[i]->os_index;
    } else {
      for(i=0; i<nbobjs; i++)
	dist->indexes[i] = objs[i]->gp_index;
    }
  }

  dist->values = values;

  dist->id = topology->next_dist_id++;

  if (topology->last_dist)
    topology->last_dist->next = dist;
  else
    topology->first_dist = dist;
  dist->prev = topology->last_dist;
  dist->next = NULL;
  topology->last_dist = dist;
  return 0;

 err_with_dist:
  free(dist);
 err:
  free(objs);
  free(indexes);
  free(values);
  return -1;
}

int hwloc_internal_distances_add_by_index(hwloc_topology_t topology,
					  hwloc_obj_type_t type, unsigned nbobjs, uint64_t *indexes, uint64_t *values,
					  unsigned long kind, unsigned long flags)
{
  if (nbobjs < 2) {
    errno = EINVAL;
    goto err;
  }

  /* cannot group without objects,
   * and we don't group from XML anyway since the hwloc that generated the XML should have grouped already.
   */
  if (flags & HWLOC_DISTANCES_ADD_FLAG_GROUP) {
    errno = EINVAL;
    goto err;
  }

  return hwloc_internal_distances__add(topology, type, nbobjs, NULL, indexes, values, kind);

 err:
  free(indexes);
  free(values);
  return -1;
}

int hwloc_internal_distances_add(hwloc_topology_t topology,
				 unsigned nbobjs, hwloc_obj_t *objs, uint64_t *values,
				 unsigned long kind, unsigned long flags)
{
  if (nbobjs < 2) {
    errno = EINVAL;
    goto err;
  }

  if (topology->grouping && (flags & HWLOC_DISTANCES_ADD_FLAG_GROUP)) {
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
      unsigned i, j;
      int gp = (objs[0]->type != HWLOC_OBJ_NUMANODE && objs[0]->type != HWLOC_OBJ_PU);
      fprintf(stderr, "Trying to group objects using distance matrix:\n");
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

    hwloc__groups_by_distances(topology, nbobjs, objs, values,
			       kind, nbaccuracies, accuracies, 1 /* check the first matrice */);
  }

  return hwloc_internal_distances__add(topology, objs[0]->type, nbobjs, objs, NULL, values, kind);

 err:
  free(objs);
  free(values);
  return -1;
}

#define HWLOC_DISTANCES_KIND_FROM_ALL (HWLOC_DISTANCES_KIND_FROM_OS|HWLOC_DISTANCES_KIND_FROM_USER)
#define HWLOC_DISTANCES_KIND_MEANS_ALL (HWLOC_DISTANCES_KIND_MEANS_LATENCY|HWLOC_DISTANCES_KIND_MEANS_BANDWIDTH)
#define HWLOC_DISTANCES_KIND_ALL (HWLOC_DISTANCES_KIND_FROM_ALL|HWLOC_DISTANCES_KIND_MEANS_ALL)
#define HWLOC_DISTANCES_ADD_FLAG_ALL (HWLOC_DISTANCES_ADD_FLAG_GROUP|HWLOC_DISTANCES_ADD_FLAG_GROUP_INACCURATE)

/* The actual function exported to the user
 */
int hwloc_distances_add(hwloc_topology_t topology,
			unsigned nbobjs, hwloc_obj_t *objs, hwloc_uint64_t *values,
			unsigned long kind, unsigned long flags)
{
  hwloc_obj_type_t type;
  unsigned i;
  uint64_t *_values;
  hwloc_obj_t *_objs;
  int err;

  if (nbobjs < 2 || !objs || !values || !topology->is_loaded) {
    errno = EINVAL;
    return -1;
  }
  if ((kind & ~HWLOC_DISTANCES_KIND_ALL)
      || hwloc_weight_long(kind & HWLOC_DISTANCES_KIND_FROM_ALL) != 1
      || hwloc_weight_long(kind & HWLOC_DISTANCES_KIND_MEANS_ALL) != 1
      || (flags & ~HWLOC_DISTANCES_ADD_FLAG_ALL)) {
    errno = EINVAL;
    return -1;
  }

  /* no strict need to check for duplicates, things shouldn't break */

  type = objs[0]->type;
  if (type == HWLOC_OBJ_GROUP) {
    /* not supported yet, would require we save the subkind together with the type. */
    errno = EINVAL;
    return -1;
  }

  for(i=1; i<nbobjs; i++)
    if (!objs[i] || objs[i]->type != type) {
      errno = EINVAL;
      return -1;
    }

  /* copy the input arrays and give them to the topology */
  _objs = malloc(nbobjs*sizeof(hwloc_obj_t));
  _values = malloc(nbobjs*nbobjs*sizeof(*_values));
  if (!_objs || !_values)
    goto out_with_arrays;

  memcpy(_objs, objs, nbobjs*sizeof(hwloc_obj_t));
  memcpy(_values, values, nbobjs*nbobjs*sizeof(*_values));
  err = hwloc_internal_distances_add(topology, nbobjs, _objs, _values, kind, flags);
  if (err < 0)
    goto out; /* _objs and _values freed in hwloc_internal_distances_add() */

  /* in case we added some groups, see if we need to reconnect */
  hwloc_topology_reconnect(topology, 0);

  return 0;

 out_with_arrays:
  free(_values);
  free(_objs);
 out:
  return -1;
}

/******************************************************
 * Refresh objects in distances
 */

static hwloc_obj_t hwloc_find_obj_by_type_and_gp_index(hwloc_topology_t topology, hwloc_obj_type_t type, uint64_t gp_index)
{
  hwloc_obj_t obj = hwloc_get_obj_by_type(topology, type, 0);
  while (obj) {
    if (obj->gp_index == gp_index)
      return obj;
    obj = obj->next_cousin;
  }
  return NULL;
}

static void
hwloc_internal_distances_restrict(struct hwloc_internal_distances_s *dist,
				  hwloc_obj_t *objs,
				  unsigned disappeared)
{
  unsigned nbobjs = dist->nbobjs;
  unsigned i, newi;
  unsigned j, newj;

  for(i=0, newi=0; i<nbobjs; i++)
    if (objs[i]) {
      for(j=0, newj=0; j<nbobjs; j++)
	if (objs[j]) {
	  dist->values[newi*(nbobjs-disappeared)+newj] = dist->values[i*nbobjs+j];
	  newj++;
	}
      newi++;
    }

  for(i=0, newi=0; i<nbobjs; i++)
    if (objs[i]) {
      objs[newi] = objs[i];
      dist->indexes[newi] = dist->indexes[i];
      newi++;
    }

  dist->nbobjs -= disappeared;
}

static int
hwloc_internal_distances_refresh_one(hwloc_topology_t topology,
				     struct hwloc_internal_distances_s *dist)
{
  hwloc_obj_type_t type = dist->type;
  unsigned nbobjs = dist->nbobjs;
  hwloc_obj_t *objs = dist->objs;
  uint64_t *indexes = dist->indexes;
  unsigned disappeared = 0;
  unsigned i;

  if (dist->objs_are_valid)
    return 0;

  for(i=0; i<nbobjs; i++) {
    hwloc_obj_t obj;
    /* TODO use cpuset/nodeset to find pus/numas from the root?
     * faster than traversing the entire level?
     */
    if (type == HWLOC_OBJ_PU)
      obj = hwloc_get_pu_obj_by_os_index(topology, (unsigned) indexes[i]);
    else if (type == HWLOC_OBJ_NUMANODE)
      obj = hwloc_get_numanode_obj_by_os_index(topology, (unsigned) indexes[i]);
    else
      obj = hwloc_find_obj_by_type_and_gp_index(topology, type, indexes[i]);
    objs[i] = obj;
    if (!obj)
      disappeared++;
  }

  if (nbobjs-disappeared < 2)
    /* became useless, drop */
    return -1;

  if (disappeared)
    hwloc_internal_distances_restrict(dist, objs, disappeared);

  dist->objs_are_valid = 1;
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
    dist->objs_are_valid = 0;
}

/******************************************************
 * User API for getting distances
 */

void
hwloc_distances_release(hwloc_topology_t topology __hwloc_attribute_unused,
			struct hwloc_distances_s *distances)
{
  free(distances->values);
  free(distances->objs);
  free(distances);
}

static struct hwloc_distances_s *
hwloc_distances_get_one(hwloc_topology_t topology __hwloc_attribute_unused,
			struct hwloc_internal_distances_s *dist)
{
  struct hwloc_distances_s *distances;
  unsigned nbobjs;

  distances = malloc(sizeof(*distances));
  if (!distances)
    return NULL;

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
  return distances;

 out_with_objs:
  free(distances->objs);
 out:
  free(distances);
  return NULL;
}

static int
hwloc__distances_get(hwloc_topology_t topology,
		     hwloc_obj_type_t type,
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

    if (type != HWLOC_OBJ_TYPE_NONE && type != dist->type)
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

  return hwloc__distances_get(topology, HWLOC_OBJ_TYPE_NONE, nrp, distancesp, kind, flags);
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

  /* switch back to types since we don't support groups for now */
  type = hwloc_get_depth_type(topology, depth);
  if (type == (hwloc_obj_type_t)-1) {
    errno = EINVAL;
    return -1;
  }

  return hwloc__distances_get(topology, type, nrp, distancesp, kind, flags);
}

/******************************************************
 * Grouping objects according to distances
 */

static void hwloc_report_user_distance_error(const char *msg, int line)
{
  static int reported = 0;

  if (!reported && !hwloc_hide_errors()) {
    fprintf(stderr, "****************************************************************************\n");
    fprintf(stderr, "* hwloc %s was given invalid distances by the user.\n", HWLOC_VERSION);
    fprintf(stderr, "*\n");
    fprintf(stderr, "* %s\n", msg);
    fprintf(stderr, "* Error occurred in topology.c line %d\n", line);
    fprintf(stderr, "*\n");
    fprintf(stderr, "* Please make sure that distances given through the programming API\n");
    fprintf(stderr, "* do not contradict any other topology information.\n");
    fprintf(stderr, "* \n");
    fprintf(stderr, "* hwloc will now ignore this invalid topology information and continue.\n");
    fprintf(stderr, "****************************************************************************\n");
    reported = 1;
  }
}

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
  HWLOC_VLA(unsigned, groupids, nbobjs);
  unsigned nbgroups = 0;
  unsigned i,j;
  int verbose = topology->grouping_verbose;

  if (nbobjs <= 2)
      return;

  if (!(kind & HWLOC_DISTANCES_KIND_MEANS_LATENCY))
    /* don't know use to use those for grouping */
    /* TODO hwloc__find_groups_by_max_distance() for bandwidth */
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
    return;

  {
      HWLOC_VLA(hwloc_obj_t, groupobjs, nbgroups);
      HWLOC_VLA(unsigned, groupsizes, nbgroups);
      HWLOC_VLA(uint64_t, groupvalues, nbgroups*nbgroups);
      unsigned failed = 0;

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
						   (kind & HWLOC_DISTANCES_KIND_FROM_USER) ? hwloc_report_user_distance_error : hwloc_report_os_error);
	  /* res_obj may be NULL on failure to insert. */
	  if (!res_obj)
	    failed++;
	  /* or it may be different from groupobjs if we got groups from XML import before grouping */
          groupobjs[i] = res_obj;
      }
      topology->grouping_next_subkind++;

      if (failed)
	/* don't try to group above if we got a NULL group here, just keep this incomplete level */
	return;

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
  }
}
