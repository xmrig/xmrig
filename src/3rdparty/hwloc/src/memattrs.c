/*
 * Copyright © 2020-2023 Inria.  All rights reserved.
 * See COPYING in top-level directory.
 */

#include "private/autogen/config.h"
#include "hwloc.h"
#include "private/private.h"
#include "private/debug.h"


/*****************************
 * Attributes
 */

static __hwloc_inline
hwloc_uint64_t hwloc__memattr_get_convenience_value(hwloc_memattr_id_t id,
                                                    hwloc_obj_t node)
{
  if (id == HWLOC_MEMATTR_ID_CAPACITY)
    return node->attr->numanode.local_memory;
  else if (id == HWLOC_MEMATTR_ID_LOCALITY)
    return hwloc_bitmap_weight(node->cpuset);
  else
    assert(0);
  return 0; /* shut up the compiler */
}

void
hwloc_internal_memattrs_init(struct hwloc_topology *topology)
{
  topology->nr_memattrs = 0;
  topology->memattrs = NULL;
}

static void
hwloc__setup_memattr(struct hwloc_internal_memattr_s *imattr,
                     char *name,
                     unsigned long flags,
                     unsigned long iflags)
{
  imattr->name = name;
  imattr->flags = flags;
  imattr->iflags = iflags;

  imattr->nr_targets = 0;
  imattr->targets = NULL;
}

void
hwloc_internal_memattrs_prepare(struct hwloc_topology *topology)
{
  topology->memattrs = malloc(HWLOC_MEMATTR_ID_MAX * sizeof(*topology->memattrs));
  if (!topology->memattrs)
    return;

  hwloc__setup_memattr(&topology->memattrs[HWLOC_MEMATTR_ID_CAPACITY],
                       (char *) "Capacity",
                       HWLOC_MEMATTR_FLAG_HIGHER_FIRST,
                       HWLOC_IMATTR_FLAG_STATIC_NAME|HWLOC_IMATTR_FLAG_CONVENIENCE);

  hwloc__setup_memattr(&topology->memattrs[HWLOC_MEMATTR_ID_LOCALITY],
                       (char *) "Locality",
                       HWLOC_MEMATTR_FLAG_LOWER_FIRST,
                       HWLOC_IMATTR_FLAG_STATIC_NAME|HWLOC_IMATTR_FLAG_CONVENIENCE);

  hwloc__setup_memattr(&topology->memattrs[HWLOC_MEMATTR_ID_BANDWIDTH],
                       (char *) "Bandwidth",
                       HWLOC_MEMATTR_FLAG_HIGHER_FIRST|HWLOC_MEMATTR_FLAG_NEED_INITIATOR,
                       HWLOC_IMATTR_FLAG_STATIC_NAME);

  hwloc__setup_memattr(&topology->memattrs[HWLOC_MEMATTR_ID_READ_BANDWIDTH],
                       (char *) "ReadBandwidth",
                       HWLOC_MEMATTR_FLAG_HIGHER_FIRST|HWLOC_MEMATTR_FLAG_NEED_INITIATOR,
                       HWLOC_IMATTR_FLAG_STATIC_NAME);

  hwloc__setup_memattr(&topology->memattrs[HWLOC_MEMATTR_ID_WRITE_BANDWIDTH],
                       (char *) "WriteBandwidth",
                       HWLOC_MEMATTR_FLAG_HIGHER_FIRST|HWLOC_MEMATTR_FLAG_NEED_INITIATOR,
                       HWLOC_IMATTR_FLAG_STATIC_NAME);

  hwloc__setup_memattr(&topology->memattrs[HWLOC_MEMATTR_ID_LATENCY],
                       (char *) "Latency",
                       HWLOC_MEMATTR_FLAG_LOWER_FIRST|HWLOC_MEMATTR_FLAG_NEED_INITIATOR,
                       HWLOC_IMATTR_FLAG_STATIC_NAME);

  hwloc__setup_memattr(&topology->memattrs[HWLOC_MEMATTR_ID_READ_LATENCY],
                       (char *) "ReadLatency",
                       HWLOC_MEMATTR_FLAG_LOWER_FIRST|HWLOC_MEMATTR_FLAG_NEED_INITIATOR,
                       HWLOC_IMATTR_FLAG_STATIC_NAME);

  hwloc__setup_memattr(&topology->memattrs[HWLOC_MEMATTR_ID_WRITE_LATENCY],
                       (char *) "WriteLatency",
                       HWLOC_MEMATTR_FLAG_LOWER_FIRST|HWLOC_MEMATTR_FLAG_NEED_INITIATOR,
                       HWLOC_IMATTR_FLAG_STATIC_NAME);

  topology->nr_memattrs = HWLOC_MEMATTR_ID_MAX;
}

static void
hwloc__imi_destroy(struct hwloc_internal_memattr_initiator_s *imi)
{
  if (imi->initiator.type == HWLOC_LOCATION_TYPE_CPUSET)
    hwloc_bitmap_free(imi->initiator.location.cpuset);
}

static void
hwloc__imtg_destroy(struct hwloc_internal_memattr_s *imattr,
                    struct hwloc_internal_memattr_target_s *imtg)
{
  if (imattr->flags & HWLOC_MEMATTR_FLAG_NEED_INITIATOR) {
    /* only attributes with initiators may have something to free() in the array */
    unsigned k;
    for(k=0; k<imtg->nr_initiators; k++)
      hwloc__imi_destroy(&imtg->initiators[k]);
  }
  free(imtg->initiators);
}

void
hwloc_internal_memattrs_destroy(struct hwloc_topology *topology)
{
  unsigned id;
  for(id=0; id<topology->nr_memattrs; id++) {
    struct hwloc_internal_memattr_s *imattr = &topology->memattrs[id];
    unsigned j;
    for(j=0; j<imattr->nr_targets; j++)
      hwloc__imtg_destroy(imattr, &imattr->targets[j]);
    free(imattr->targets);
    if (!(imattr->iflags & HWLOC_IMATTR_FLAG_STATIC_NAME))
      free(imattr->name);
  }
  free(topology->memattrs);

  topology->memattrs = NULL;
  topology->nr_memattrs = 0;
}

int
hwloc_internal_memattrs_dup(struct hwloc_topology *new, struct hwloc_topology *old)
{
  struct hwloc_tma *tma = new->tma;
  struct hwloc_internal_memattr_s *imattrs;
  hwloc_memattr_id_t id;

  /* old->nr_memattrs is always > 0 thanks to default memattrs */

  imattrs = hwloc_tma_malloc(tma, old->nr_memattrs * sizeof(*imattrs));
  if (!imattrs)
    return -1;
  new->memattrs = imattrs;
  new->nr_memattrs = old->nr_memattrs;
  memcpy(imattrs, old->memattrs, old->nr_memattrs * sizeof(*imattrs));

  for(id=0; id<old->nr_memattrs; id++) {
    struct hwloc_internal_memattr_s *oimattr = &old->memattrs[id];
    struct hwloc_internal_memattr_s *nimattr = &imattrs[id];
    unsigned j;

    assert(oimattr->name);
    nimattr->name = hwloc_tma_strdup(tma, oimattr->name);
    if (!nimattr->name) {
      assert(!tma || !tma->dontfree); /* this tma cannot fail to allocate */
      new->nr_memattrs = id;
      goto failed;
    }
    nimattr->iflags &= ~HWLOC_IMATTR_FLAG_STATIC_NAME;
    nimattr->iflags &= ~HWLOC_IMATTR_FLAG_CACHE_VALID; /* cache will need refresh */

    if (!oimattr->nr_targets)
      continue;

    nimattr->targets = hwloc_tma_malloc(tma, oimattr->nr_targets * sizeof(*nimattr->targets));
    if (!nimattr->targets) {
      free(nimattr->name);
      new->nr_memattrs = id;
      goto failed;
    }
    memcpy(nimattr->targets, oimattr->targets, oimattr->nr_targets * sizeof(*nimattr->targets));

    for(j=0; j<oimattr->nr_targets; j++) {
      struct hwloc_internal_memattr_target_s *oimtg = &oimattr->targets[j];
      struct hwloc_internal_memattr_target_s *nimtg = &nimattr->targets[j];
      unsigned k;

      nimtg->obj = NULL; /* cache will need refresh */

      if (!oimtg->nr_initiators)
        continue;

      nimtg->initiators = hwloc_tma_malloc(tma, oimtg->nr_initiators * sizeof(*nimtg->initiators));
      if (!nimtg->initiators) {
        nimattr->nr_targets = j;
        new->nr_memattrs = id+1;
        goto failed;
      }
      memcpy(nimtg->initiators, oimtg->initiators, oimtg->nr_initiators * sizeof(*nimtg->initiators));

      for(k=0; k<oimtg->nr_initiators; k++) {
        struct hwloc_internal_memattr_initiator_s *oimi = &oimtg->initiators[k];
        struct hwloc_internal_memattr_initiator_s *nimi = &nimtg->initiators[k];
        if (oimi->initiator.type == HWLOC_LOCATION_TYPE_CPUSET) {
          nimi->initiator.location.cpuset = hwloc_bitmap_tma_dup(tma, oimi->initiator.location.cpuset);
          if (!nimi->initiator.location.cpuset) {
            nimtg->nr_initiators = k;
            nimattr->nr_targets = j+1;
            new->nr_memattrs = id+1;
            goto failed;
          }
        } else if (oimi->initiator.type == HWLOC_LOCATION_TYPE_OBJECT) {
          nimi->initiator.location.object.obj = NULL; /* cache will need refresh */
        }
      }
    }
  }
  return 0;

 failed:
  hwloc_internal_memattrs_destroy(new);
  return -1;
}

int
hwloc_memattr_get_by_name(hwloc_topology_t topology,
                          const char *name,
                          hwloc_memattr_id_t *idp)
{
  unsigned id;
  for(id=0; id<topology->nr_memattrs; id++) {
    if (!strcmp(topology->memattrs[id].name, name)) {
      *idp = id;
      return 0;
    }
  }
  errno = EINVAL;
  return -1;
}

int
hwloc_memattr_get_name(hwloc_topology_t topology,
                       hwloc_memattr_id_t id,
                       const char **namep)
{
  if (id >= topology->nr_memattrs) {
    errno = EINVAL;
    return -1;
  }
  *namep = topology->memattrs[id].name;
  return 0;
}

int
hwloc_memattr_get_flags(hwloc_topology_t topology,
                        hwloc_memattr_id_t id,
                        unsigned long *flagsp)
{
  if (id >= topology->nr_memattrs) {
    errno = EINVAL;
    return -1;
  }
  *flagsp = topology->memattrs[id].flags;
  return 0;
}

int
hwloc_memattr_register(hwloc_topology_t topology,
                       const char *_name,
                       unsigned long flags,
                       hwloc_memattr_id_t *id)
{
  struct hwloc_internal_memattr_s *newattrs;
  char *name;
  unsigned i;

  /* check flags */
  if (flags & ~(HWLOC_MEMATTR_FLAG_NEED_INITIATOR|HWLOC_MEMATTR_FLAG_LOWER_FIRST|HWLOC_MEMATTR_FLAG_HIGHER_FIRST)) {
    errno = EINVAL;
    return -1;
  }
  if (!(flags & (HWLOC_MEMATTR_FLAG_LOWER_FIRST|HWLOC_MEMATTR_FLAG_HIGHER_FIRST))) {
    errno = EINVAL;
    return -1;
  }
  if ((flags & (HWLOC_MEMATTR_FLAG_LOWER_FIRST|HWLOC_MEMATTR_FLAG_HIGHER_FIRST))
      == (HWLOC_MEMATTR_FLAG_LOWER_FIRST|HWLOC_MEMATTR_FLAG_HIGHER_FIRST)) {
    errno = EINVAL;
    return -1;
  }

  if (!_name) {
    errno = EINVAL;
    return -1;
  }

  /* check name isn't already used */
  for(i=0; i<topology->nr_memattrs; i++) {
    if (!strcmp(_name, topology->memattrs[i].name)) {
      errno = EBUSY;
      return -1;
    }
  }

  name = strdup(_name);
  if (!name)
    return -1;

  newattrs = realloc(topology->memattrs, (topology->nr_memattrs + 1) * sizeof(*topology->memattrs));
  if (!newattrs) {
    free(name);
    return -1;
  }

  hwloc__setup_memattr(&newattrs[topology->nr_memattrs],
                       name, flags, 0);

  /* memattr valid when just created */
  newattrs[topology->nr_memattrs].iflags |= HWLOC_IMATTR_FLAG_CACHE_VALID;

  *id = topology->nr_memattrs;
  topology->nr_memattrs++;
  topology->memattrs = newattrs;
  return 0;
}


/***************************
 * Internal Locations
 */

/* return 1 if cpuset/obj matchs the existing initiator location,
 * for instance if the cpuset of query is included in the cpuset of existing
 */
static int
match_internal_location(struct hwloc_internal_location_s *iloc,
                        struct hwloc_internal_memattr_initiator_s *imi)
{
  if (iloc->type != imi->initiator.type)
    return 0;
  switch (iloc->type) {
  case HWLOC_LOCATION_TYPE_CPUSET:
    return hwloc_bitmap_isincluded(iloc->location.cpuset, imi->initiator.location.cpuset);
  case HWLOC_LOCATION_TYPE_OBJECT:
    return iloc->location.object.type == imi->initiator.location.object.type
      && iloc->location.object.gp_index == imi->initiator.location.object.gp_index;
  default:
    return 0;
  }
}

static int
to_internal_location(struct hwloc_internal_location_s *iloc,
                     struct hwloc_location *location)
{
  iloc->type = location->type;

  switch (location->type) {
  case HWLOC_LOCATION_TYPE_CPUSET:
    if (!location->location.cpuset || hwloc_bitmap_iszero(location->location.cpuset)) {
      errno = EINVAL;
      return -1;
    }
    iloc->location.cpuset = location->location.cpuset;
    return 0;
  case HWLOC_LOCATION_TYPE_OBJECT:
    if (!location->location.object) {
      errno = EINVAL;
      return -1;
    }
    iloc->location.object.gp_index = location->location.object->gp_index;
    iloc->location.object.type = location->location.object->type;
    return 0;
  default:
    errno = EINVAL;
    return -1;
  }
}

static int
from_internal_location(struct hwloc_internal_location_s *iloc,
                       struct hwloc_location *location)
{
  location->type = iloc->type;

  switch (iloc->type) {
  case HWLOC_LOCATION_TYPE_CPUSET:
    location->location.cpuset = iloc->location.cpuset;
    return 0;
  case HWLOC_LOCATION_TYPE_OBJECT:
    /* requires the cache to be refreshed */
    location->location.object = iloc->location.object.obj;
    if (!location->location.object)
      return -1;
    return 0;
  default:
    errno = EINVAL;
    return -1;
  }
}


/************************
 * Refreshing
 */

static int
hwloc__imi_refresh(struct hwloc_topology *topology,
                   struct hwloc_internal_memattr_initiator_s *imi)
{
  switch (imi->initiator.type) {
  case HWLOC_LOCATION_TYPE_CPUSET: {
    hwloc_bitmap_and(imi->initiator.location.cpuset, imi->initiator.location.cpuset, topology->levels[0][0]->cpuset);
    if (hwloc_bitmap_iszero(imi->initiator.location.cpuset)) {
      hwloc__imi_destroy(imi);
      return -1;
    }
    return 0;
  }
  case HWLOC_LOCATION_TYPE_OBJECT: {
    hwloc_obj_t obj = hwloc_get_obj_by_type_and_gp_index(topology,
                                                         imi->initiator.location.object.type,
                                                         imi->initiator.location.object.gp_index);
    if (!obj) {
      hwloc__imi_destroy(imi);
      return -1;
    }
    imi->initiator.location.object.obj = obj;
    return 0;
  }
  default:
    assert(0);
  }
  return -1;
}

static int
hwloc__imtg_refresh(struct hwloc_topology *topology,
                    struct hwloc_internal_memattr_s *imattr,
                    struct hwloc_internal_memattr_target_s *imtg)
{
  hwloc_obj_t node;

  /* no need to refresh convenience memattrs */
  assert(!(imattr->iflags & HWLOC_IMATTR_FLAG_CONVENIENCE));

  /* check the target object */
  if (imtg->gp_index == (hwloc_uint64_t) -1) {
    /* only NUMA and PU may work with os_index, and only NUMA is currently used internally */
    if (imtg->type == HWLOC_OBJ_NUMANODE)
      node = hwloc_get_numanode_obj_by_os_index(topology, imtg->os_index);
    else if (imtg->type == HWLOC_OBJ_PU)
      node = hwloc_get_pu_obj_by_os_index(topology, imtg->os_index);
    else
      node = NULL;
  } else {
    node = hwloc_get_obj_by_type_and_gp_index(topology, imtg->type, imtg->gp_index);
  }
  if (!node) {
    hwloc__imtg_destroy(imattr, imtg);
    return -1;
  }

  /* save the gp_index in case it wasn't initialized yet */
  imtg->gp_index = node->gp_index;
  /* cache the object */
  imtg->obj = node;

  if (imattr->flags & HWLOC_MEMATTR_FLAG_NEED_INITIATOR) {
    /* check the initiators */
    unsigned k, l;
    for(k=0, l=0; k<imtg->nr_initiators; k++) {
      int err = hwloc__imi_refresh(topology, &imtg->initiators[k]);
      if (err < 0)
        continue;
      if (k != l)
        memcpy(&imtg->initiators[l], &imtg->initiators[k], sizeof(*imtg->initiators));
      l++;
    }
    imtg->nr_initiators = l;
    if (!imtg->nr_initiators) {
      hwloc__imtg_destroy(imattr, imtg);
      return -1;
    }
  }
  return 0;
}

static void
hwloc__imattr_refresh(struct hwloc_topology *topology,
                      struct hwloc_internal_memattr_s *imattr)
{
  unsigned j, k;
  for(j=0, k=0; j<imattr->nr_targets; j++) {
    int ret = hwloc__imtg_refresh(topology, imattr, &imattr->targets[j]);
    if (!ret) {
      /* target still valid, move it if some former targets were removed */
      if (j != k)
        memcpy(&imattr->targets[k], &imattr->targets[j], sizeof(*imattr->targets));
      k++;
    }
  }
  imattr->nr_targets = k;
  imattr->iflags |= HWLOC_IMATTR_FLAG_CACHE_VALID;
}

void
hwloc_internal_memattrs_refresh(struct hwloc_topology *topology)
{
  unsigned id;
  for(id=0; id<topology->nr_memattrs; id++) {
    struct hwloc_internal_memattr_s *imattr = &topology->memattrs[id];
    if (imattr->iflags & HWLOC_IMATTR_FLAG_CACHE_VALID)
      /* nothing to refresh */
      continue;
    hwloc__imattr_refresh(topology, imattr);
  }
}

void
hwloc_internal_memattrs_need_refresh(struct hwloc_topology *topology)
{
  unsigned id;
  for(id=0; id<topology->nr_memattrs; id++) {
    struct hwloc_internal_memattr_s *imattr = &topology->memattrs[id];
    if (imattr->iflags & HWLOC_IMATTR_FLAG_CONVENIENCE)
      /* no need to refresh convenience memattrs */
      continue;
    imattr->iflags &= ~HWLOC_IMATTR_FLAG_CACHE_VALID;
  }
}


/********************************
 * Targets
 */

static struct hwloc_internal_memattr_target_s *
hwloc__memattr_get_target(struct hwloc_internal_memattr_s *imattr,
                          hwloc_obj_type_t target_type,
                          hwloc_uint64_t target_gp_index,
                          unsigned target_os_index,
                          int create)
{
  struct hwloc_internal_memattr_target_s *news, *new;
  unsigned j;

  for(j=0; j<imattr->nr_targets; j++) {
    if (target_type == imattr->targets[j].type)
      if ((target_gp_index != (hwloc_uint64_t)-1 && target_gp_index == imattr->targets[j].gp_index)
          || (target_os_index != (unsigned)-1 && target_os_index == imattr->targets[j].os_index))
        return &imattr->targets[j];
  }
  if (!create)
    return NULL;

  news = realloc(imattr->targets, (imattr->nr_targets+1)*sizeof(*imattr->targets));
  if (!news)
    return NULL;
  imattr->targets = news;

  /* FIXME sort targets? by logical index at the end of load? */

  new = &news[imattr->nr_targets];
  new->type = target_type;
  new->gp_index = target_gp_index;
  new->os_index = target_os_index;

  /* cached object will be refreshed later on actual access */
  new->obj = NULL;
  imattr->iflags &= ~HWLOC_IMATTR_FLAG_CACHE_VALID;
  /* When setting a value after load(), the caller has the target object
   * (and initiator object, if not CPU set). Hence, we could avoid invalidating
   * the cache here.
   * The overhead of the imattr-wide refresh isn't high enough so far
   * to justify making the cache management more complex.
   */

  new->nr_initiators = 0;
  new->initiators = NULL;
  new->noinitiator_value = 0;
  imattr->nr_targets++;
  return new;
}

static struct hwloc_internal_memattr_initiator_s *
hwloc__memattr_get_initiator_from_location(struct hwloc_internal_memattr_s *imattr,
                                           struct hwloc_internal_memattr_target_s *imtg,
                                           struct hwloc_location *location);

int
hwloc_memattr_get_targets(hwloc_topology_t topology,
                          hwloc_memattr_id_t id,
                          struct hwloc_location *initiator,
                          unsigned long flags,
                          unsigned *nrp, hwloc_obj_t *targets, hwloc_uint64_t *values)
{
  struct hwloc_internal_memattr_s *imattr;
  unsigned i, found = 0, max;

  if (flags) {
    errno = EINVAL;
    return -1;
  }

  if (!nrp || (*nrp && !targets)) {
    errno = EINVAL;
    return -1;
  }
  max = *nrp;

  if (id >= topology->nr_memattrs) {
    errno = EINVAL;
    return -1;
  }
  imattr = &topology->memattrs[id];

  if (imattr->iflags & HWLOC_IMATTR_FLAG_CONVENIENCE) {
    /* convenience attributes */
    for(i=0; ; i++) {
      hwloc_obj_t node = hwloc_get_obj_by_type(topology, HWLOC_OBJ_NUMANODE, i);
      if (!node)
        break;
      if (found<max) {
        targets[found] = node;
        if (values)
          values[found] = hwloc__memattr_get_convenience_value(id, node);
      }
      found++;
    }
    goto done;
  }

  /* normal attributes */

  if (!(imattr->iflags & HWLOC_IMATTR_FLAG_CACHE_VALID))
    hwloc__imattr_refresh(topology, imattr);

  for(i=0; i<imattr->nr_targets; i++) {
    struct hwloc_internal_memattr_target_s *imtg = &imattr->targets[i];
    hwloc_uint64_t value = 0;

    if (imattr->flags & HWLOC_MEMATTR_FLAG_NEED_INITIATOR) {
      if (initiator) {
        /* find a matching initiator */
        struct hwloc_internal_memattr_initiator_s *imi = hwloc__memattr_get_initiator_from_location(imattr, imtg, initiator);
        if (!imi)
          continue;
        value = imi->value;
      }
    } else {
      value = imtg->noinitiator_value;
    }

    if (found<max) {
      targets[found] = imtg->obj;
      if (values)
        values[found] = value;
    }
    found++;
  }

 done:
  *nrp = found;
  return 0;
}


/************************
 * Initiators
 */

static struct hwloc_internal_memattr_initiator_s *
hwloc__memattr_target_get_initiator(struct hwloc_internal_memattr_target_s *imtg,
                                    struct hwloc_internal_location_s *iloc,
                                    int create)
{
  struct hwloc_internal_memattr_initiator_s *news, *new;
  unsigned k;

  for(k=0; k<imtg->nr_initiators; k++) {
    struct hwloc_internal_memattr_initiator_s *imi = &imtg->initiators[k];
    if (match_internal_location(iloc, imi)) {
      return imi;
    }
  }

  if (!create)
    return NULL;

  news = realloc(imtg->initiators, (imtg->nr_initiators+1)*sizeof(*imtg->initiators));
  if (!news)
    return NULL;
  new = &news[imtg->nr_initiators];

  new->initiator = *iloc;
  if (iloc->type == HWLOC_LOCATION_TYPE_CPUSET) {
    new->initiator.location.cpuset = hwloc_bitmap_dup(iloc->location.cpuset);
    if (!new->initiator.location.cpuset)
      goto out_with_realloc;
  }

  imtg->nr_initiators++;
  imtg->initiators = news;
  return new;

 out_with_realloc:
  imtg->initiators = news;
  return NULL;
}

static struct hwloc_internal_memattr_initiator_s *
hwloc__memattr_get_initiator_from_location(struct hwloc_internal_memattr_s *imattr,
                                           struct hwloc_internal_memattr_target_s *imtg,
                                           struct hwloc_location *location)
{
  struct hwloc_internal_memattr_initiator_s *imi;
  struct hwloc_internal_location_s iloc;

  assert(imattr->flags & HWLOC_MEMATTR_FLAG_NEED_INITIATOR);

  /* use the initiator value */
  if (!location) {
    errno = EINVAL;
    return NULL;
  }

  if (to_internal_location(&iloc, location) < 0) {
    errno = EINVAL;
    return NULL;
  }

  imi = hwloc__memattr_target_get_initiator(imtg, &iloc, 0);
  if (!imi) {
    errno = EINVAL;
    return NULL;
  }

  return imi;
}

int
hwloc_memattr_get_initiators(hwloc_topology_t topology,
                             hwloc_memattr_id_t id,
                             hwloc_obj_t target_node,
                             unsigned long flags,
                             unsigned *nrp, struct hwloc_location *initiators, hwloc_uint64_t *values)
{
  struct hwloc_internal_memattr_s *imattr;
  struct hwloc_internal_memattr_target_s *imtg;
  unsigned i, max;

  if (flags) {
    errno = EINVAL;
    return -1;
  }

  if (!nrp || (*nrp && !initiators)) {
    errno = EINVAL;
    return -1;
  }
  max = *nrp;

  if (id >= topology->nr_memattrs) {
    errno = EINVAL;
    return -1;
  }
  imattr = &topology->memattrs[id];
  if (!(imattr->flags & HWLOC_MEMATTR_FLAG_NEED_INITIATOR)) {
    *nrp = 0;
    return 0;
  }

  /* all convenience attributes have no initiators */
  assert(!(imattr->iflags & HWLOC_IMATTR_FLAG_CONVENIENCE));

  if (!(imattr->iflags & HWLOC_IMATTR_FLAG_CACHE_VALID))
    hwloc__imattr_refresh(topology, imattr);

  imtg = hwloc__memattr_get_target(imattr, target_node->type, target_node->gp_index, target_node->os_index, 0);
  if (!imtg) {
    errno = EINVAL;
    return -1;
  }

  for(i=0; i<imtg->nr_initiators && i<max; i++) {
    struct hwloc_internal_memattr_initiator_s *imi = &imtg->initiators[i];
    int err = from_internal_location(&imi->initiator, &initiators[i]);
    assert(!err);
    if (values)
      /* no need to handle capacity/locality special cases here, those are initiator-less attributes */
      values[i] = imi->value;
  }

  *nrp = imtg->nr_initiators;
  return 0;
}


/**************************
 * Values
 */

int
hwloc_memattr_get_value(hwloc_topology_t topology,
                        hwloc_memattr_id_t id,
                        hwloc_obj_t target_node,
                        struct hwloc_location *initiator,
                        unsigned long flags,
                        hwloc_uint64_t *valuep)
{
  struct hwloc_internal_memattr_s *imattr;
  struct hwloc_internal_memattr_target_s *imtg;

  if (flags) {
    errno = EINVAL;
    return -1;
  }

  if (id >= topology->nr_memattrs) {
    errno = EINVAL;
    return -1;
  }
  imattr = &topology->memattrs[id];

  if (imattr->iflags & HWLOC_IMATTR_FLAG_CONVENIENCE) {
    /* convenience attributes */
    *valuep = hwloc__memattr_get_convenience_value(id, target_node);
    return 0;
  }

  /* normal attributes */

  if (!(imattr->iflags & HWLOC_IMATTR_FLAG_CACHE_VALID))
    hwloc__imattr_refresh(topology, imattr);

  imtg = hwloc__memattr_get_target(imattr, target_node->type, target_node->gp_index, target_node->os_index, 0);
  if (!imtg) {
    errno = EINVAL;
    return -1;
  }

  if (imattr->flags & HWLOC_MEMATTR_FLAG_NEED_INITIATOR) {
    /* find the initiator and set its value */
    struct hwloc_internal_memattr_initiator_s *imi = hwloc__memattr_get_initiator_from_location(imattr, imtg, initiator);
    if (!imi)
      return -1;
    *valuep = imi->value;
  } else {
    /* get the no-initiator value */
    *valuep = imtg->noinitiator_value;
  }
  return 0;
}

static int
hwloc__internal_memattr_set_value(hwloc_topology_t topology,
                                  hwloc_memattr_id_t id,
                                  hwloc_obj_type_t target_type,
                                  hwloc_uint64_t target_gp_index,
                                  unsigned target_os_index,
                                  struct hwloc_internal_location_s *initiator,
                                  hwloc_uint64_t value)
{
  struct hwloc_internal_memattr_s *imattr;
  struct hwloc_internal_memattr_target_s *imtg;

  if (id >= topology->nr_memattrs) {
    /* something bad happened during init */
    errno = EINVAL;
    return -1;
  }
  imattr = &topology->memattrs[id];

  if (imattr->flags & HWLOC_MEMATTR_FLAG_NEED_INITIATOR) {
    /* check given initiator */
    if (!initiator) {
      errno = EINVAL;
      return -1;
    }
  }

  if (imattr->iflags & HWLOC_IMATTR_FLAG_CONVENIENCE) {
    /* convenience attributes are read-only */
    errno = EINVAL;
    return -1;
  }

  if (topology->is_loaded && !(imattr->iflags & HWLOC_IMATTR_FLAG_CACHE_VALID))
    /* don't refresh when adding values during load (some nodes might not be ready yet),
     * we'll refresh later
     */
    hwloc__imattr_refresh(topology, imattr);

  imtg = hwloc__memattr_get_target(imattr, target_type, target_gp_index, target_os_index, 1);
  if (!imtg)
    return -1;

  if (imattr->flags & HWLOC_MEMATTR_FLAG_NEED_INITIATOR) {
    /* find/add the initiator and set its value */
    // FIXME what if cpuset is larger than an existing one ?
    struct hwloc_internal_memattr_initiator_s *imi = hwloc__memattr_target_get_initiator(imtg, initiator, 1);
    if (!imi)
      return -1;
    imi->value = value;

  } else {
    /* set the no-initiator value */
    imtg->noinitiator_value = value;
  }

  return 0;

}

int
hwloc_internal_memattr_set_value(hwloc_topology_t topology,
                                 hwloc_memattr_id_t id,
                                 hwloc_obj_type_t target_type,
                                 hwloc_uint64_t target_gp_index,
                                 unsigned target_os_index,
                                 struct hwloc_internal_location_s *initiator,
                                 hwloc_uint64_t value)
{
  assert(id != HWLOC_MEMATTR_ID_CAPACITY);
  assert(id != HWLOC_MEMATTR_ID_LOCALITY);

  return hwloc__internal_memattr_set_value(topology, id, target_type, target_gp_index, target_os_index, initiator, value);
}

int
hwloc_memattr_set_value(hwloc_topology_t topology,
                        hwloc_memattr_id_t id,
                        hwloc_obj_t target_node,
                        struct hwloc_location *initiator,
                        unsigned long flags,
                        hwloc_uint64_t value)
{
  struct hwloc_internal_location_s iloc, *ilocp;

  if (flags) {
    errno = EINVAL;
    return -1;
  }

  if (initiator) {
    if (to_internal_location(&iloc, initiator) < 0) {
      errno = EINVAL;
      return -1;
    }
    ilocp = &iloc;
  } else {
    ilocp = NULL;
  }

  return hwloc__internal_memattr_set_value(topology, id, target_node->type, target_node->gp_index, target_node->os_index, ilocp, value);
}


/**********************
 * Best target
 */

static void
hwloc__update_best_target(hwloc_obj_t *best_obj, hwloc_uint64_t *best_value, int *found,
                          hwloc_obj_t new_obj, hwloc_uint64_t new_value,
                          int keep_highest)
{
  if (*found) {
    if (keep_highest) {
      if (new_value <= *best_value)
        return;
    } else {
      if (new_value >= *best_value)
        return;
    }
  }

  *best_obj = new_obj;
  *best_value = new_value;
  *found = 1;
}

int
hwloc_memattr_get_best_target(hwloc_topology_t topology,
                              hwloc_memattr_id_t id,
                              struct hwloc_location *initiator,
                              unsigned long flags,
                              hwloc_obj_t *bestp, hwloc_uint64_t *valuep)
{
  struct hwloc_internal_memattr_s *imattr;
  hwloc_uint64_t best_value = 0; /* shutup the compiler */
  hwloc_obj_t best = NULL;
  int found = 0;
  unsigned j;

  if (flags) {
    errno = EINVAL;
    return -1;
  }

  if (id >= topology->nr_memattrs) {
    errno = EINVAL;
    return -1;
  }
  imattr = &topology->memattrs[id];

  if (imattr->iflags & HWLOC_IMATTR_FLAG_CONVENIENCE) {
    /* convenience attributes */
    for(j=0; ; j++) {
      hwloc_obj_t node = hwloc_get_obj_by_type(topology, HWLOC_OBJ_NUMANODE, j);
      hwloc_uint64_t value;
      if (!node)
        break;
      value = hwloc__memattr_get_convenience_value(id, node);
      hwloc__update_best_target(&best, &best_value, &found,
                                node, value,
                                imattr->flags & HWLOC_MEMATTR_FLAG_HIGHER_FIRST);
    }
    goto done;
  }

  /* normal attributes */

  if (!(imattr->iflags & HWLOC_IMATTR_FLAG_CACHE_VALID))
    /* not strictly need */
    hwloc__imattr_refresh(topology, imattr);

  for(j=0; j<imattr->nr_targets; j++) {
    struct hwloc_internal_memattr_target_s *imtg = &imattr->targets[j];
    hwloc_uint64_t value;
    if (imattr->flags & HWLOC_MEMATTR_FLAG_NEED_INITIATOR) {
      /* find the initiator and set its value */
      struct hwloc_internal_memattr_initiator_s *imi = hwloc__memattr_get_initiator_from_location(imattr, imtg, initiator);
      if (!imi)
        continue;
      value = imi->value;
    } else {
      /* get the no-initiator value */
      value = imtg->noinitiator_value;
    }
    hwloc__update_best_target(&best, &best_value, &found,
                              imtg->obj, value,
                              imattr->flags & HWLOC_MEMATTR_FLAG_HIGHER_FIRST);
  }

 done:
  if (found) {
    assert(best);
    *bestp = best;
    if (valuep)
      *valuep = best_value;
    return 0;
  } else {
    errno = ENOENT;
    return -1;
  }
}

/**********************
 * Best initiators
 */

static void
hwloc__update_best_initiator(struct hwloc_internal_location_s *best_initiator, hwloc_uint64_t *best_value, int *found,
                             struct hwloc_internal_location_s *new_initiator, hwloc_uint64_t new_value,
                             int keep_highest)
{
  if (*found) {
    if (keep_highest) {
      if (new_value <= *best_value)
        return;
    } else {
      if (new_value >= *best_value)
        return;
    }
  }

  *best_initiator = *new_initiator;
  *best_value = new_value;
  *found = 1;
}

int
hwloc_memattr_get_best_initiator(hwloc_topology_t topology,
                                 hwloc_memattr_id_t id,
                                 hwloc_obj_t target_node,
                                 unsigned long flags,
                                 struct hwloc_location *bestp, hwloc_uint64_t *valuep)
{
  struct hwloc_internal_memattr_s *imattr;
  struct hwloc_internal_memattr_target_s *imtg;
  struct hwloc_internal_location_s best_initiator;
  hwloc_uint64_t best_value;
  int found;
  unsigned i;

  if (flags) {
    errno = EINVAL;
    return -1;
  }

  if (id >= topology->nr_memattrs) {
    errno = EINVAL;
    return -1;
  }
  imattr = &topology->memattrs[id];

  if (!(imattr->flags & HWLOC_MEMATTR_FLAG_NEED_INITIATOR)) {
    errno = EINVAL;
    return -1;
  }

  if (!(imattr->iflags & HWLOC_IMATTR_FLAG_CACHE_VALID))
    /* not strictly need */
    hwloc__imattr_refresh(topology, imattr);

  imtg = hwloc__memattr_get_target(imattr, target_node->type, target_node->gp_index, target_node->os_index, 0);
  if (!imtg) {
    errno = EINVAL;
    return -1;
  }

  found = 0;
  for(i=0; i<imtg->nr_initiators; i++) {
    struct hwloc_internal_memattr_initiator_s *imi = &imtg->initiators[i];
    hwloc__update_best_initiator(&best_initiator, &best_value, &found,
                                 &imi->initiator, imi->value,
                                 imattr->flags & HWLOC_MEMATTR_FLAG_HIGHER_FIRST);
  }

  if (found) {
    if (valuep)
      *valuep = best_value;
    return from_internal_location(&best_initiator, bestp);
  } else {
    errno = ENOENT;
    return -1;
  }
}

/****************************
 * Listing local nodes
 */

static __hwloc_inline int
match_local_obj_cpuset(hwloc_obj_t node, hwloc_cpuset_t cpuset, unsigned long flags)
{
  if (flags & HWLOC_LOCAL_NUMANODE_FLAG_ALL)
    return 1;
  if ((flags & HWLOC_LOCAL_NUMANODE_FLAG_LARGER_LOCALITY)
      && hwloc_bitmap_isincluded(cpuset, node->cpuset))
    return 1;
  if ((flags & HWLOC_LOCAL_NUMANODE_FLAG_SMALLER_LOCALITY)
      && hwloc_bitmap_isincluded(node->cpuset, cpuset))
    return 1;
  return hwloc_bitmap_isequal(node->cpuset, cpuset);
}

int
hwloc_get_local_numanode_objs(hwloc_topology_t topology,
                              struct hwloc_location *location,
                              unsigned *nrp,
                              hwloc_obj_t *nodes,
                              unsigned long flags)
{
  hwloc_cpuset_t cpuset;
  hwloc_obj_t node;
  unsigned i;

  if (flags & ~(HWLOC_LOCAL_NUMANODE_FLAG_SMALLER_LOCALITY
                |HWLOC_LOCAL_NUMANODE_FLAG_LARGER_LOCALITY
                | HWLOC_LOCAL_NUMANODE_FLAG_ALL)) {
    errno = EINVAL;
    return -1;
  }

  if (!nrp || (*nrp && !nodes)) {
    errno = EINVAL;
    return -1;
  }

  if (!location) {
    if (!(flags & HWLOC_LOCAL_NUMANODE_FLAG_ALL)) {
      errno = EINVAL;
      return -1;
    }
    cpuset = NULL; /* unused */

  } else {
    if (location->type == HWLOC_LOCATION_TYPE_CPUSET) {
      cpuset = location->location.cpuset;
    } else if (location->type == HWLOC_LOCATION_TYPE_OBJECT) {
      hwloc_obj_t obj = location->location.object;
      while (!obj->cpuset)
        obj = obj->parent;
      cpuset = obj->cpuset;
    } else {
      errno = EINVAL;
      return -1;
    }
  }

  i = 0;
  for(node = hwloc_get_obj_by_type(topology, HWLOC_OBJ_NUMANODE, 0);
      node;
      node = node->next_cousin) {
    if (!match_local_obj_cpuset(node, cpuset, flags))
      continue;
    if (i < *nrp)
      nodes[i] = node;
    i++;
  }

  *nrp = i;
  return 0;
}


/**************************************
 * Using memattrs to identify HBM/DRAM
 */

enum hwloc_memory_tier_type_e {
  /* WARNING: keep higher BW types first for compare_tiers_by_bw_and_type() when BW info is missing */
  HWLOC_MEMORY_TIER_HBM  = 1UL<<0,
  HWLOC_MEMORY_TIER_DRAM = 1UL<<1,
  HWLOC_MEMORY_TIER_GPU  = 1UL<<2,
  HWLOC_MEMORY_TIER_SPM  = 1UL<<3, /* Specific-Purpose Memory is usually HBM, we'll use BW to confirm or force*/
  HWLOC_MEMORY_TIER_NVM  = 1UL<<4,
  HWLOC_MEMORY_TIER_CXL  = 1UL<<5
};
typedef unsigned long hwloc_memory_tier_type_t;
#define HWLOC_MEMORY_TIER_UNKNOWN 0UL

static const char * hwloc_memory_tier_type_snprintf(hwloc_memory_tier_type_t type)
{
  switch (type) {
  case HWLOC_MEMORY_TIER_DRAM: return "DRAM";
  case HWLOC_MEMORY_TIER_HBM: return "HBM";
  case HWLOC_MEMORY_TIER_GPU: return "GPUMemory";
  case HWLOC_MEMORY_TIER_SPM: return "SPM";
  case HWLOC_MEMORY_TIER_NVM: return "NVM";
  case HWLOC_MEMORY_TIER_CXL:
  case HWLOC_MEMORY_TIER_CXL|HWLOC_MEMORY_TIER_DRAM: return "CXL-DRAM";
  case HWLOC_MEMORY_TIER_CXL|HWLOC_MEMORY_TIER_HBM: return "CXL-HBM";
  case HWLOC_MEMORY_TIER_CXL|HWLOC_MEMORY_TIER_GPU: return "CXL-GPUMemory";
  case HWLOC_MEMORY_TIER_CXL|HWLOC_MEMORY_TIER_SPM: return "CXL-SPM";
  case HWLOC_MEMORY_TIER_CXL|HWLOC_MEMORY_TIER_NVM: return "CXL-NVM";
  default: return NULL;
  }
}

static hwloc_memory_tier_type_t hwloc_memory_tier_type_sscanf(const char *name)
{
  if (!strcasecmp(name, "DRAM"))
    return HWLOC_MEMORY_TIER_DRAM;
  if (!strcasecmp(name, "HBM"))
    return HWLOC_MEMORY_TIER_HBM;
  if (!strcasecmp(name, "GPUMemory"))
    return HWLOC_MEMORY_TIER_GPU;
  if (!strcasecmp(name, "SPM"))
    return HWLOC_MEMORY_TIER_SPM;
  if (!strcasecmp(name, "NVM"))
    return HWLOC_MEMORY_TIER_NVM;
  if (!strcasecmp(name, "CXL-DRAM"))
    return HWLOC_MEMORY_TIER_CXL|HWLOC_MEMORY_TIER_DRAM;
  if (!strcasecmp(name, "CXL-HBM"))
    return HWLOC_MEMORY_TIER_CXL|HWLOC_MEMORY_TIER_HBM;
  if (!strcasecmp(name, "CXL-GPUMemory"))
    return HWLOC_MEMORY_TIER_CXL|HWLOC_MEMORY_TIER_GPU;
  if (!strcasecmp(name, "CXL-SPM"))
    return HWLOC_MEMORY_TIER_CXL|HWLOC_MEMORY_TIER_SPM;
  if (!strcasecmp(name, "CXL-NVM"))
    return HWLOC_MEMORY_TIER_CXL|HWLOC_MEMORY_TIER_NVM;
  return 0;
}

/* factorized tier, grouping multiple nodes */
struct hwloc_memory_tier_s {
  hwloc_nodeset_t nodeset;
  uint64_t local_bw_min, local_bw_max;
  uint64_t local_lat_min, local_lat_max;
  hwloc_memory_tier_type_t type;
};

/* early tier discovery, one entry per node */
struct hwloc_memory_node_info_s {
  hwloc_obj_t node;
  uint64_t local_bw;
  uint64_t local_lat;
  hwloc_memory_tier_type_t type;
  unsigned rank;
};

static int compare_node_infos_by_type_and_bw(const void *_a, const void *_b)
{
  const struct hwloc_memory_node_info_s *a = _a, *b = _b;
  /* sort by type of node first */
  if (a->type != b->type)
    return a->type - b->type;
  /* then by bandwidth */
  if (a->local_bw > b->local_bw)
    return -1;
  else if (a->local_bw < b->local_bw)
    return 1;
  return 0;
}

static int compare_tiers_by_bw_and_type(const void *_a, const void *_b)
{
  const struct hwloc_memory_tier_s *a = _a, *b = _b;
  /* sort by (average) BW first */
  if (a->local_bw_min && b->local_bw_min) {
    if (a->local_bw_min + a->local_bw_max > b->local_bw_min + b->local_bw_max)
      return -1;
    else if (a->local_bw_min + a->local_bw_max < b->local_bw_min + b->local_bw_max)
      return 1;
  }
  /* then by tier type */
  if (a->type != b->type)
    return a->type - b->type;
  return 0;
}

static struct hwloc_memory_tier_s *
hwloc__group_memory_tiers(hwloc_topology_t topology,
                          unsigned *nr_tiers_p)
{
  struct hwloc_internal_memattr_s *imattr_bw, *imattr_lat;
  struct hwloc_memory_node_info_s *nodeinfos;
  struct hwloc_memory_tier_s *tiers;
  unsigned nr_tiers;
  float bw_threshold = 0.1;
  float lat_threshold = 0.1;
  const char *env;
  unsigned i, j, n;

  n = hwloc_get_nbobjs_by_depth(topology, HWLOC_TYPE_DEPTH_NUMANODE);
  assert(n);

  env = getenv("HWLOC_MEMTIERS_BANDWIDTH_THRESHOLD");
  if (env)
    bw_threshold = atof(env);

  env = getenv("HWLOC_MEMTIERS_LATENCY_THRESHOLD");
  if (env)
    lat_threshold = atof(env);

  imattr_bw = &topology->memattrs[HWLOC_MEMATTR_ID_BANDWIDTH];
  imattr_lat = &topology->memattrs[HWLOC_MEMATTR_ID_LATENCY];

  if (!(imattr_bw->iflags & HWLOC_IMATTR_FLAG_CACHE_VALID))
    hwloc__imattr_refresh(topology, imattr_bw);
  if (!(imattr_lat->iflags & HWLOC_IMATTR_FLAG_CACHE_VALID))
    hwloc__imattr_refresh(topology, imattr_lat);

  nodeinfos = malloc(n * sizeof(*nodeinfos));
  if (!nodeinfos)
    return NULL;

  for(i=0; i<n; i++) {
    hwloc_obj_t node;
    const char *daxtype;
    struct hwloc_internal_location_s iloc;
    struct hwloc_internal_memattr_target_s *imtg;

    node = hwloc_get_obj_by_depth(topology, HWLOC_TYPE_DEPTH_NUMANODE, i);
    assert(node);
    nodeinfos[i].node = node;

    /* defaults to unknown */
    nodeinfos[i].type = HWLOC_MEMORY_TIER_UNKNOWN;
    nodeinfos[i].local_bw = 0;
    nodeinfos[i].local_lat = 0;

    daxtype = hwloc_obj_get_info_by_name(node, "DAXType");
    /* mark NVM, SPM and GPU nodes */
    if (node->subtype && !strcmp(node->subtype, "GPUMemory"))
      nodeinfos[i].type = HWLOC_MEMORY_TIER_GPU;
    else if (daxtype && !strcmp(daxtype, "NVM"))
      nodeinfos[i].type = HWLOC_MEMORY_TIER_NVM;
    else if (daxtype && !strcmp(daxtype, "SPM"))
      nodeinfos[i].type = HWLOC_MEMORY_TIER_SPM;
    /* add CXL flag */
    if (hwloc_obj_get_info_by_name(node, "CXLDevice") != NULL) {
      /* CXL is always SPM for now. HBM and DRAM not possible here yet.
       * Hence remove all but NVM first.
       */
      nodeinfos[i].type &= HWLOC_MEMORY_TIER_NVM;
      nodeinfos[i].type |= HWLOC_MEMORY_TIER_CXL;
    }

    /* get local bandwidth */
    imtg = NULL;
    for(j=0; j<imattr_bw->nr_targets; j++)
      if (imattr_bw->targets[j].obj == node) {
        imtg = &imattr_bw->targets[j];
        break;
      }
    if (imtg && !hwloc_bitmap_iszero(node->cpuset)) {
      struct hwloc_internal_memattr_initiator_s *imi;
      iloc.type = HWLOC_LOCATION_TYPE_CPUSET;
      iloc.location.cpuset = node->cpuset;
      imi = hwloc__memattr_target_get_initiator(imtg, &iloc, 0);
      if (imi)
        nodeinfos[i].local_bw = imi->value;
    }
    /* get local latency */
    imtg = NULL;
    for(j=0; j<imattr_lat->nr_targets; j++)
      if (imattr_lat->targets[j].obj == node) {
        imtg = &imattr_lat->targets[j];
        break;
      }
    if (imtg && !hwloc_bitmap_iszero(node->cpuset)) {
      struct hwloc_internal_memattr_initiator_s *imi;
      iloc.type = HWLOC_LOCATION_TYPE_CPUSET;
      iloc.location.cpuset = node->cpuset;
      imi = hwloc__memattr_target_get_initiator(imtg, &iloc, 0);
      if (imi)
        nodeinfos[i].local_lat = imi->value;
    }
  }

  /* Sort nodes.
   * We could also sort by the existing subtype.
   * KNL is the only case where subtypes are set in backends, but we set memattrs as well there.
   * Also HWLOC_MEMTIERS_REFRESH would be a special value to ignore existing subtypes.
   */
  hwloc_debug("Sorting memory node infos...\n");
  qsort(nodeinfos, n, sizeof(*nodeinfos), compare_node_infos_by_type_and_bw);
#ifdef HWLOC_DEBUG
  for(i=0; i<n; i++)
    hwloc_debug("  node info %u = node L#%u P#%u with info type %lx and local BW %llu lat %llu\n",
                i,
                nodeinfos[i].node->logical_index, nodeinfos[i].node->os_index,
                nodeinfos[i].type,
                (unsigned long long) nodeinfos[i].local_bw,
                (unsigned long long) nodeinfos[i].local_lat);
#endif
  /* now we have UNKNOWN nodes (sorted by BW only), then known ones */

  /* iterate among them and add a rank value.
   * start from rank 0 and switch to next rank when the type changes or when the BW or latendy difference is > threshold */
  hwloc_debug("Starting memory tier #0 and iterating over nodes...\n");
  nodeinfos[0].rank = 0;
  for(i=1; i<n; i++) {
    /* reuse the same rank by default */
    nodeinfos[i].rank = nodeinfos[i-1].rank;
    /* comparing type */
    if (nodeinfos[i].type != nodeinfos[i-1].type) {
      hwloc_debug("  Switching to memory tier #%u starting with node L#%u P#%u because of type\n",
                  nodeinfos[i].rank, nodeinfos[i].node->logical_index, nodeinfos[i].node->os_index);
      nodeinfos[i].rank++;
      continue;
    }
    /* comparing bandwidth */
    if (nodeinfos[i].local_bw && nodeinfos[i-1].local_bw) {
      float bw_ratio = (float)nodeinfos[i].local_bw/(float)nodeinfos[i-1].local_bw;
      if (bw_ratio < 1.)
        bw_ratio = 1./bw_ratio;
      if (bw_ratio > 1.0 + bw_threshold) {
        nodeinfos[i].rank++;
        hwloc_debug("  Switching to memory tier #%u starting with node L#%u P#%u because of bandwidth\n",
                    nodeinfos[i].rank, nodeinfos[i].node->logical_index, nodeinfos[i].node->os_index);
        continue;
      }
    }
    /* comparing latency */
    if (nodeinfos[i].local_lat && nodeinfos[i-1].local_lat) {
      float lat_ratio = (float)nodeinfos[i].local_lat/(float)nodeinfos[i-1].local_lat;
      if (lat_ratio < 1.)
        lat_ratio = 1./lat_ratio;
      if (lat_ratio > 1.0 + lat_threshold) {
        hwloc_debug("  Switching to memory tier #%u starting with node L#%u P#%u because of latency\n",
                    nodeinfos[i].rank, nodeinfos[i].node->logical_index, nodeinfos[i].node->os_index);
        nodeinfos[i].rank++;
        continue;
      }
    }
  }
  /* FIXME: if there are cpuset-intersecting nodes in same tier, split again? */
  hwloc_debug("  Found %u tiers total\n", nodeinfos[n-1].rank + 1);

  /* now group nodeinfos into factorized tiers */
  nr_tiers = nodeinfos[n-1].rank + 1;
  tiers = calloc(nr_tiers, sizeof(*tiers));
  if (!tiers)
    goto out_with_nodeinfos;
  for(i=0; i<nr_tiers; i++) {
    tiers[i].nodeset = hwloc_bitmap_alloc();
    if (!tiers[i].nodeset)
      goto out_with_tiers;
    tiers[i].local_bw_min = tiers[i].local_bw_max = 0;
    tiers[i].local_lat_min = tiers[i].local_lat_max = 0;
    tiers[i].type = HWLOC_MEMORY_TIER_UNKNOWN;
  }
  for(i=0; i<n; i++) {
    unsigned rank = nodeinfos[i].rank;
    assert(rank < nr_tiers);
    hwloc_bitmap_set(tiers[rank].nodeset, nodeinfos[i].node->os_index);
    assert(tiers[rank].type == HWLOC_MEMORY_TIER_UNKNOWN
           || tiers[rank].type == nodeinfos[i].type);
    tiers[rank].type = nodeinfos[i].type;
    /* nodeinfos are sorted in BW order, no need to compare */
    if (!tiers[rank].local_bw_min)
      tiers[rank].local_bw_min = nodeinfos[i].local_bw;
    tiers[rank].local_bw_max = nodeinfos[i].local_bw;
    /* compare latencies to update min/max */
    if (!tiers[rank].local_lat_min || nodeinfos[i].local_lat < tiers[rank].local_lat_min)
      tiers[rank].local_lat_min = nodeinfos[i].local_lat;
    if (!tiers[rank].local_lat_max || nodeinfos[i].local_lat > tiers[rank].local_lat_max)
      tiers[rank].local_lat_max = nodeinfos[i].local_lat;
  }

  free(nodeinfos);
  *nr_tiers_p = nr_tiers;
  return tiers;

 out_with_tiers:
  for(i=0; i<nr_tiers; i++)
    hwloc_bitmap_free(tiers[i].nodeset);
  free(tiers);
 out_with_nodeinfos:
  free(nodeinfos);
  return NULL;
}

enum hwloc_guess_memtiers_flag {
  HWLOC_GUESS_MEMTIERS_FLAG_NODE0_IS_DRAM = 1<<0,
  HWLOC_GUESS_MEMTIERS_FLAG_SPM_IS_HBM = 1<<1
};

static int
hwloc__guess_dram_hbm_tiers(struct hwloc_memory_tier_s *tier1,
                            struct hwloc_memory_tier_s *tier2,
                            unsigned long flags)
{
  struct hwloc_memory_tier_s *tmp;

  if (!tier1->local_bw_min || !tier2->local_bw_min) {
    hwloc_debug("    Missing BW info\n");
    return -1;
  }

  /* reorder tiers by BW */
  if (tier1->local_bw_min > tier2->local_bw_min) {
    tmp = tier1; tier1 = tier2; tier2 = tmp;
  }
  /* tier1 < tier2 */

  hwloc_debug("    tier1 BW %llu-%llu vs tier2 BW %llu-%llu\n",
              (unsigned long long) tier1->local_bw_min,
              (unsigned long long) tier1->local_bw_max,
              (unsigned long long) tier2->local_bw_min,
              (unsigned long long) tier2->local_bw_max);
  if (tier2->local_bw_min <= tier1->local_bw_max * 2) {
    /* tier2 BW isn't 2x tier1, we cannot guess HBM */
    hwloc_debug("    BW difference isn't >2x\n");
    return -1;
  }
  /* tier2 BW is >2x tier1 */

  if ((flags & HWLOC_GUESS_MEMTIERS_FLAG_NODE0_IS_DRAM)
      && hwloc_bitmap_isset(tier2->nodeset, 0)) {
    /* node0 is not DRAM, and we assume that's not possible */
    hwloc_debug("    node0 shouldn't have HBM BW\n");
    return -1;
  }

  /* assume tier1 == DRAM and tier2 == HBM */
  tier1->type = HWLOC_MEMORY_TIER_DRAM;
  tier2->type = HWLOC_MEMORY_TIER_HBM;
  hwloc_debug("    Success\n");
  return 0;
}

static int
hwloc__guess_memory_tiers_types(hwloc_topology_t topology __hwloc_attribute_unused,
                                unsigned nr_tiers,
                                struct hwloc_memory_tier_s *tiers)
{
  unsigned long flags;
  const char *env;
  unsigned nr_unknown, nr_spm;
  struct hwloc_memory_tier_s *unknown_tier[2], *spm_tier;
  unsigned i;

  flags = 0;
  env = getenv("HWLOC_MEMTIERS_GUESS");
  if (env) {
    if (!strcmp(env, "none"))
      return 0;
    /* by default, we don't guess anything unsure */
    if (!strcmp(env, "all"))
      /* enable all typical cases */
      flags = ~0UL;
    if (strstr(env, "spm_is_hbm")) {
      hwloc_debug("Assuming SPM-tier is HBM, ignore bandwidth\n");
      flags |= HWLOC_GUESS_MEMTIERS_FLAG_SPM_IS_HBM;
    }
    if (strstr(env, "node0_is_dram")) {
      hwloc_debug("Assuming node0 is DRAM\n");
      flags |= HWLOC_GUESS_MEMTIERS_FLAG_NODE0_IS_DRAM;
    }
  }

  if (nr_tiers == 1)
    /* Likely DRAM only, but could also be HBM-only in non-SPM mode.
     * We cannot be sure, but it doesn't matter since there's a single tier.
     */
    return 0;

  nr_unknown = nr_spm = 0;
  unknown_tier[0] = unknown_tier[1] = spm_tier = NULL;
  for(i=0; i<nr_tiers; i++) {
    switch (tiers[i].type) {
    case HWLOC_MEMORY_TIER_UNKNOWN:
      if (nr_unknown < 2)
        unknown_tier[nr_unknown] = &tiers[i];
      nr_unknown++;
      break;
    case HWLOC_MEMORY_TIER_SPM:
      spm_tier = &tiers[i];
      nr_spm++;
      break;
    case HWLOC_MEMORY_TIER_DRAM:
    case HWLOC_MEMORY_TIER_HBM:
      /* not possible */
      abort();
    default:
      /* ignore HBM, NVM, ... */
      break;
    }
  }
  hwloc_debug("Found %u unknown memory tiers and %u SPM\n",
              nr_unknown, nr_spm);

  /* Try to guess DRAM + HBM common cases.
   * Other things we'd like to detect:
   * single unknown => DRAM or HBM? HBM won't be SPM on HBM-only CPUs
   * unknown + CXL DRAM => DRAM or HBM?
   */
  if (nr_unknown == 2 && !nr_spm) {
    /* 2 unknown, could be DRAM + non-SPM HBM */
    hwloc_debug("  Trying to guess 2 unknown tiers using BW\n");
    hwloc__guess_dram_hbm_tiers(unknown_tier[0], unknown_tier[1], flags);
  } else if (nr_unknown == 1 && nr_spm == 1) {
    /* 1 unknown + 1 SPM, could be DRAM + SPM HBM */
    hwloc_debug("  Trying to guess 1 unknown + 1 SPM tiers using BW\n");
    hwloc__guess_dram_hbm_tiers(unknown_tier[0], spm_tier, flags);
  }

  if (flags & HWLOC_GUESS_MEMTIERS_FLAG_SPM_IS_HBM) {
    /* force mark SPM as HBM */
    for(i=0; i<nr_tiers; i++)
      if (tiers[i].type == HWLOC_MEMORY_TIER_SPM) {
        hwloc_debug("Forcing SPM tier to HBM");
        tiers[i].type = HWLOC_MEMORY_TIER_HBM;
      }
  }

  if (flags & HWLOC_GUESS_MEMTIERS_FLAG_NODE0_IS_DRAM) {
    /* force mark node0's tier as DRAM if we couldn't guess it */
    for(i=0; i<nr_tiers; i++)
      if (hwloc_bitmap_isset(tiers[i].nodeset, 0)
          && tiers[i].type == HWLOC_MEMORY_TIER_UNKNOWN) {
        hwloc_debug("Forcing node0 tier to DRAM");
        tiers[i].type = HWLOC_MEMORY_TIER_DRAM;
        break;
      }
  }

  return 0;
}

/* parses something like 0xf=HBM;0x0f=DRAM;0x00f=CXL-DRAM */
static struct hwloc_memory_tier_s *
hwloc__force_memory_tiers(hwloc_topology_t topology __hwloc_attribute_unused,
                          unsigned *nr_tiers_p,
                          const char *_env)
{
  struct hwloc_memory_tier_s *tiers = NULL;
  unsigned nr_tiers, i;
  hwloc_bitmap_t nodeset = NULL;
  char *env;
  const char *tmp;

  env = strdup(_env);
  if (!env) {
    fprintf(stderr, "[hwloc/memtiers] failed to duplicate HWLOC_MEMTIERS envvar\n");
    goto out;
  }

  tmp = env;
  nr_tiers = 1;
  while (1) {
    tmp = strchr(tmp, ';');
    if (!tmp)
      break;
    tmp++;
    nr_tiers++;
  }

  nodeset = hwloc_bitmap_alloc();
  if (!nodeset) {
    fprintf(stderr, "[hwloc/memtiers] failed to allocated forced tiers' nodeset\n");
    goto out_with_envvar;
  }

  tiers = calloc(nr_tiers, sizeof(*tiers));
  if (!tiers) {
    fprintf(stderr, "[hwloc/memtiers] failed to allocated forced tiers\n");
    goto out_with_nodeset;
  }
  nr_tiers = 0;

  tmp = env;
  while (1) {
    char *end;
    char *equal;
    hwloc_memory_tier_type_t type;

    end = strchr(tmp, ';');
    if (end)
      *end = '\0';

    equal = strchr(tmp, '=');
    if (!equal) {
      fprintf(stderr, "[hwloc/memtiers] missing `=' before end of forced tier description at `%s'\n", tmp);
      goto out_with_tiers;
    }
    *equal = '\0';

    hwloc_bitmap_sscanf(nodeset, tmp);
    if (hwloc_bitmap_iszero(nodeset)) {
      fprintf(stderr, "[hwloc/memtiers] empty forced tier nodeset `%s', aborting\n", tmp);
      goto out_with_tiers;
    }
    type = hwloc_memory_tier_type_sscanf(equal+1);
    if (!type)
      hwloc_debug("failed to recognize forced tier type `%s'\n", equal+1);
    tiers[nr_tiers].nodeset = hwloc_bitmap_dup(nodeset);
    tiers[nr_tiers].type = type;
    tiers[nr_tiers].local_bw_min = tiers[nr_tiers].local_bw_max = 0;
    tiers[nr_tiers].local_lat_min = tiers[nr_tiers].local_lat_max = 0;
    nr_tiers++;
    if (!end)
      break;
    tmp = end+1;
  }

  free(env);
  hwloc_bitmap_free(nodeset);
  hwloc_debug("Forcing %u memory tiers\n", nr_tiers);
#ifdef HWLOC_DEBUG
  for(i=0; i<nr_tiers; i++) {
    char *s;
    hwloc_bitmap_asprintf(&s, tiers[i].nodeset);
    hwloc_debug("  tier #%u type %lx nodeset %s\n", i, tiers[i].type, s);
    free(s);
  }
#endif
  *nr_tiers_p = nr_tiers;
  return tiers;

 out_with_tiers:
  for(i=0; i<nr_tiers; i++)
    hwloc_bitmap_free(tiers[i].nodeset);
  free(tiers);
 out_with_nodeset:
  hwloc_bitmap_free(nodeset);
 out_with_envvar:
  free(env);
 out:
  return NULL;
}

static void
hwloc__apply_memory_tiers_subtypes(hwloc_topology_t topology,
                                   unsigned nr_tiers,
                                   struct hwloc_memory_tier_s *tiers,
                                   int force)
{
  hwloc_obj_t node = NULL;
  hwloc_debug("Marking node tiers\n");
  while ((node = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_NUMANODE, node)) != NULL) {
    unsigned j;
    for(j=0; j<nr_tiers; j++) {
      if (hwloc_bitmap_isset(tiers[j].nodeset, node->os_index)) {
        const char *subtype = hwloc_memory_tier_type_snprintf(tiers[j].type);
        if (!node->subtype || force) { /* don't overwrite the existing subtype unless forced */
          if (subtype) { /* don't set a subtype for unknown tiers */
            hwloc_debug("  marking node L#%u P#%u as %s (was %s)\n", node->logical_index, node->os_index, subtype, node->subtype);
            free(node->subtype);
            node->subtype = strdup(subtype);
          }
        } else
          hwloc_debug("  node L#%u P#%u already marked as %s, not setting %s\n",
                      node->logical_index, node->os_index, node->subtype, subtype);
        if (nr_tiers > 1) {
          char tmp[20];
          snprintf(tmp, sizeof(tmp), "%u", j);
          hwloc__add_info_nodup(&node->infos, &node->infos_count, "MemoryTier", tmp, 1);
        }
        break; /* each node is in a single tier */
      }
    }
  }
}

int
hwloc_internal_memattrs_guess_memory_tiers(hwloc_topology_t topology, int force_subtype)
{
  struct hwloc_memory_tier_s *tiers;
  unsigned nr_tiers;
  unsigned i;
  const char *env;

  env = getenv("HWLOC_MEMTIERS");
  if (env) {
    if (!strcmp(env, "none"))
      goto out;
    tiers = hwloc__force_memory_tiers(topology, &nr_tiers, env);
    if (tiers) {
      assert(nr_tiers > 0);
      force_subtype = 1;
      goto ready;
    }
  }

  tiers = hwloc__group_memory_tiers(topology, &nr_tiers);
  if (!tiers)
    goto out;

  hwloc__guess_memory_tiers_types(topology, nr_tiers, tiers);

  /* sort tiers by BW first, then by type */
  hwloc_debug("Sorting memory tiers...\n");
  qsort(tiers, nr_tiers, sizeof(*tiers), compare_tiers_by_bw_and_type);

 ready:
#ifdef HWLOC_DEBUG
  for(i=0; i<nr_tiers; i++) {
    char *s;
    hwloc_bitmap_asprintf(&s, tiers[i].nodeset);
    hwloc_debug("  tier %u = nodes %s with type %lx and local BW %llu-%llu lat %llu-%llu\n",
                i,
                s, tiers[i].type,
                (unsigned long long) tiers[i].local_bw_min,
                (unsigned long long) tiers[i].local_bw_max,
                (unsigned long long) tiers[i].local_lat_min,
                (unsigned long long) tiers[i].local_lat_max);
    free(s);
  }
#endif

  hwloc__apply_memory_tiers_subtypes(topology, nr_tiers, tiers, force_subtype);

  for(i=0; i<nr_tiers; i++)
    hwloc_bitmap_free(tiers[i].nodeset);
  free(tiers);
 out:
  return 0;
}
