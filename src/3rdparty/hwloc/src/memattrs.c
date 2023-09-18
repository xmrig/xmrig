/*
 * Copyright © 2020-2022 Inria.  All rights reserved.
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

struct hwloc_memory_tier_s {
  hwloc_obj_t node;
  uint64_t local_bw;
  enum hwloc_memory_tier_type_e {
    /* warning the order is important for guess_memory_tiers() after qsort() */
    HWLOC_MEMORY_TIER_UNKNOWN,
    HWLOC_MEMORY_TIER_DRAM,
    HWLOC_MEMORY_TIER_HBM,
    HWLOC_MEMORY_TIER_SPM, /* Specific-Purpose Memory is usually HBM, we'll use BW to confirm */
    HWLOC_MEMORY_TIER_NVM,
    HWLOC_MEMORY_TIER_GPU,
  } type;
};

static int compare_tiers(const void *_a, const void *_b)
{
  const struct hwloc_memory_tier_s *a = _a, *b = _b;
  /* sort by type of tier first */
  if (a->type != b->type)
    return a->type - b->type;
  /* then by bandwidth */
  if (a->local_bw > b->local_bw)
    return -1;
  else if (a->local_bw < b->local_bw)
    return 1;
  return 0;
}

int
hwloc_internal_memattrs_guess_memory_tiers(hwloc_topology_t topology)
{
  struct hwloc_internal_memattr_s *imattr;
  struct hwloc_memory_tier_s *tiers;
  unsigned i, j, n;
  const char *env;
  int spm_is_hbm = -1; /* -1 will guess from BW, 0 no, 1 forced */
  int mark_dram = 1;
  unsigned first_spm, first_nvm;
  hwloc_uint64_t max_unknown_bw, min_spm_bw;

  env = getenv("HWLOC_MEMTIERS_GUESS");
  if (env) {
    if (!strcmp(env, "none")) {
      return 0;
    } else if (!strcmp(env, "default")) {
      /* nothing */
    } else if (!strcmp(env, "spm_is_hbm")) {
      hwloc_debug("Assuming SPM-tier is HBM, ignore bandwidth\n");
      spm_is_hbm = 1;
    } else if (HWLOC_SHOW_CRITICAL_ERRORS()) {
      fprintf(stderr, "hwloc: Failed to recognize HWLOC_MEMTIERS_GUESS value %s\n", env);
    }
  }

  imattr = &topology->memattrs[HWLOC_MEMATTR_ID_BANDWIDTH];

  if (!(imattr->iflags & HWLOC_IMATTR_FLAG_CACHE_VALID))
    hwloc__imattr_refresh(topology, imattr);

  n = hwloc_get_nbobjs_by_depth(topology, HWLOC_TYPE_DEPTH_NUMANODE);
  assert(n);

  tiers = malloc(n * sizeof(*tiers));
  if (!tiers)
    return -1;

  for(i=0; i<n; i++) {
    hwloc_obj_t node;
    const char *daxtype;
    struct hwloc_internal_location_s iloc;
    struct hwloc_internal_memattr_target_s *imtg = NULL;
    struct hwloc_internal_memattr_initiator_s *imi;

    node = hwloc_get_obj_by_depth(topology, HWLOC_TYPE_DEPTH_NUMANODE, i);
    assert(node);
    tiers[i].node = node;

    /* defaults */
    tiers[i].type = HWLOC_MEMORY_TIER_UNKNOWN;
    tiers[i].local_bw = 0; /* unknown */

    daxtype = hwloc_obj_get_info_by_name(node, "DAXType");
    /* mark NVM, SPM and GPU nodes */
    if (daxtype && !strcmp(daxtype, "NVM"))
      tiers[i].type = HWLOC_MEMORY_TIER_NVM;
    if (daxtype && !strcmp(daxtype, "SPM"))
      tiers[i].type = HWLOC_MEMORY_TIER_SPM;
    if (node->subtype && !strcmp(node->subtype, "GPUMemory"))
      tiers[i].type = HWLOC_MEMORY_TIER_GPU;

    if (spm_is_hbm == -1) {
      for(j=0; j<imattr->nr_targets; j++)
        if (imattr->targets[j].obj == node) {
          imtg = &imattr->targets[j];
          break;
        }
      if (imtg && !hwloc_bitmap_iszero(node->cpuset)) {
        iloc.type = HWLOC_LOCATION_TYPE_CPUSET;
        iloc.location.cpuset = node->cpuset;
        imi = hwloc__memattr_target_get_initiator(imtg, &iloc, 0);
        if (imi)
          tiers[i].local_bw = imi->value;
      }
    }
  }

  /* sort tiers */
  qsort(tiers, n, sizeof(*tiers), compare_tiers);
  hwloc_debug("Sorting memory tiers...\n");
  for(i=0; i<n; i++)
    hwloc_debug("  tier %u = node L#%u P#%u with tier type %d and local BW #%llu\n",
                i,
                tiers[i].node->logical_index, tiers[i].node->os_index,
                tiers[i].type, (unsigned long long) tiers[i].local_bw);

  /* now we have UNKNOWN tiers (sorted by BW), then SPM tiers (sorted by BW), then NVM, then GPU */

  /* iterate over UNKNOWN tiers, and find their BW */
  for(i=0; i<n; i++) {
    if (tiers[i].type > HWLOC_MEMORY_TIER_UNKNOWN)
      break;
  }
  first_spm = i;
  /* get max BW from first */
  if (first_spm > 0)
    max_unknown_bw = tiers[0].local_bw;
  else
    max_unknown_bw = 0;

  /* there are no DRAM or HBM tiers yet */

  /* iterate over SPM tiers, and find their BW */
  for(i=first_spm; i<n; i++) {
    if (tiers[i].type > HWLOC_MEMORY_TIER_SPM)
      break;
  }
  first_nvm = i;
  /* get min BW from last */
  if (first_nvm > first_spm)
    min_spm_bw = tiers[first_nvm-1].local_bw;
  else
    min_spm_bw = 0;

  /* FIXME: if there's more than 10% between some sets of nodes inside a tier, split it? */
  /* FIXME: if there are cpuset-intersecting nodes in same tier, abort? */

  if (spm_is_hbm == -1) {
    /* if we have BW for all SPM and UNKNOWN
     * and all SPM BW are 2x superior to all UNKNOWN BW
     */
    hwloc_debug("UNKNOWN-memory-tier max bandwidth %llu\n", (unsigned long long) max_unknown_bw);
    hwloc_debug("SPM-memory-tier min bandwidth %llu\n", (unsigned long long) min_spm_bw);
    if (max_unknown_bw > 0 && min_spm_bw > 0 && max_unknown_bw*2 < min_spm_bw) {
      hwloc_debug("assuming SPM means HBM and !SPM means DRAM since bandwidths are very different\n");
      spm_is_hbm = 1;
    } else {
      hwloc_debug("cannot assume SPM means HBM\n");
      spm_is_hbm = 0;
    }
  }

  if (spm_is_hbm) {
    for(i=0; i<first_spm; i++)
      tiers[i].type = HWLOC_MEMORY_TIER_DRAM;
    for(i=first_spm; i<first_nvm; i++)
      tiers[i].type = HWLOC_MEMORY_TIER_HBM;
  }

  if (first_spm == n)
    mark_dram = 0;

    /* now apply subtypes */
  for(i=0; i<n; i++) {
    const char *type = NULL;
    if (tiers[i].node->subtype) /* don't overwrite the existing subtype */
      continue;
    switch (tiers[i].type) {
    case HWLOC_MEMORY_TIER_DRAM:
      if (mark_dram)
        type = "DRAM";
      break;
    case HWLOC_MEMORY_TIER_HBM:
      type = "HBM";
      break;
    case HWLOC_MEMORY_TIER_SPM:
      type = "SPM";
      break;
    case HWLOC_MEMORY_TIER_NVM:
      type = "NVM";
      break;
    default:
      /* GPU memory is already marked with subtype="GPUMemory",
       * UNKNOWN doesn't deserve any subtype
       */
      break;
    }
    if (type) {
      hwloc_debug("Marking node L#%u P#%u as %s\n", tiers[i].node->logical_index, tiers[i].node->os_index, type);
      tiers[i].node->subtype = strdup(type);
    }
  }

  free(tiers);
  return 0;
}
