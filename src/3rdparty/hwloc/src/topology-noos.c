/*
 * Copyright © 2009 CNRS
 * Copyright © 2009-2019 Inria.  All rights reserved.
 * Copyright © 2009-2012 Université Bordeaux
 * Copyright © 2009-2011 Cisco Systems, Inc.  All rights reserved.
 * See COPYING in top-level directory.
 */

#include "private/autogen/config.h"
#include "hwloc.h"
#include "private/private.h"

static int
hwloc_look_noos(struct hwloc_backend *backend, struct hwloc_disc_status *dstatus)
{
  /*
   * This backend uses the underlying OS.
   * However we don't enforce topology->is_thissystem so that
   * we may still force use this backend when debugging with !thissystem.
   */

  struct hwloc_topology *topology = backend->topology;
  int nbprocs;

  assert(dstatus->phase == HWLOC_DISC_PHASE_CPU);

  if (topology->levels[0][0]->cpuset)
    /* somebody discovered things */
    return -1;

  nbprocs = hwloc_fallback_nbprocessors(0);
  if (nbprocs >= 1)
    topology->support.discovery->pu = 1;
  else
    nbprocs = 1;

  hwloc_alloc_root_sets(topology->levels[0][0]);
  hwloc_setup_pu_level(topology, nbprocs);
  hwloc_add_uname_info(topology, NULL);
  return 0;
}

static struct hwloc_backend *
hwloc_noos_component_instantiate(struct hwloc_topology *topology,
				 struct hwloc_disc_component *component,
				 unsigned excluded_phases __hwloc_attribute_unused,
				 const void *_data1 __hwloc_attribute_unused,
				 const void *_data2 __hwloc_attribute_unused,
				 const void *_data3 __hwloc_attribute_unused)
{
  struct hwloc_backend *backend;
  backend = hwloc_backend_alloc(topology, component);
  if (!backend)
    return NULL;
  backend->discover = hwloc_look_noos;
  return backend;
}

static struct hwloc_disc_component hwloc_noos_disc_component = {
  "no_os",
  HWLOC_DISC_PHASE_CPU,
  HWLOC_DISC_PHASE_GLOBAL,
  hwloc_noos_component_instantiate,
  40, /* lower than native OS component, higher than globals */
  1,
  NULL
};

const struct hwloc_component hwloc_noos_component = {
  HWLOC_COMPONENT_ABI,
  NULL, NULL,
  HWLOC_COMPONENT_TYPE_DISC,
  0,
  &hwloc_noos_disc_component
};
