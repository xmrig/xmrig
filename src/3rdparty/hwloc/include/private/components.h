/*
 * Copyright © 2012-2019 Inria.  All rights reserved.
 * See COPYING in top-level directory.
 */


#ifdef HWLOC_INSIDE_PLUGIN
/*
 * these declarations are internal only, they are not available to plugins
 * (many functions below are internal static symbols).
 */
#error This file should not be used in plugins
#endif


#ifndef PRIVATE_COMPONENTS_H
#define PRIVATE_COMPONENTS_H 1

#include "hwloc/plugins.h"

struct hwloc_topology;

extern int hwloc_disc_component_force_enable(struct hwloc_topology *topology,
					     int envvar_forced, /* 1 if forced through envvar, 0 if forced through API */
					     const char *name,
					     const void *data1, const void *data2, const void *data3);
extern void hwloc_disc_components_enable_others(struct hwloc_topology *topology);

/* Compute the topology is_thissystem flag and find some callbacks based on enabled backends */
extern void hwloc_backends_is_thissystem(struct hwloc_topology *topology);
extern void hwloc_backends_find_callbacks(struct hwloc_topology *topology);

/* Initialize the lists of components and backends used by a topology */
extern void hwloc_topology_components_init(struct hwloc_topology *topology);
/* Disable and destroy all backends used by a topology */
extern void hwloc_backends_disable_all(struct hwloc_topology *topology);
/* Cleanup the lists of components used by a topology */
extern void hwloc_topology_components_fini(struct hwloc_topology *topology);

/* Used by the core to setup/destroy the list of components */
extern void hwloc_components_init(void); /* increases components refcount, should be called exactly once per topology (during init) */
extern void hwloc_components_fini(void); /* decreases components refcount, should be called exactly once per topology (during destroy) */

#endif /* PRIVATE_COMPONENTS_H */

