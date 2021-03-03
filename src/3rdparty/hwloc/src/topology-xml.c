/*
 * Copyright © 2009 CNRS
 * Copyright © 2009-2020 Inria.  All rights reserved.
 * Copyright © 2009-2011, 2020 Université Bordeaux
 * Copyright © 2009-2018 Cisco Systems, Inc.  All rights reserved.
 * See COPYING in top-level directory.
 */

#include "private/autogen/config.h"
#include "hwloc.h"
#include "private/xml.h"
#include "private/private.h"
#include "private/misc.h"
#include "private/debug.h"

#include <math.h>

int
hwloc__xml_verbose(void)
{
  static int checked = 0;
  static int verbose = 0;
  if (!checked) {
    const char *env = getenv("HWLOC_XML_VERBOSE");
    if (env)
      verbose = atoi(env);
    checked = 1;
  }
  return verbose;
}

static int
hwloc_nolibxml_import(void)
{
  static int checked = 0;
  static int nolibxml = 0;
  if (!checked) {
    const char *env = getenv("HWLOC_LIBXML");
    if (env) {
      nolibxml = !atoi(env);
    } else {
      env = getenv("HWLOC_LIBXML_IMPORT");
      if (env)
	nolibxml = !atoi(env);
    }
    checked = 1;
  }
  return nolibxml;
}

static int
hwloc_nolibxml_export(void)
{
  static int checked = 0;
  static int nolibxml = 0;
  if (!checked) {
    const char *env = getenv("HWLOC_LIBXML");
    if (env) {
      nolibxml = !atoi(env);
    } else {
      env = getenv("HWLOC_LIBXML_EXPORT");
      if (env)
	nolibxml = !atoi(env);
    }
    checked = 1;
  }
  return nolibxml;
}

#define BASE64_ENCODED_LENGTH(length) (4*(((length)+2)/3))

/*********************************
 ********* XML callbacks *********
 *********************************/

/* set when registering nolibxml and libxml components.
 * modifications protected by the components mutex.
 * read by the common XML code in topology-xml.c to jump to the right XML backend.
 */
static struct hwloc_xml_callbacks *hwloc_nolibxml_callbacks = NULL, *hwloc_libxml_callbacks = NULL;

void
hwloc_xml_callbacks_register(struct hwloc_xml_component *comp)
{
  if (!hwloc_nolibxml_callbacks)
    hwloc_nolibxml_callbacks = comp->nolibxml_callbacks;
  if (!hwloc_libxml_callbacks)
    hwloc_libxml_callbacks = comp->libxml_callbacks;
}

void
hwloc_xml_callbacks_reset(void)
{
  hwloc_nolibxml_callbacks = NULL;
  hwloc_libxml_callbacks = NULL;
}

/************************************************
 ********* XML import (common routines) *********
 ************************************************/

#define _HWLOC_OBJ_CACHE_OLD (HWLOC_OBJ_TYPE_MAX+1) /* temporarily used when importing pre-v2.0 attribute-less cache types */
#define _HWLOC_OBJ_FUTURE    (HWLOC_OBJ_TYPE_MAX+2) /* temporarily used when ignoring future types */

static void
hwloc__xml_import_object_attr(struct hwloc_topology *topology,
			      struct hwloc_xml_backend_data_s *data,
			      struct hwloc_obj *obj,
			      const char *name, const char *value,
			      hwloc__xml_import_state_t state,
			      int *ignore)
{
  if (!strcmp(name, "type")) {
    /* already handled */
    return;
  }

  else if (!strcmp(name, "os_index"))
    obj->os_index = strtoul(value, NULL, 10);
  else if (!strcmp(name, "gp_index")) {
    obj->gp_index = strtoull(value, NULL, 10);
    if (!obj->gp_index && hwloc__xml_verbose())
      fprintf(stderr, "%s: unexpected zero gp_index, topology may be invalid\n", state->global->msgprefix);
    if (obj->gp_index >= topology->next_gp_index)
      topology->next_gp_index = obj->gp_index + 1;
  } else if (!strcmp(name, "cpuset")) {
    if (!obj->cpuset)
      obj->cpuset = hwloc_bitmap_alloc();
    hwloc_bitmap_sscanf(obj->cpuset, value);
  } else if (!strcmp(name, "complete_cpuset")) {
    if (!obj->complete_cpuset)
      obj->complete_cpuset = hwloc_bitmap_alloc();
    hwloc_bitmap_sscanf(obj->complete_cpuset, value);
  } else if (!strcmp(name, "allowed_cpuset")) {
    /* ignored except for root */
    if (!obj->parent)
      hwloc_bitmap_sscanf(topology->allowed_cpuset, value);
  } else if (!strcmp(name, "nodeset")) {
    if (!obj->nodeset)
      obj->nodeset = hwloc_bitmap_alloc();
    hwloc_bitmap_sscanf(obj->nodeset, value);
  } else if (!strcmp(name, "complete_nodeset")) {
    if (!obj->complete_nodeset)
      obj->complete_nodeset = hwloc_bitmap_alloc();
    hwloc_bitmap_sscanf(obj->complete_nodeset, value);
  } else if (!strcmp(name, "allowed_nodeset")) {
    /* ignored except for root */
    if (!obj->parent)
      hwloc_bitmap_sscanf(topology->allowed_nodeset, value);
  } else if (!strcmp(name, "name")) {
    if (obj->name)
      free(obj->name);
    obj->name = strdup(value);
  } else if (!strcmp(name, "subtype")) {
    if (obj->subtype)
      free(obj->subtype);
    obj->subtype = strdup(value);
  }

  else if (!strcmp(name, "cache_size")) {
    unsigned long long lvalue = strtoull(value, NULL, 10);
    if (hwloc__obj_type_is_cache(obj->type) || obj->type == _HWLOC_OBJ_CACHE_OLD || obj->type == HWLOC_OBJ_MEMCACHE)
      obj->attr->cache.size = lvalue;
    else if (hwloc__xml_verbose())
      fprintf(stderr, "%s: ignoring cache_size attribute for non-cache object type\n",
	      state->global->msgprefix);
  }

  else if (!strcmp(name, "cache_linesize")) {
    unsigned long lvalue = strtoul(value, NULL, 10);
    if (hwloc__obj_type_is_cache(obj->type) || obj->type == _HWLOC_OBJ_CACHE_OLD || obj->type == HWLOC_OBJ_MEMCACHE)
      obj->attr->cache.linesize = lvalue;
    else if (hwloc__xml_verbose())
      fprintf(stderr, "%s: ignoring cache_linesize attribute for non-cache object type\n",
	      state->global->msgprefix);
  }

  else if (!strcmp(name, "cache_associativity")) {
    int lvalue = atoi(value);
    if (hwloc__obj_type_is_cache(obj->type) || obj->type == _HWLOC_OBJ_CACHE_OLD || obj->type == HWLOC_OBJ_MEMCACHE)
      obj->attr->cache.associativity = lvalue;
    else if (hwloc__xml_verbose())
      fprintf(stderr, "%s: ignoring cache_associativity attribute for non-cache object type\n",
	      state->global->msgprefix);
  }

  else if (!strcmp(name, "cache_type")) {
    unsigned long lvalue = strtoul(value, NULL, 10);
    if (hwloc__obj_type_is_cache(obj->type) || obj->type == _HWLOC_OBJ_CACHE_OLD || obj->type == HWLOC_OBJ_MEMCACHE) {
      if (lvalue == HWLOC_OBJ_CACHE_UNIFIED
	  || lvalue == HWLOC_OBJ_CACHE_DATA
	  || lvalue == HWLOC_OBJ_CACHE_INSTRUCTION)
	obj->attr->cache.type = (hwloc_obj_cache_type_t) lvalue;
      else
	fprintf(stderr, "%s: ignoring invalid cache_type attribute %lu\n",
		state->global->msgprefix, lvalue);
    } else if (hwloc__xml_verbose())
      fprintf(stderr, "%s: ignoring cache_type attribute for non-cache object type\n",
	      state->global->msgprefix);
  }

  else if (!strcmp(name, "local_memory")) {
    unsigned long long lvalue = strtoull(value, NULL, 10);
    if (obj->type == HWLOC_OBJ_NUMANODE)
      obj->attr->numanode.local_memory = lvalue;
    else if (!obj->parent)
      topology->machine_memory.local_memory = lvalue;
    else if (hwloc__xml_verbose())
      fprintf(stderr, "%s: ignoring local_memory attribute for non-NUMAnode non-root object\n",
	      state->global->msgprefix);
  }

  else if (!strcmp(name, "depth")) {
    unsigned long lvalue = strtoul(value, NULL, 10);
     if (hwloc__obj_type_is_cache(obj->type) || obj->type == _HWLOC_OBJ_CACHE_OLD || obj->type == HWLOC_OBJ_MEMCACHE) {
	obj->attr->cache.depth = lvalue;
     } else if (obj->type == HWLOC_OBJ_GROUP || obj->type == HWLOC_OBJ_BRIDGE) {
       /* will be overwritten by the core */
     } else if (hwloc__xml_verbose())
       fprintf(stderr, "%s: ignoring depth attribute for object type without depth\n",
	       state->global->msgprefix);
  }

  else if (!strcmp(name, "kind")) {
    unsigned long lvalue = strtoul(value, NULL, 10);
    if (obj->type == HWLOC_OBJ_GROUP)
      obj->attr->group.kind = lvalue;
    else if (hwloc__xml_verbose())
      fprintf(stderr, "%s: ignoring kind attribute for non-group object type\n",
	      state->global->msgprefix);
  }

  else if (!strcmp(name, "subkind")) {
    unsigned long lvalue = strtoul(value, NULL, 10);
    if (obj->type == HWLOC_OBJ_GROUP)
      obj->attr->group.subkind = lvalue;
    else if (hwloc__xml_verbose())
      fprintf(stderr, "%s: ignoring subkind attribute for non-group object type\n",
	      state->global->msgprefix);
  }

  else if (!strcmp(name, "dont_merge")) {
    unsigned long lvalue = strtoul(value, NULL, 10);
    if (obj->type == HWLOC_OBJ_GROUP)
      obj->attr->group.dont_merge = lvalue;
    else if (hwloc__xml_verbose())
      fprintf(stderr, "%s: ignoring dont_merge attribute for non-group object type\n",
	      state->global->msgprefix);
  }

  else if (!strcmp(name, "pci_busid")) {
    switch (obj->type) {
    case HWLOC_OBJ_PCI_DEVICE:
    case HWLOC_OBJ_BRIDGE: {
      unsigned domain, bus, dev, func;
      if (sscanf(value, "%x:%02x:%02x.%01x",
		 &domain, &bus, &dev, &func) != 4) {
	if (hwloc__xml_verbose())
	  fprintf(stderr, "%s: ignoring invalid pci_busid format string %s\n",
		  state->global->msgprefix, value);
	*ignore = 1;
#ifndef HWLOC_HAVE_32BITS_PCI_DOMAIN
      } else if (domain > 0xffff) {
	static int warned = 0;
	if (!warned && !hwloc_hide_errors())
	  fprintf(stderr, "Ignoring PCI device with non-16bit domain.\nPass --enable-32bits-pci-domain to configure to support such devices\n(warning: it would break the library ABI, don't enable unless really needed).\n");
	warned = 1;
	*ignore = 1;
#endif
      } else {
	obj->attr->pcidev.domain = domain;
	obj->attr->pcidev.bus = bus;
	obj->attr->pcidev.dev = dev;
	obj->attr->pcidev.func = func;
      }
      break;
    }
    default:
      if (hwloc__xml_verbose())
	fprintf(stderr, "%s: ignoring pci_busid attribute for non-PCI object\n",
		state->global->msgprefix);
      break;
    }
  }

  else if (!strcmp(name, "pci_type")) {
    switch (obj->type) {
    case HWLOC_OBJ_PCI_DEVICE:
    case HWLOC_OBJ_BRIDGE: {
      unsigned classid, vendor, device, subvendor, subdevice, revision;
      if (sscanf(value, "%x [%04x:%04x] [%04x:%04x] %02x",
		 &classid, &vendor, &device, &subvendor, &subdevice, &revision) != 6) {
	if (hwloc__xml_verbose())
	  fprintf(stderr, "%s: ignoring invalid pci_type format string %s\n",
		  state->global->msgprefix, value);
      } else {
	obj->attr->pcidev.class_id = classid;
	obj->attr->pcidev.vendor_id = vendor;
	obj->attr->pcidev.device_id = device;
	obj->attr->pcidev.subvendor_id = subvendor;
	obj->attr->pcidev.subdevice_id = subdevice;
	obj->attr->pcidev.revision = revision;
      }
      break;
    }
    default:
      if (hwloc__xml_verbose())
	fprintf(stderr, "%s: ignoring pci_type attribute for non-PCI object\n",
		state->global->msgprefix);
      break;
    }
  }

  else if (!strcmp(name, "pci_link_speed")) {
    switch (obj->type) {
    case HWLOC_OBJ_PCI_DEVICE:
    case HWLOC_OBJ_BRIDGE: {
      obj->attr->pcidev.linkspeed = (float) atof(value);
      break;
    }
    default:
      if (hwloc__xml_verbose())
	fprintf(stderr, "%s: ignoring pci_link_speed attribute for non-PCI object\n",
		state->global->msgprefix);
      break;
    }
  }

  else if (!strcmp(name, "bridge_type")) {
    switch (obj->type) {
    case HWLOC_OBJ_BRIDGE: {
      unsigned upstream_type, downstream_type;
      if (sscanf(value, "%u-%u", &upstream_type, &downstream_type) != 2) {
	if (hwloc__xml_verbose())
	  fprintf(stderr, "%s: ignoring invalid bridge_type format string %s\n",
		  state->global->msgprefix, value);
      } else {
	obj->attr->bridge.upstream_type = (hwloc_obj_bridge_type_t) upstream_type;
	obj->attr->bridge.downstream_type = (hwloc_obj_bridge_type_t) downstream_type;
      };
      break;
    }
    default:
      if (hwloc__xml_verbose())
	fprintf(stderr, "%s: ignoring bridge_type attribute for non-bridge object\n",
		state->global->msgprefix);
      break;
    }
  }

  else if (!strcmp(name, "bridge_pci")) {
    switch (obj->type) {
    case HWLOC_OBJ_BRIDGE: {
      unsigned domain, secbus, subbus;
      if (sscanf(value, "%x:[%02x-%02x]",
		 &domain, &secbus, &subbus) != 3) {
	if (hwloc__xml_verbose())
	  fprintf(stderr, "%s: ignoring invalid bridge_pci format string %s\n",
		  state->global->msgprefix, value);
	*ignore = 1;
#ifndef HWLOC_HAVE_32BITS_PCI_DOMAIN
      } else if (domain > 0xffff) {
	static int warned = 0;
	if (!warned && !hwloc_hide_errors())
	  fprintf(stderr, "Ignoring bridge to PCI with non-16bit domain.\nPass --enable-32bits-pci-domain to configure to support such devices\n(warning: it would break the library ABI, don't enable unless really needed).\n");
	warned = 1;
	*ignore = 1;
#endif
      } else {
	obj->attr->bridge.downstream.pci.domain = domain;
	obj->attr->bridge.downstream.pci.secondary_bus = secbus;
	obj->attr->bridge.downstream.pci.subordinate_bus = subbus;
      }
      break;
    }
    default:
      if (hwloc__xml_verbose())
	fprintf(stderr, "%s: ignoring bridge_pci attribute for non-bridge object\n",
		state->global->msgprefix);
      break;
    }
  }

  else if (!strcmp(name, "osdev_type")) {
    switch (obj->type) {
    case HWLOC_OBJ_OS_DEVICE: {
      unsigned osdev_type;
      if (sscanf(value, "%u", &osdev_type) != 1) {
	if (hwloc__xml_verbose())
	  fprintf(stderr, "%s: ignoring invalid osdev_type format string %s\n",
		  state->global->msgprefix, value);
      } else
	obj->attr->osdev.type = (hwloc_obj_osdev_type_t) osdev_type;
      break;
    }
    default:
      if (hwloc__xml_verbose())
	fprintf(stderr, "%s: ignoring osdev_type attribute for non-osdev object\n",
		state->global->msgprefix);
      break;
    }
  }

  else if (data->version_major < 2) {
    /************************
     * deprecated from 1.x
     */
    if (!strcmp(name, "os_level")
	|| !strcmp(name, "online_cpuset"))
      { /* ignored */ }

    /*************************
     * deprecated from 1.0
     */
    else if (!strcmp(name, "dmi_board_vendor")) {
      if (value[0])
	hwloc_obj_add_info(obj, "DMIBoardVendor", value);
    }
    else if (!strcmp(name, "dmi_board_name")) {
      if (value[0])
	hwloc_obj_add_info(obj, "DMIBoardName", value);
    }

    else if (data->version_major < 1) {
      /*************************
       * deprecated from 0.9
       */
      if (!strcmp(name, "memory_kB")) {
	unsigned long long lvalue = strtoull(value, NULL, 10);
	if (obj->type == _HWLOC_OBJ_CACHE_OLD)
	  obj->attr->cache.size = lvalue << 10;
	else if (obj->type == HWLOC_OBJ_NUMANODE)
	  obj->attr->numanode.local_memory = lvalue << 10;
	else if (!obj->parent)
	  topology->machine_memory.local_memory = lvalue << 10;
	else if (hwloc__xml_verbose())
	  fprintf(stderr, "%s: ignoring memory_kB attribute for non-NUMAnode non-root object\n",
		  state->global->msgprefix);
      }
      else if (!strcmp(name, "huge_page_size_kB")) {
	unsigned long lvalue = strtoul(value, NULL, 10);
	if (obj->type == HWLOC_OBJ_NUMANODE || !obj->parent) {
	  struct hwloc_numanode_attr_s *memory = obj->type == HWLOC_OBJ_NUMANODE ? &obj->attr->numanode : &topology->machine_memory;
	  if (!memory->page_types) {
	    memory->page_types = malloc(sizeof(*memory->page_types));
	    memory->page_types_len = 1;
	  }
	  assert(memory->page_types);
	  memory->page_types[0].size = lvalue << 10;
	} else if (hwloc__xml_verbose()) {
	  fprintf(stderr, "%s: ignoring huge_page_size_kB attribute for non-NUMAnode non-root object\n",
		  state->global->msgprefix);
	}
      }
      else if (!strcmp(name, "huge_page_free")) {
	unsigned long lvalue = strtoul(value, NULL, 10);
	if (obj->type == HWLOC_OBJ_NUMANODE || !obj->parent) {
	  struct hwloc_numanode_attr_s *memory = obj->type == HWLOC_OBJ_NUMANODE ? &obj->attr->numanode : &topology->machine_memory;
	  if (!memory->page_types) {
	    memory->page_types = malloc(sizeof(*memory->page_types));
	    memory->page_types_len = 1;
	  }
	  assert(memory->page_types);
	  memory->page_types[0].count = lvalue;
	} else if (hwloc__xml_verbose()) {
	  fprintf(stderr, "%s: ignoring huge_page_free attribute for non-NUMAnode non-root object\n",
		  state->global->msgprefix);
	}
      }
      /* end of deprecated from 0.9 */
      else goto unknown;
    }
    /* end of deprecated from 1.0 */
    else goto unknown;
  }
  else {
  unknown:
    if (hwloc__xml_verbose())
      fprintf(stderr, "%s: ignoring unknown object attribute %s\n",
	      state->global->msgprefix, name);
  }
}

static int
hwloc___xml_import_info(char **infonamep, char **infovaluep,
                        hwloc__xml_import_state_t state)
{
  char *infoname = NULL;
  char *infovalue = NULL;

  while (1) {
    char *attrname, *attrvalue;
    if (state->global->next_attr(state, &attrname, &attrvalue) < 0)
      break;
    if (!strcmp(attrname, "name"))
      infoname = attrvalue;
    else if (!strcmp(attrname, "value"))
      infovalue = attrvalue;
    else
      return -1;
  }

  *infonamep = infoname;
  *infovaluep = infovalue;

  return state->global->close_tag(state);
}

static int
hwloc__xml_import_obj_info(struct hwloc_xml_backend_data_s *data,
                           hwloc_obj_t obj,
                           hwloc__xml_import_state_t state)
{
  char *infoname = NULL;
  char *infovalue = NULL;
  int err;

  err = hwloc___xml_import_info(&infoname, &infovalue, state);
  if (err < 0)
    return err;

  if (infoname) {
    /* empty strings are ignored by libxml */
    if (data->version_major < 2 &&
	(!strcmp(infoname, "Type") || !strcmp(infoname, "CoProcType"))) {
      /* 1.x stored subtype in Type or CoProcType */
      if (infovalue) {
	if (obj->subtype)
	  free(obj->subtype);
	obj->subtype = strdup(infovalue);
      }
    } else {
      if (infovalue)
	hwloc_obj_add_info(obj, infoname, infovalue);
    }
  }

  return err;
}

static int
hwloc__xml_import_pagetype(hwloc_topology_t topology __hwloc_attribute_unused, struct hwloc_numanode_attr_s *memory,
			   hwloc__xml_import_state_t state)
{
  uint64_t size = 0, count = 0;

  while (1) {
    char *attrname, *attrvalue;
    if (state->global->next_attr(state, &attrname, &attrvalue) < 0)
      break;
    if (!strcmp(attrname, "size"))
      size = strtoull(attrvalue, NULL, 10);
    else if (!strcmp(attrname, "count"))
      count = strtoull(attrvalue, NULL, 10);
    else
      return -1;
  }

  if (size) {
    unsigned idx = memory->page_types_len;
    struct hwloc_memory_page_type_s *tmp;
    tmp = realloc(memory->page_types, (idx+1)*sizeof(*memory->page_types));
    if (tmp) { /* if failed to allocate, ignore this page_type entry */
      memory->page_types = tmp;
      memory->page_types_len = idx+1;
      memory->page_types[idx].size = size;
      memory->page_types[idx].count = count;
    }
  }

  return state->global->close_tag(state);
}

static int
hwloc__xml_v1import_distances(struct hwloc_xml_backend_data_s *data,
			      hwloc_obj_t obj,
			      hwloc__xml_import_state_t state)
{
  unsigned long reldepth = 0, nbobjs = 0;
  float latbase = 0;
  char *tag;
  int ret;

  while (1) {
    char *attrname, *attrvalue;
    if (state->global->next_attr(state, &attrname, &attrvalue) < 0)
      break;
    if (!strcmp(attrname, "nbobjs"))
      nbobjs = strtoul(attrvalue, NULL, 10);
    else if (!strcmp(attrname, "relative_depth"))
      reldepth = strtoul(attrvalue, NULL, 10);
    else if (!strcmp(attrname, "latency_base"))
      latbase = (float) atof(attrvalue);
    else
      return -1;
  }

  if (nbobjs && reldepth && latbase) {
    unsigned i;
    float *matrix;
    struct hwloc__xml_imported_v1distances_s *v1dist;

    matrix = malloc(nbobjs*nbobjs*sizeof(float));
    v1dist = malloc(sizeof(*v1dist));
    if (!matrix || !v1dist) {
      if (hwloc__xml_verbose())
	fprintf(stderr, "%s: failed to allocate v1distance matrix for %lu objects\n",
		state->global->msgprefix, nbobjs);
      free(v1dist);
      free(matrix);
      return -1;
    }

    v1dist->kind = HWLOC_DISTANCES_KIND_FROM_OS|HWLOC_DISTANCES_KIND_MEANS_LATENCY;
    /* TODO: we can't know for sure if it comes from the OS.
     * On Linux/x86, it would be 10 on the diagonal.
     * On Solaris/T5, 15 on the diagonal.
     * Just check whether all values are integers, and that all values on the diagonal are minimal and identical?
     */

    v1dist->nbobjs = nbobjs;
    v1dist->floats = matrix;

    for(i=0; i<nbobjs*nbobjs; i++) {
      struct hwloc__xml_import_state_s childstate;
      char *attrname, *attrvalue;
      float val;

      ret = state->global->find_child(state, &childstate, &tag);
      if (ret <= 0 || strcmp(tag, "latency")) {
	/* a latency child is needed */
	free(matrix);
	free(v1dist);
	return -1;
      }

      ret = state->global->next_attr(&childstate, &attrname, &attrvalue);
      if (ret < 0 || strcmp(attrname, "value")) {
	free(matrix);
	free(v1dist);
	return -1;
      }

      val = (float) atof((char *) attrvalue);
      matrix[i] = val * latbase;

      ret = state->global->close_tag(&childstate);
      if (ret < 0) {
	free(matrix);
	free(v1dist);
	return -1;
      }

      state->global->close_child(&childstate);
    }

    if (nbobjs < 2) {
      /* distances with a single object are useless, even if the XML isn't invalid */
      assert(nbobjs == 1);
      if (hwloc__xml_verbose())
	fprintf(stderr, "%s: ignoring invalid distance matrix with only 1 object\n",
		state->global->msgprefix);
      free(matrix);
      free(v1dist);

    } else if (obj->parent) {
      /* we currently only import distances attached to root.
       * we can't save obj in v1dist because obj could be dropped during insert if ignored.
       * we could save its complete_cpu/nodeset instead to find it back later.
       * but it doesn't matter much since only NUMA distances attached to root matter.
       */
      free(matrix);
      free(v1dist);

    } else {
      /* queue the distance for real */
      v1dist->prev = data->last_v1dist;
      v1dist->next = NULL;
      if (data->last_v1dist)
	data->last_v1dist->next = v1dist;
      else
	data->first_v1dist = v1dist;
      data->last_v1dist = v1dist;
    }
  }

  return state->global->close_tag(state);
}

static int
hwloc__xml_import_userdata(hwloc_topology_t topology __hwloc_attribute_unused, hwloc_obj_t obj,
			   hwloc__xml_import_state_t state)
{
  size_t length = 0;
  int encoded = 0;
  char *name = NULL; /* optional */
  int ret;

  while (1) {
    char *attrname, *attrvalue;
    if (state->global->next_attr(state, &attrname, &attrvalue) < 0)
      break;
    if (!strcmp(attrname, "length"))
      length = strtoul(attrvalue, NULL, 10);
    else if (!strcmp(attrname, "encoding"))
      encoded = !strcmp(attrvalue, "base64");
    else if (!strcmp(attrname, "name"))
      name = attrvalue;
    else
      return -1;
  }

  if (!topology->userdata_import_cb) {
    const char *buffer;
    size_t reallength = encoded ? BASE64_ENCODED_LENGTH(length) : length;
    ret = state->global->get_content(state, &buffer, reallength);
    if (ret < 0)
      return -1;

  } else if (topology->userdata_not_decoded) {
      const char *buffer;
      char *fakename;
      size_t reallength = encoded ? BASE64_ENCODED_LENGTH(length) : length;
      ret = state->global->get_content(state, &buffer, reallength);
      if (ret < 0)
        return -1;
      fakename = malloc(6 + 1 + (name ? strlen(name) : 4) + 1);
      if (!fakename)
	return -1;
      sprintf(fakename, encoded ? "base64%c%s" : "normal%c%s", name ? ':' : '-', name ? name : "anon");
      topology->userdata_import_cb(topology, obj, fakename, buffer, length);
      free(fakename);

  } else if (encoded && length) {
      const char *encoded_buffer;
      size_t encoded_length = BASE64_ENCODED_LENGTH(length);
      ret = state->global->get_content(state, &encoded_buffer, encoded_length);
      if (ret < 0)
        return -1;
      if (ret) {
	char *decoded_buffer = malloc(length+1);
	if (!decoded_buffer)
	  return -1;
	assert(encoded_buffer[encoded_length] == 0);
	ret = hwloc_decode_from_base64(encoded_buffer, decoded_buffer, length+1);
	if (ret != (int) length) {
	  free(decoded_buffer);
	  return -1;
	}
	topology->userdata_import_cb(topology, obj, name, decoded_buffer, length);
	free(decoded_buffer);
      }

  } else { /* always handle length==0 in the non-encoded case */
      const char *buffer = "";
      if (length) {
	ret = state->global->get_content(state, &buffer, length);
	if (ret < 0)
	  return -1;
      }
      topology->userdata_import_cb(topology, obj, name, buffer, length);
  }

  state->global->close_content(state);
  return state->global->close_tag(state);
}

static void hwloc__xml_import_report_outoforder(hwloc_topology_t topology, hwloc_obj_t new, hwloc_obj_t old)
{
  char *progname = hwloc_progname(topology);
  const char *origversion = hwloc_obj_get_info_by_name(topology->levels[0][0], "hwlocVersion");
  const char *origprogname = hwloc_obj_get_info_by_name(topology->levels[0][0], "ProcessName");
  char *c1, *cc1, t1[64];
  char *c2 = NULL, *cc2 = NULL, t2[64];

  hwloc_bitmap_asprintf(&c1, new->cpuset);
  hwloc_bitmap_asprintf(&cc1, new->complete_cpuset);
  hwloc_obj_type_snprintf(t1, sizeof(t1), new, 0);

  if (old->cpuset)
    hwloc_bitmap_asprintf(&c2, old->cpuset);
  if (old->complete_cpuset)
    hwloc_bitmap_asprintf(&cc2, old->complete_cpuset);
  hwloc_obj_type_snprintf(t2, sizeof(t2), old, 0);

  fprintf(stderr, "****************************************************************************\n");
  fprintf(stderr, "* hwloc has encountered an out-of-order XML topology load.\n");
  fprintf(stderr, "* Object %s cpuset %s complete %s\n",
	  t1, c1, cc1);
  fprintf(stderr, "* was inserted after object %s with %s and %s.\n",
	  t2, c2 ? c2 : "none", cc2 ? cc2 : "none");
  fprintf(stderr, "* The error occured in hwloc %s inside process `%s', while\n",
	  HWLOC_VERSION,
	  progname ? progname : "<unknown>");
  if (origversion || origprogname)
    fprintf(stderr, "* the input XML was generated by hwloc %s inside process `%s'.\n",
	    origversion ? origversion : "(unknown version)",
	    origprogname ? origprogname : "<unknown>");
  else
    fprintf(stderr, "* the input XML was generated by an unspecified ancient hwloc release.\n");
  fprintf(stderr, "* Please check that your input topology XML file is valid.\n");
  fprintf(stderr, "* Set HWLOC_DEBUG_CHECK=1 in the environment to detect further issues.\n");
  fprintf(stderr, "****************************************************************************\n");

  free(c1);
  free(cc1);
  free(c2);
  free(cc2);
  free(progname);
}

static int
hwloc__xml_import_object(hwloc_topology_t topology,
			 struct hwloc_xml_backend_data_s *data,
			 hwloc_obj_t parent, hwloc_obj_t obj, int *gotignored,
			 hwloc__xml_import_state_t state)
{
  int ignored = 0;
  int childrengotignored = 0;
  int attribute_less_cache = 0;
  int numa_was_root = 0;
  char *tag;
  struct hwloc__xml_import_state_s childstate;

  /* set parent now since it's used during import below or in subfunctions */
  obj->parent = parent;

  /* process attributes */
  while (1) {
    char *attrname, *attrvalue;
    if (state->global->next_attr(state, &attrname, &attrvalue) < 0)
      break;
    if (!strcmp(attrname, "type")) {
      if (hwloc_type_sscanf(attrvalue, &obj->type, NULL, 0) < 0) {
	if (!strcasecmp(attrvalue, "Cache")) {
	  obj->type = _HWLOC_OBJ_CACHE_OLD; /* will be fixed below */
	  attribute_less_cache = 1;
	} else if (!strcasecmp(attrvalue, "System")) {
	  if (!parent)
	    obj->type = HWLOC_OBJ_MACHINE;
	  else {
	    if (hwloc__xml_verbose())
	      fprintf(stderr, "%s: obsolete System object only allowed at root\n",
		      state->global->msgprefix);
	    goto error_with_object;
	  }
	} else if (!strcasecmp(attrvalue, "Tile")) {
	  /* deal with possible future type */
	  obj->type = HWLOC_OBJ_GROUP;
	  obj->attr->group.kind = HWLOC_GROUP_KIND_INTEL_TILE;
	} else if (!strcasecmp(attrvalue, "Module")) {
	  /* deal with possible future type */
	  obj->type = HWLOC_OBJ_GROUP;
	  obj->attr->group.kind = HWLOC_GROUP_KIND_INTEL_MODULE;
	} else if (!strcasecmp(attrvalue, "MemCache")) {
	  /* ignore possible future type */
	  obj->type = _HWLOC_OBJ_FUTURE;
	  ignored = 1;
	  if (hwloc__xml_verbose())
	    fprintf(stderr, "%s: %s object not-supported, will be ignored\n",
		    state->global->msgprefix, attrvalue);
	} else {
	  if (hwloc__xml_verbose())
	    fprintf(stderr, "%s: unrecognized object type string %s\n",
		    state->global->msgprefix, attrvalue);
	  goto error_with_object;
	}
      }
    } else {
      /* type needed first */
      if (obj->type == HWLOC_OBJ_TYPE_NONE) {
	if (hwloc__xml_verbose())
	  fprintf(stderr, "%s: object attribute %s found before type\n",
		  state->global->msgprefix,  attrname);
	goto error_with_object;
      }
      hwloc__xml_import_object_attr(topology, data, obj, attrname, attrvalue, state, &ignored);
    }
  }

  /* process non-object subnodes to get info attrs (as well as page_types, etc) */
  while (1) {
    int ret;

    tag = NULL;
    ret = state->global->find_child(state, &childstate, &tag);
    if (ret < 0)
      goto error;
    if (!ret)
      break;

    if (!strcmp(tag, "object")) {
      /* we'll handle children later */
      break;

    } else if (!strcmp(tag, "page_type")) {
      if (obj->type == HWLOC_OBJ_NUMANODE) {
	ret = hwloc__xml_import_pagetype(topology, &obj->attr->numanode, &childstate);
      } else if (!parent) {
	ret = hwloc__xml_import_pagetype(topology, &topology->machine_memory, &childstate);
      } else {
	if (hwloc__xml_verbose())
	  fprintf(stderr, "%s: invalid non-NUMAnode object child %s\n",
		  state->global->msgprefix, tag);
	ret = -1;
      }

    } else if (!strcmp(tag, "info")) {
      ret = hwloc__xml_import_obj_info(data, obj, &childstate);
    } else if (data->version_major < 2 && !strcmp(tag, "distances")) {
      ret = hwloc__xml_v1import_distances(data, obj, &childstate);
    } else if (!strcmp(tag, "userdata")) {
      ret = hwloc__xml_import_userdata(topology, obj, &childstate);
    } else {
      if (hwloc__xml_verbose())
	fprintf(stderr, "%s: invalid special object child %s\n",
		state->global->msgprefix, tag);
      ret = -1;
    }

    if (ret < 0)
      goto error;

    state->global->close_child(&childstate);
  }

  if (parent && obj->type == HWLOC_OBJ_MACHINE) {
    /* replace non-root Machine with Groups */
    obj->type = HWLOC_OBJ_GROUP;
  }

  if (parent && data->version_major >= 2) {
    /* check parent/child types for 2.x */
    if (hwloc__obj_type_is_normal(obj->type)) {
      if (!hwloc__obj_type_is_normal(parent->type)) {
	if (hwloc__xml_verbose())
	  fprintf(stderr, "normal object %s cannot be child of non-normal parent %s\n",
		  hwloc_obj_type_string(obj->type), hwloc_obj_type_string(parent->type));
	goto error_with_object;
      }
    } else if (hwloc__obj_type_is_memory(obj->type)) {
      if (hwloc__obj_type_is_io(parent->type) || HWLOC_OBJ_MISC == parent->type) {
	if (hwloc__xml_verbose())
	  fprintf(stderr, "Memory object %s cannot be child of non-normal-or-memory parent %s\n",
		  hwloc_obj_type_string(obj->type), hwloc_obj_type_string(parent->type));
	goto error_with_object;
      }
    } else if (hwloc__obj_type_is_io(obj->type)) {
      if (hwloc__obj_type_is_memory(parent->type) || HWLOC_OBJ_MISC == parent->type) {
	if (hwloc__xml_verbose())
	  fprintf(stderr, "I/O object %s cannot be child of non-normal-or-I/O parent %s\n",
		  hwloc_obj_type_string(obj->type), hwloc_obj_type_string(parent->type));
	goto error_with_object;
      }
    }

  } else if (parent && data->version_major < 2) {
    /* check parent/child types for pre-v2.0 */
    if (hwloc__obj_type_is_normal(obj->type) || HWLOC_OBJ_NUMANODE == obj->type) {
      if (hwloc__obj_type_is_special(parent->type)) {
	if (hwloc__xml_verbose())
	  fprintf(stderr, "v1.x normal v1.x object %s cannot be child of special parent %s\n",
		  hwloc_obj_type_string(obj->type), hwloc_obj_type_string(parent->type));
	goto error_with_object;
      }
    } else if (hwloc__obj_type_is_io(obj->type)) {
      if (HWLOC_OBJ_MISC == parent->type) {
	if (hwloc__xml_verbose())
	  fprintf(stderr, "I/O object %s cannot be child of Misc parent\n",
		  hwloc_obj_type_string(obj->type));
	goto error_with_object;
      }
    }
  }

  if (data->version_major < 2) {
    /***************************
     * 1.x specific checks
     */

    /* attach pre-v2.0 children of NUMA nodes to normal parent */
    if (parent && parent->type == HWLOC_OBJ_NUMANODE) {
      parent = parent->parent;
      assert(parent);
    }

    /* insert a group above pre-v2.0 NUMA nodes if needed */
    if (obj->type == HWLOC_OBJ_NUMANODE) {
      if (!parent) {
	/* crazy case of NUMA node root (only possible when filtering Machine keep_structure in v1.x),
	 * reinsert a Machine object
	 */
	hwloc_obj_t machine = hwloc_alloc_setup_object(topology, HWLOC_OBJ_MACHINE, HWLOC_UNKNOWN_INDEX);
	machine->cpuset = hwloc_bitmap_dup(obj->cpuset);
	machine->complete_cpuset = hwloc_bitmap_dup(obj->cpuset);
	machine->nodeset = hwloc_bitmap_dup(obj->nodeset);
	machine->complete_nodeset = hwloc_bitmap_dup(obj->complete_nodeset);
	topology->levels[0][0] = machine;
	parent = machine;
	numa_was_root = 1;

      } else if (!hwloc_bitmap_isequal(obj->complete_cpuset, parent->complete_cpuset)) {
	/* This NUMA node has a different locality from its parent.
	 * Don't attach it to this parent, or it well get its parent cpusets.
	 * Add an intermediate Group with the desired locality.
	 */
	int needgroup = 1;
	hwloc_obj_t sibling;

	sibling = parent->memory_first_child;
	if (sibling && !sibling->subtype
	    && !sibling->next_sibling
	    && obj->subtype && !strcmp(obj->subtype, "MCDRAM")
	    && hwloc_bitmap_iszero(obj->complete_cpuset)) {
	  /* this is KNL MCDRAM, we want to attach it near its DDR sibling */
	  needgroup = 0;
	}
	/* Ideally we would also detect similar cases on future non-KNL platforms with multiple local NUMA nodes.
	 * That's unlikely to occur with v1.x.
	 * And we have no way to be sure if this CPU-less node is desired or not.
	 */

	if (needgroup
	    && hwloc_filter_check_keep_object_type(topology, HWLOC_OBJ_GROUP)) {
	  hwloc_obj_t group = hwloc_alloc_setup_object(topology, HWLOC_OBJ_GROUP, HWLOC_UNKNOWN_INDEX);
	  group->gp_index = 0; /* will be initialized at the end of the discovery once we know the max */
	  group->cpuset = hwloc_bitmap_dup(obj->cpuset);
	  group->complete_cpuset = hwloc_bitmap_dup(obj->cpuset);
	  group->nodeset = hwloc_bitmap_dup(obj->nodeset);
	  group->complete_nodeset = hwloc_bitmap_dup(obj->complete_nodeset);
	  group->attr->group.kind = HWLOC_GROUP_KIND_MEMORY;
	  hwloc_insert_object_by_parent(topology, parent, group);
	  parent = group;
	}
      }
    }

    /* fixup attribute-less caches imported from pre-v2.0 XMLs */
    if (attribute_less_cache) {
      assert(obj->type == _HWLOC_OBJ_CACHE_OLD);
      obj->type = hwloc_cache_type_by_depth_type(obj->attr->cache.depth, obj->attr->cache.type);
    }

    /* fixup Misc objects inserted by cpusets in pre-v2.0 XMLs */
    if (obj->type == HWLOC_OBJ_MISC && obj->cpuset)
      obj->type = HWLOC_OBJ_GROUP;

    /* check set consistency.
     * 1.7.2 and earlier reported I/O Groups with only a cpuset, we don't want to reject those XMLs yet.
     * Ignore those Groups since fixing the missing sets is hard (would need to look at children sets which are not available yet).
     * Just abort the XML for non-Groups.
     */
    if (!obj->cpuset != !obj->complete_cpuset) {
      /* has some cpuset without others */
      if (obj->type == HWLOC_OBJ_GROUP) {
	ignored = 1;
      } else {
	if (hwloc__xml_verbose())
	  fprintf(stderr, "%s: invalid object %s P#%u with some missing cpusets\n",
		  state->global->msgprefix, hwloc_obj_type_string(obj->type), obj->os_index);
	goto error_with_object;
      }
    } else if (!obj->nodeset != !obj->complete_nodeset) {
      /* has some nodeset without others */
      if (obj->type == HWLOC_OBJ_GROUP) {
	ignored = 1;
      } else {
	if (hwloc__xml_verbose())
	  fprintf(stderr, "%s: invalid object %s P#%u with some missing nodesets\n",
		  state->global->msgprefix, hwloc_obj_type_string(obj->type), obj->os_index);
	goto error_with_object;
      }
    } else if (obj->nodeset && !obj->cpuset) {
      /* has nodesets without cpusets (the contrary is allowed in pre-2.0) */
      if (obj->type == HWLOC_OBJ_GROUP) {
	ignored = 1;
      } else {
	if (hwloc__xml_verbose())
	  fprintf(stderr, "%s: invalid object %s P#%u with either cpuset or nodeset missing\n",
		  state->global->msgprefix, hwloc_obj_type_string(obj->type), obj->os_index);
	goto error_with_object;
      }
    }
    /* end of 1.x specific checks */
  }

  /* 2.0 backward compatibility */
  if (obj->type == HWLOC_OBJ_GROUP) {
    if (obj->attr->group.kind == HWLOC_GROUP_KIND_INTEL_DIE
	|| (obj->subtype && !strcmp(obj->subtype, "Die")))
      obj->type = HWLOC_OBJ_DIE;
  }

  /* check that cache attributes are coherent with the actual type */
  if (hwloc__obj_type_is_cache(obj->type)
      && obj->type != hwloc_cache_type_by_depth_type(obj->attr->cache.depth, obj->attr->cache.type)) {
    if (hwloc__xml_verbose())
      fprintf(stderr, "%s: invalid cache type %s with attribute depth %u and type %d\n",
	      state->global->msgprefix, hwloc_obj_type_string(obj->type), obj->attr->cache.depth, (int) obj->attr->cache.type);
    goto error_with_object;
  }

  /* check special types vs cpuset */
  if (!obj->cpuset && !hwloc__obj_type_is_special(obj->type)) {
    if (hwloc__xml_verbose())
      fprintf(stderr, "%s: invalid normal object %s P#%u without cpuset\n",
	      state->global->msgprefix, hwloc_obj_type_string(obj->type), obj->os_index);
    goto error_with_object;
  }
  if (obj->cpuset && hwloc__obj_type_is_special(obj->type)) {
    if (hwloc__xml_verbose())
      fprintf(stderr, "%s: invalid special object %s with cpuset\n",
	      state->global->msgprefix, hwloc_obj_type_string(obj->type));
    goto error_with_object;
  }

  /* check parent vs child sets */
  if (obj->cpuset && parent && !parent->cpuset) {
    if (hwloc__xml_verbose())
      fprintf(stderr, "%s: invalid object %s P#%u with cpuset while parent has none\n",
	      state->global->msgprefix, hwloc_obj_type_string(obj->type), obj->os_index);
    goto error_with_object;
  }
  if (obj->nodeset && parent && !parent->nodeset) {
    if (hwloc__xml_verbose())
      fprintf(stderr, "%s: invalid object %s P#%u with nodeset while parent has none\n",
	      state->global->msgprefix, hwloc_obj_type_string(obj->type), obj->os_index);
    goto error_with_object;
  }

  /* check NUMA nodes */
  if (obj->type == HWLOC_OBJ_NUMANODE) {
    if (!obj->nodeset) {
      if (hwloc__xml_verbose())
	fprintf(stderr, "%s: invalid NUMA node object P#%u without nodeset\n",
		state->global->msgprefix, obj->os_index);
      goto error_with_object;
    }
    data->nbnumanodes++;
    obj->prev_cousin = data->last_numanode;
    obj->next_cousin = NULL;
    if (data->last_numanode)
      data->last_numanode->next_cousin = obj;
    else
      data->first_numanode = obj;
    data->last_numanode = obj;
  }

  if (!hwloc_filter_check_keep_object(topology, obj)) {
    /* Ignore this object instead of inserting it.
     *
     * Well, let the core ignore the root object later
     * because we don't know yet if root has more than one child.
     */
    if (parent)
      ignored = 1;
  }

  if (parent && !ignored) {
    /* root->parent is NULL, and root is already inserted */
    hwloc_insert_object_by_parent(topology, parent, obj);
    /* insert_object_by_parent() doesn't merge during insert, so obj is still valid */
  }

  /* process object subnodes, if we found one win the above loop */
  while (tag) {
    int ret;

    if (!strcmp(tag, "object")) {
      hwloc_obj_t childobj = hwloc_alloc_setup_object(topology, HWLOC_OBJ_TYPE_MAX, HWLOC_UNKNOWN_INDEX);
      childobj->parent = ignored ? parent : obj;
      ret = hwloc__xml_import_object(topology, data, ignored ? parent : obj, childobj,
				     &childrengotignored,
				     &childstate);
    } else {
      if (hwloc__xml_verbose())
	fprintf(stderr, "%s: invalid special object child %s while looking for objects\n",
		state->global->msgprefix, tag);
      ret = -1;
    }

    if (ret < 0) {
      if (parent && !ignored)
        goto error;
      else
        goto error_with_object;
    }

    state->global->close_child(&childstate);

    tag = NULL;
    ret = state->global->find_child(state, &childstate, &tag);
    if (ret < 0) {
      if (parent && !ignored)
        goto error;
      else
        goto error_with_object;
    }
    if (!ret)
      break;
  }

  if (numa_was_root) {
    /* duplicate NUMA infos to root, most of them are likely root-specific */
    unsigned i;
    for(i=0; i<obj->infos_count; i++) {
      struct hwloc_info_s *info = &obj->infos[i];
      hwloc_obj_add_info(parent, info->name, info->value);
    }
    /* TODO some infos are root-only (hwlocVersion, ProcessName, etc), remove them from obj? */
  }

  if (ignored) {
    /* drop that object, and tell the parent that one child got ignored */
    hwloc_free_unlinked_object(obj);
    *gotignored = 1;

  } else if (obj->first_child) {
    /* now that all children are inserted, make sure they are in-order,
     * so that the core doesn't have to deal with crappy children list.
     */
    hwloc_obj_t cur, next;
    for(cur = obj->first_child, next = cur->next_sibling;
	next;
	cur = next, next = next->next_sibling) {
      /* If reordering is needed, at least one pair of consecutive children will be out-of-order.
       * So just check pairs of consecutive children.
       *
       * We checked above that complete_cpuset is always set.
       */
      if (hwloc_bitmap_compare_first(next->complete_cpuset, cur->complete_cpuset) < 0) {
	/* next should be before cur */
	if (!childrengotignored) {
	  static int reported = 0;
	  if (!reported && !hwloc_hide_errors()) {
	    hwloc__xml_import_report_outoforder(topology, next, cur);
	    reported = 1;
	  }
	}
	hwloc__reorder_children(obj);
	break;
      }
    }
    /* no need to reorder memory children as long as there are no intermediate memory objects
     * that could cause reordering when filtered-out.
     */
  }

  return state->global->close_tag(state);

 error_with_object:
  if (parent)
    /* root->parent is NULL, and root is already inserted. the caller will cleanup that root. */
    hwloc_free_unlinked_object(obj);
 error:
  return -1;
}

static int
hwloc__xml_v2import_support(hwloc_topology_t topology,
                            hwloc__xml_import_state_t state)
{
  char *name = NULL;
  int value = 1; /* value is optional */
  while (1) {
    char *attrname, *attrvalue;
    if (state->global->next_attr(state, &attrname, &attrvalue) < 0)
      break;
    if (!strcmp(attrname, "name"))
      name = attrvalue;
    else if (!strcmp(attrname, "value"))
      value = atoi(attrvalue);
    else {
      if (hwloc__xml_verbose())
	fprintf(stderr, "%s: ignoring unknown support attribute %s\n",
		state->global->msgprefix, attrname);
    }
  }

  if (name && topology->flags & HWLOC_TOPOLOGY_FLAG_IMPORT_SUPPORT) {
#ifdef HWLOC_DEBUG
    HWLOC_BUILD_ASSERT(sizeof(struct hwloc_topology_support) == 4*sizeof(void*));
    HWLOC_BUILD_ASSERT(sizeof(struct hwloc_topology_discovery_support) == 6);
    HWLOC_BUILD_ASSERT(sizeof(struct hwloc_topology_cpubind_support) == 11);
    HWLOC_BUILD_ASSERT(sizeof(struct hwloc_topology_membind_support) == 15);
    HWLOC_BUILD_ASSERT(sizeof(struct hwloc_topology_misc_support) == 1);
#endif

#define DO(_cat,_name) if (!strcmp(#_cat "." #_name, name)) topology->support._cat->_name = value
    DO(discovery,pu);
    else DO(discovery,numa);
    else DO(discovery,numa_memory);
    else DO(discovery,disallowed_pu);
    else DO(discovery,disallowed_numa);
    else DO(discovery,cpukind_efficiency);
    else DO(cpubind,set_thisproc_cpubind);
    else DO(cpubind,get_thisproc_cpubind);
    else DO(cpubind,set_proc_cpubind);
    else DO(cpubind,get_proc_cpubind);
    else DO(cpubind,set_thisthread_cpubind);
    else DO(cpubind,get_thisthread_cpubind);
    else DO(cpubind,set_thread_cpubind);
    else DO(cpubind,get_thread_cpubind);
    else DO(cpubind,get_thisproc_last_cpu_location);
    else DO(cpubind,get_proc_last_cpu_location);
    else DO(cpubind,get_thisthread_last_cpu_location);
    else DO(membind,set_thisproc_membind);
    else DO(membind,get_thisproc_membind);
    else DO(membind,set_proc_membind);
    else DO(membind,get_proc_membind);
    else DO(membind,set_thisthread_membind);
    else DO(membind,get_thisthread_membind);
    else DO(membind,set_area_membind);
    else DO(membind,get_area_membind);
    else DO(membind,alloc_membind);
    else DO(membind,firsttouch_membind);
    else DO(membind,bind_membind);
    else DO(membind,interleave_membind);
    else DO(membind,nexttouch_membind);
    else DO(membind,migrate_membind);
    else DO(membind,get_area_memlocation);

    else if (!strcmp("custom.exported_support", name))
      /* support was exported in a custom/fake field, mark it as imported here */
      topology->support.misc->imported_support = 1;

#undef DO
  }

  return 0;
}

static int
hwloc__xml_v2import_distances(hwloc_topology_t topology,
			      hwloc__xml_import_state_t state,
			      int heterotypes)
{
  hwloc_obj_type_t unique_type = HWLOC_OBJ_TYPE_NONE;
  hwloc_obj_type_t *different_types = NULL;
  unsigned nbobjs = 0;
  int indexing = heterotypes;
  int os_indexing = 0;
  int gp_indexing = heterotypes;
  char *name = NULL;
  unsigned long kind = 0;
  unsigned nr_indexes, nr_u64values;
  uint64_t *indexes;
  uint64_t *u64values;
  int ret;

#define _TAG_NAME (heterotypes ? "distances2hetero" : "distances2")

  /* process attributes */
  while (1) {
    char *attrname, *attrvalue;
    if (state->global->next_attr(state, &attrname, &attrvalue) < 0)
      break;
    if (!strcmp(attrname, "nbobjs"))
      nbobjs = strtoul(attrvalue, NULL, 10);
    else if (!strcmp(attrname, "type")) {
      if (hwloc_type_sscanf(attrvalue, &unique_type, NULL, 0) < 0) {
	if (hwloc__xml_verbose())
	  fprintf(stderr, "%s: unrecognized %s type %s\n",
		  state->global->msgprefix, _TAG_NAME, attrvalue);
	goto out;
      }
    }
    else if (!strcmp(attrname, "indexing")) {
      indexing = 1;
      if (!strcmp(attrvalue, "os"))
	os_indexing = 1;
      else if (!strcmp(attrvalue, "gp"))
	gp_indexing = 1;
    }
    else if (!strcmp(attrname, "kind")) {
      kind = strtoul(attrvalue, NULL, 10);
    }
    else if (!strcmp(attrname, "name")) {
      name = attrvalue;
    }
    else {
      if (hwloc__xml_verbose())
	fprintf(stderr, "%s: ignoring unknown %s attribute %s\n",
		state->global->msgprefix, _TAG_NAME, attrname);
    }
  }

  /* abort if missing attribute */
  if (!nbobjs || (!heterotypes && unique_type == HWLOC_OBJ_TYPE_NONE) || !indexing || !kind) {
    if (hwloc__xml_verbose())
      fprintf(stderr, "%s: %s missing some attributes\n",
	      state->global->msgprefix, _TAG_NAME);
    goto out;
  }

  indexes = malloc(nbobjs*sizeof(*indexes));
  u64values = malloc(nbobjs*nbobjs*sizeof(*u64values));
  if (heterotypes)
    different_types = malloc(nbobjs*sizeof(*different_types));
  if (!indexes || !u64values || (heterotypes && !different_types)) {
    if (hwloc__xml_verbose())
      fprintf(stderr, "%s: failed to allocate %s arrays for %u objects\n",
	      state->global->msgprefix, _TAG_NAME, nbobjs);
    goto out_with_arrays;
  }

  /* process children */
  nr_indexes = 0;
  nr_u64values = 0;
  while (1) {
    struct hwloc__xml_import_state_s childstate;
    char *attrname, *attrvalue, *tag;
    const char *buffer;
    int length;
    int is_index = 0;
    int is_u64values = 0;

    ret = state->global->find_child(state, &childstate, &tag);
    if (ret <= 0)
      break;

    if (!strcmp(tag, "indexes"))
      is_index = 1;
    else if (!strcmp(tag, "u64values"))
      is_u64values = 1;
    if (!is_index && !is_u64values) {
      if (hwloc__xml_verbose())
	fprintf(stderr, "%s: %s with unrecognized child %s\n",
		state->global->msgprefix, _TAG_NAME, tag);
      goto out_with_arrays;
    }

    if (state->global->next_attr(&childstate, &attrname, &attrvalue) < 0
	|| strcmp(attrname, "length")) {
      if (hwloc__xml_verbose())
	fprintf(stderr, "%s: %s child must have length attribute\n",
		state->global->msgprefix, _TAG_NAME);
      goto out_with_arrays;
    }
    length = atoi(attrvalue);

    ret = state->global->get_content(&childstate, &buffer, length);
    if (ret < 0) {
      if (hwloc__xml_verbose())
	fprintf(stderr, "%s: %s child needs content of length %d\n",
		state->global->msgprefix, _TAG_NAME, length);
      goto out_with_arrays;
    }

    if (is_index) {
      /* get indexes */
      const char *tmp, *tmp2;
      if (nr_indexes >= nbobjs) {
	if (hwloc__xml_verbose())
	  fprintf(stderr, "%s: %s with more than %u indexes\n",
		  state->global->msgprefix, _TAG_NAME, nbobjs);
	goto out_with_arrays;
      }
      tmp = buffer;
      while (1) {
	char *next;
	unsigned long long u;
	if (heterotypes) {
	  hwloc_obj_type_t t = HWLOC_OBJ_TYPE_NONE;
          if (!*tmp)
            /* reached the end of this indexes attribute */
            break;
	  if (hwloc_type_sscanf(tmp, &t, NULL, 0) < 0) {
	    if (hwloc__xml_verbose())
	      fprintf(stderr, "%s: %s with unrecognized heterogeneous type %s\n",
		      state->global->msgprefix, _TAG_NAME, tmp);
	    goto out_with_arrays;
	  }
	  tmp2 = strchr(tmp, ':');
	  if (!tmp2) {
	    if (hwloc__xml_verbose())
	      fprintf(stderr, "%s: %s with missing colon after heterogeneous type %s\n",
		      state->global->msgprefix, _TAG_NAME, tmp);
	    goto out_with_arrays;
	  }
	  tmp = tmp2+1;
	  different_types[nr_indexes] = t;
	}
	u = strtoull(tmp, &next, 0);
	if (next == tmp)
	  break;
	indexes[nr_indexes++] = u;
	if (*next != ' ')
	  break;
	if (nr_indexes == nbobjs)
	  break;
	tmp = next+1;
      }

    } else if (is_u64values) {
      /* get uint64_t values */
      const char *tmp;
      if (nr_u64values >= nbobjs*nbobjs) {
	if (hwloc__xml_verbose())
	  fprintf(stderr, "%s: %s with more than %u u64values\n",
		  state->global->msgprefix, _TAG_NAME, nbobjs*nbobjs);
	goto out_with_arrays;
      }
      tmp = buffer;
      while (1) {
	char *next;
	unsigned long long u = strtoull(tmp, &next, 0);
	if (next == tmp)
	  break;
	u64values[nr_u64values++] = u;
	if (*next != ' ')
	  break;
	if (nr_u64values == nbobjs*nbobjs)
	  break;
	tmp = next+1;
      }
    }

    state->global->close_content(&childstate);

    ret = state->global->close_tag(&childstate);
    if (ret < 0) {
      if (hwloc__xml_verbose())
	fprintf(stderr, "%s: %s with more than %u indexes\n",
		state->global->msgprefix, _TAG_NAME, nbobjs);
      goto out_with_arrays;
    }

    state->global->close_child(&childstate);
  }

  if (nr_indexes != nbobjs) {
    if (hwloc__xml_verbose())
      fprintf(stderr, "%s: %s with less than %u indexes\n",
	      state->global->msgprefix, _TAG_NAME, nbobjs);
    goto out_with_arrays;
  }
  if (nr_u64values != nbobjs*nbobjs) {
    if (hwloc__xml_verbose())
      fprintf(stderr, "%s: %s with less than %u u64values\n",
	      state->global->msgprefix, _TAG_NAME, nbobjs*nbobjs);
    goto out_with_arrays;
  }

  if (nbobjs < 2) {
    /* distances with a single object are useless, even if the XML isn't invalid */
    if (hwloc__xml_verbose())
      fprintf(stderr, "%s: ignoring %s with only %u objects\n",
	      state->global->msgprefix, _TAG_NAME, nbobjs);
    goto out_ignore;
  }
  if (unique_type == HWLOC_OBJ_PU || unique_type == HWLOC_OBJ_NUMANODE) {
    if (!os_indexing) {
      if (hwloc__xml_verbose())
	fprintf(stderr, "%s: ignoring PU or NUMA %s without os_indexing\n",
		state->global->msgprefix, _TAG_NAME);
      goto out_ignore;
    }
  } else {
    if (!gp_indexing) {
      if (hwloc__xml_verbose())
	fprintf(stderr, "%s: ignoring !PU or !NUMA %s without gp_indexing\n",
		state->global->msgprefix, _TAG_NAME);
      goto out_ignore;
    }
  }

  hwloc_internal_distances_add_by_index(topology, name, unique_type, different_types, nbobjs, indexes, u64values, kind, 0);

  /* prevent freeing below */
  indexes = NULL;
  u64values = NULL;
  different_types = NULL;

 out_ignore:
  free(different_types);
  free(indexes);
  free(u64values);
  return state->global->close_tag(state);

 out_with_arrays:
  free(different_types);
  free(indexes);
  free(u64values);
 out:
  return -1;
#undef _TAG_NAME
}

static int
hwloc__xml_import_memattr_value(hwloc_topology_t topology,
                                hwloc_memattr_id_t id,
                                unsigned long flags,
                                hwloc__xml_import_state_t state)
{
  char *target_obj_gp_index_s = NULL;
  char *target_obj_type_s = NULL;
  hwloc_uint64_t target_obj_gp_index;
  char *value_s = NULL;
  hwloc_uint64_t value;
  char *initiator_cpuset_s = NULL;
  char *initiator_obj_gp_index_s = NULL;
  char *initiator_obj_type_s = NULL;
  hwloc_obj_type_t target_obj_type = HWLOC_OBJ_TYPE_NONE;

  while (1) {
    char *attrname, *attrvalue;
    if (state->global->next_attr(state, &attrname, &attrvalue) < 0)
      break;
    if (!strcmp(attrname, "target_obj_gp_index"))
      target_obj_gp_index_s = attrvalue;
    else if (!strcmp(attrname, "target_obj_type"))
      target_obj_type_s = attrvalue;
    else if (!strcmp(attrname, "value"))
      value_s = attrvalue;
    else if (!strcmp(attrname, "initiator_cpuset"))
      initiator_cpuset_s = attrvalue;
    else if (!strcmp(attrname, "initiator_obj_gp_index"))
      initiator_obj_gp_index_s = attrvalue;
    else if (!strcmp(attrname, "initiator_obj_type"))
      initiator_obj_type_s = attrvalue;
    else {
      if (hwloc__xml_verbose())
        fprintf(stderr, "%s: ignoring unknown memattr_value attribute %s\n",
                state->global->msgprefix, attrname);
      return -1;
    }
  }

  if (!target_obj_type_s) {
    if (hwloc__xml_verbose())
      fprintf(stderr, "%s: ignoring memattr_value without target_obj_type.\n",
              state->global->msgprefix);
    return -1;
  }
  if (hwloc_type_sscanf(target_obj_type_s, &target_obj_type, NULL, 0) < 0) {
    if (hwloc__xml_verbose())
      fprintf(stderr, "%s: failed to identify memattr_value target object type %s\n",
              state->global->msgprefix, target_obj_type_s);
    return -1;
  }

  if (!value_s || !target_obj_gp_index_s) {
    if (hwloc__xml_verbose())
      fprintf(stderr, "%s: ignoring memattr_value without value and target_obj_gp_index\n",
              state->global->msgprefix);
    return -1;
  }
  target_obj_gp_index = strtoull(target_obj_gp_index_s, NULL, 10);
  value = strtoull(value_s, NULL, 10);

  if (flags & HWLOC_MEMATTR_FLAG_NEED_INITIATOR) {
    /* add a value with initiator */
    struct hwloc_internal_location_s loc;
    if (!initiator_cpuset_s && (!initiator_obj_gp_index_s || !initiator_obj_type_s)) {
      if (hwloc__xml_verbose())
        fprintf(stderr, "%s: ignoring memattr_value without initiator attributes\n",
                state->global->msgprefix);
      return -1;
    }

    /* setup the initiator */
    if (initiator_cpuset_s) {
      loc.type = HWLOC_LOCATION_TYPE_CPUSET;
      loc.location.cpuset = hwloc_bitmap_alloc();
      if (!loc.location.cpuset) {
        if (hwloc__xml_verbose())
          fprintf(stderr, "%s: failed to allocated memattr_value initiator cpuset\n",
                  state->global->msgprefix);
        return -1;
      }
      hwloc_bitmap_sscanf(loc.location.cpuset, initiator_cpuset_s);
    } else {
      loc.type = HWLOC_LOCATION_TYPE_OBJECT;
      loc.location.object.gp_index = strtoull(initiator_obj_gp_index_s, NULL, 10);
      if (hwloc_type_sscanf(initiator_obj_type_s, &loc.location.object.type, NULL, 0) < 0) {
        if (hwloc__xml_verbose())
          fprintf(stderr, "%s: failed to identify memattr_value initiator object type %s\n",
                  state->global->msgprefix, initiator_obj_type_s);
        return -1;
      }
    }

    hwloc_internal_memattr_set_value(topology, id, target_obj_type, target_obj_gp_index, (unsigned)-1, &loc, value);

    if (loc.type == HWLOC_LOCATION_TYPE_CPUSET)
      hwloc_bitmap_free(loc.location.cpuset);

  } else {
    /* add a value without initiator */
    hwloc_internal_memattr_set_value(topology, id, target_obj_type, target_obj_gp_index, (unsigned)-1, NULL, value);
  }

  return 0;
}

static int
hwloc__xml_import_memattr(hwloc_topology_t topology,
                          hwloc__xml_import_state_t state)
{
  char *name = NULL;
  unsigned long flags = (unsigned long) -1;
  hwloc_memattr_id_t id = (hwloc_memattr_id_t) -1;
  int ret;

  while (1) {
    char *attrname, *attrvalue;
    if (state->global->next_attr(state, &attrname, &attrvalue) < 0)
      break;
    if (!strcmp(attrname, "name"))
      name = attrvalue;
    else if (!strcmp(attrname, "flags"))
      flags = strtoul(attrvalue, NULL, 10);
    else {
      if (hwloc__xml_verbose())
        fprintf(stderr, "%s: ignoring unknown memattr attribute %s\n",
                state->global->msgprefix, attrname);
      return -1;
    }
  }

  if (name && flags != (unsigned long) -1) {
    hwloc_memattr_id_t _id;

    ret = hwloc_memattr_get_by_name(topology, name, &_id);
    if (ret < 0) {
      /* register a new attribute */
      ret = hwloc_memattr_register(topology, name, flags, &_id);
      if (!ret)
        id = _id;
    } else {
      /* check the flags of the existing attribute  */
      unsigned long mflags;
      ret = hwloc_memattr_get_flags(topology, _id, &mflags);
      if (!ret && mflags == flags)
        id = _id;
    }
    /* if there's no matching attribute, id is -1 and values will be ignored below */
  }

  while (1) {
    struct hwloc__xml_import_state_s childstate;
    char *tag;

    ret = state->global->find_child(state, &childstate, &tag);
    if (ret <= 0)
      break;

    if (!strcmp(tag, "memattr_value")) {
      ret = hwloc__xml_import_memattr_value(topology, id, flags, &childstate);
    } else {
      if (hwloc__xml_verbose())
        fprintf(stderr, "%s: memattr with unrecognized child %s\n",
                state->global->msgprefix, tag);
      ret = -1;
    }

    if (ret < 0)
      goto error;

    state->global->close_child(&childstate);
  }

  return state->global->close_tag(state);

 error:
  return -1;
}

static int
hwloc__xml_import_cpukind(hwloc_topology_t topology,
                          hwloc__xml_import_state_t state)
{
  hwloc_bitmap_t cpuset = NULL;
  int forced_efficiency = HWLOC_CPUKIND_EFFICIENCY_UNKNOWN;
  unsigned nr_infos = 0;
  struct hwloc_info_s *infos = NULL;
  int ret;

  while (1) {
    char *attrname, *attrvalue;
    if (state->global->next_attr(state, &attrname, &attrvalue) < 0)
      break;
    if (!strcmp(attrname, "cpuset")) {
      if (!cpuset)
        cpuset = hwloc_bitmap_alloc();
      hwloc_bitmap_sscanf(cpuset, attrvalue);
    } else if (!strcmp(attrname, "forced_efficiency")) {
      forced_efficiency = atoi(attrvalue);
    } else {
      if (hwloc__xml_verbose())
        fprintf(stderr, "%s: ignoring unknown cpukind attribute %s\n",
                state->global->msgprefix, attrname);
      hwloc_bitmap_free(cpuset);
      return -1;
    }
  }

  while (1) {
    struct hwloc__xml_import_state_s childstate;
    char *tag;

    ret = state->global->find_child(state, &childstate, &tag);
    if (ret <= 0)
      break;

    if (!strcmp(tag, "info")) {
      char *infoname = NULL;
      char *infovalue = NULL;
      ret = hwloc___xml_import_info(&infoname, &infovalue, &childstate);
      if (!ret && infoname && infovalue)
        hwloc__add_info(&infos, &nr_infos, infoname, infovalue);
    } else {
      if (hwloc__xml_verbose())
        fprintf(stderr, "%s: cpukind with unrecognized child %s\n",
                state->global->msgprefix, tag);
      ret = -1;
    }

    if (ret < 0)
      goto error;

    state->global->close_child(&childstate);
  }

  if (!cpuset) {
    if (hwloc__xml_verbose())
      fprintf(stderr, "%s: ignoring cpukind without cpuset\n",
              state->global->msgprefix);
    goto error;
  }

  hwloc_internal_cpukinds_register(topology, cpuset, forced_efficiency, infos, nr_infos, HWLOC_CPUKINDS_REGISTER_FLAG_OVERWRITE_FORCED_EFFICIENCY);

  return state->global->close_tag(state);

 error:
  hwloc__free_infos(infos, nr_infos);
  hwloc_bitmap_free(cpuset);
  return -1;
}

static int
hwloc__xml_import_diff_one(hwloc__xml_import_state_t state,
			   hwloc_topology_diff_t *firstdiffp,
			   hwloc_topology_diff_t *lastdiffp)
{
  char *type_s = NULL;
  char *obj_depth_s = NULL;
  char *obj_index_s = NULL;
  char *obj_attr_type_s = NULL;
/* char *obj_attr_index_s = NULL; unused for now */
  char *obj_attr_name_s = NULL;
  char *obj_attr_oldvalue_s = NULL;
  char *obj_attr_newvalue_s = NULL;

  while (1) {
    char *attrname, *attrvalue;
    if (state->global->next_attr(state, &attrname, &attrvalue) < 0)
      break;
    if (!strcmp(attrname, "type"))
      type_s = attrvalue;
    else if (!strcmp(attrname, "obj_depth"))
      obj_depth_s = attrvalue;
    else if (!strcmp(attrname, "obj_index"))
      obj_index_s = attrvalue;
    else if (!strcmp(attrname, "obj_attr_type"))
      obj_attr_type_s = attrvalue;
    else if (!strcmp(attrname, "obj_attr_index"))
      { /* obj_attr_index_s = attrvalue; unused for now */ }
    else if (!strcmp(attrname, "obj_attr_name"))
      obj_attr_name_s = attrvalue;
    else if (!strcmp(attrname, "obj_attr_oldvalue"))
      obj_attr_oldvalue_s = attrvalue;
    else if (!strcmp(attrname, "obj_attr_newvalue"))
      obj_attr_newvalue_s = attrvalue;
    else {
      if (hwloc__xml_verbose())
	fprintf(stderr, "%s: ignoring unknown diff attribute %s\n",
		state->global->msgprefix, attrname);
      return -1;
    }
  }

  if (type_s) {
    switch (atoi(type_s)) {
    default:
      break;
    case HWLOC_TOPOLOGY_DIFF_OBJ_ATTR: {
      /* object attribute diff */
      hwloc_topology_diff_obj_attr_type_t obj_attr_type;
      hwloc_topology_diff_t diff;

      /* obj_attr mandatory generic attributes */
      if (!obj_depth_s || !obj_index_s || !obj_attr_type_s) {
	if (hwloc__xml_verbose())
	  fprintf(stderr, "%s: missing mandatory obj attr generic attributes\n",
		  state->global->msgprefix);
	break;
      }

      /* obj_attr mandatory attributes common to all subtypes */
      if (!obj_attr_oldvalue_s || !obj_attr_newvalue_s) {
	if (hwloc__xml_verbose())
	  fprintf(stderr, "%s: missing mandatory obj attr value attributes\n",
		  state->global->msgprefix);
	break;
      }

      /* mandatory attributes for obj_attr_info subtype */
      obj_attr_type = atoi(obj_attr_type_s);
      if (obj_attr_type == HWLOC_TOPOLOGY_DIFF_OBJ_ATTR_INFO && !obj_attr_name_s) {
	if (hwloc__xml_verbose())
	  fprintf(stderr, "%s: missing mandatory obj attr info name attribute\n",
		  state->global->msgprefix);
	break;
      }

      /* now we know we have everything we need */
      diff = malloc(sizeof(*diff));
      if (!diff)
	return -1;
      diff->obj_attr.type = HWLOC_TOPOLOGY_DIFF_OBJ_ATTR;
      diff->obj_attr.obj_depth = atoi(obj_depth_s);
      diff->obj_attr.obj_index = atoi(obj_index_s);
      memset(&diff->obj_attr.diff, 0, sizeof(diff->obj_attr.diff));
      diff->obj_attr.diff.generic.type = obj_attr_type;

      switch (obj_attr_type) {
      case HWLOC_TOPOLOGY_DIFF_OBJ_ATTR_SIZE:
	diff->obj_attr.diff.uint64.oldvalue = strtoull(obj_attr_oldvalue_s, NULL, 0);
	diff->obj_attr.diff.uint64.newvalue = strtoull(obj_attr_newvalue_s, NULL, 0);
	break;
      case HWLOC_TOPOLOGY_DIFF_OBJ_ATTR_INFO:
	diff->obj_attr.diff.string.name = strdup(obj_attr_name_s);
	/* FALLTHRU */
      case HWLOC_TOPOLOGY_DIFF_OBJ_ATTR_NAME:
	diff->obj_attr.diff.string.oldvalue = strdup(obj_attr_oldvalue_s);
	diff->obj_attr.diff.string.newvalue = strdup(obj_attr_newvalue_s);
	break;
      }

      if (*firstdiffp)
	(*lastdiffp)->generic.next = diff;
      else
        *firstdiffp = diff;
      *lastdiffp = diff;
      diff->generic.next = NULL;
    }
    }
  }

  return state->global->close_tag(state);
}

int
hwloc__xml_import_diff(hwloc__xml_import_state_t state,
		       hwloc_topology_diff_t *firstdiffp)
{
  hwloc_topology_diff_t firstdiff = NULL, lastdiff = NULL;
  *firstdiffp = NULL;

  while (1) {
    struct hwloc__xml_import_state_s childstate;
    char *tag;
    int ret;

    ret = state->global->find_child(state, &childstate, &tag);
    if (ret < 0)
      return -1;
    if (!ret)
      break;

    if (!strcmp(tag, "diff")) {
      ret = hwloc__xml_import_diff_one(&childstate, &firstdiff, &lastdiff);
    } else
      ret = -1;

    if (ret < 0)
      return ret;

    state->global->close_child(&childstate);
  }

  *firstdiffp = firstdiff;
  return 0;
}

/***********************************
 ********* main XML import *********
 ***********************************/

static void
hwloc_convert_from_v1dist_floats(hwloc_topology_t topology, unsigned nbobjs, float *floats, uint64_t *u64s)
{
  unsigned i;
  int is_uint;
  char *env;
  float scale = 1000.f;
  char scalestring[20];

  env = getenv("HWLOC_XML_V1DIST_SCALE");
  if (env) {
    scale = (float) atof(env);
    goto scale;
  }

  is_uint = 1;
  /* find out if all values are integers */
  for(i=0; i<nbobjs*nbobjs; i++) {
    float f, iptr, fptr;
    f = floats[i];
    if (f < 0.f) {
      is_uint = 0;
      break;
    }
    fptr = modff(f, &iptr);
    if (fptr > .001f && fptr < .999f) {
      is_uint = 0;
      break;
    }
    u64s[i] = (int)(f+.5f);
  }
  if (is_uint)
    return;

 scale:
  /* TODO heuristic to find a good scale */
  for(i=0; i<nbobjs*nbobjs; i++)
    u64s[i] = (uint64_t)(scale * floats[i]);

  /* save the scale in root info attrs.
   * Not perfect since we may have multiple of them,
   * and some distances might disappear in case of restrict, etc.
   */
  sprintf(scalestring, "%f", scale);
  hwloc_obj_add_info(hwloc_get_root_obj(topology), "xmlv1DistancesScale", scalestring);
}

/* this canNOT be the first XML call */
static int
hwloc_look_xml(struct hwloc_backend *backend, struct hwloc_disc_status *dstatus)
{
  /*
   * This backend enforces !topology->is_thissystem by default.
   */

  struct hwloc_topology *topology = backend->topology;
  struct hwloc_xml_backend_data_s *data = backend->private_data;
  struct hwloc__xml_import_state_s state, childstate;
  struct hwloc_obj *root = topology->levels[0][0];
  char *tag;
  int gotignored = 0;
  hwloc_localeswitch_declare;
  int ret;

  assert(dstatus->phase == HWLOC_DISC_PHASE_GLOBAL);

  state.global = data;

  assert(!root->cpuset);

  hwloc_localeswitch_init();

  data->nbnumanodes = 0;
  data->first_numanode = data->last_numanode = NULL;
  data->first_v1dist = data->last_v1dist = NULL;

  ret = data->look_init(data, &state);
  if (ret < 0)
    goto failed;

  if (data->version_major > 2) {
    if (hwloc__xml_verbose())
      fprintf(stderr, "%s: cannot import XML version %u.%u > 2\n",
	      data->msgprefix, data->version_major, data->version_minor);
    goto err;
  }

  /* find root object tag and import it */
  ret = state.global->find_child(&state, &childstate, &tag);
  if (ret < 0 || !ret || strcmp(tag, "object"))
    goto failed;
  ret = hwloc__xml_import_object(topology, data, NULL /*  no parent */, root,
				 &gotignored,
				 &childstate);
  if (ret < 0)
    goto failed;
  state.global->close_child(&childstate);
  assert(!gotignored);

  /* the root may have changed if we had to reinsert a Machine */
  root = topology->levels[0][0];

  if (data->version_major >= 2) {
    /* find v2 distances */
    while (1) {
      ret = state.global->find_child(&state, &childstate, &tag);
      if (ret < 0)
	goto failed;
      if (!ret)
	break;
      if (!strcmp(tag, "distances2")) {
	ret = hwloc__xml_v2import_distances(topology, &childstate, 0);
	if (ret < 0)
	  goto failed;
      } else if (!strcmp(tag, "distances2hetero")) {
	ret = hwloc__xml_v2import_distances(topology, &childstate, 1);
	if (ret < 0)
	  goto failed;
      } else if (!strcmp(tag, "support")) {
	ret = hwloc__xml_v2import_support(topology, &childstate);
	if (ret < 0)
	  goto failed;
      } else if (!strcmp(tag, "memattr")) {
        ret = hwloc__xml_import_memattr(topology, &childstate);
        if (ret < 0)
          goto failed;
      } else if (!strcmp(tag, "cpukind")) {
        ret = hwloc__xml_import_cpukind(topology, &childstate);
        if (ret < 0)
          goto failed;
      } else {
	if (hwloc__xml_verbose())
	  fprintf(stderr, "%s: ignoring unknown tag `%s' after root object.\n",
		  data->msgprefix, tag);
	goto done;
      }
      state.global->close_child(&childstate);
    }
  }

  /* find end of topology tag */
  state.global->close_tag(&state);

done:
  if (!root->cpuset) {
    if (hwloc__xml_verbose())
      fprintf(stderr, "%s: invalid root object without cpuset\n",
	      data->msgprefix);
    goto err;
  }

  /* update pre-v2.0 memory group gp_index */
  if (data->version_major < 2 && data->first_numanode) {
    hwloc_obj_t node = data->first_numanode;
    do {
      if (node->parent->type == HWLOC_OBJ_GROUP
	  && !node->parent->gp_index)
	node->parent->gp_index = topology->next_gp_index++;
      node = node->next_cousin;
    } while (node);
  }

  if (data->version_major < 2 && data->first_v1dist) {
    /* handle v1 distances */
    struct hwloc__xml_imported_v1distances_s *v1dist, *v1next = data->first_v1dist;
    while ((v1dist = v1next) != NULL) {
      unsigned nbobjs = v1dist->nbobjs;
      v1next = v1dist->next;
      /* Handle distances as NUMA node distances if nbobjs matches.
       * Otherwise drop, only NUMA distances really matter.
       *
       * We could also attach to a random level with the right nbobjs,
       * but it would require to have those objects in the original XML order (like the first_numanode cousin-list).
       * because the topology order can be different if some parents are ignored during load.
       */
      if (nbobjs == data->nbnumanodes) {
	hwloc_obj_t *objs = malloc(nbobjs*sizeof(hwloc_obj_t));
	uint64_t *values = malloc(nbobjs*nbobjs*sizeof(*values));
        assert(data->nbnumanodes > 0); /* v1dist->nbobjs is >0 after import */
        assert(data->first_numanode);
	if (objs && values) {
	  hwloc_obj_t node;
	  unsigned i;
	  for(i=0, node = data->first_numanode;
	      i<nbobjs;
	      i++, node = node->next_cousin)
	    objs[i] = node;
	  hwloc_convert_from_v1dist_floats(topology, nbobjs, v1dist->floats, values);
	  hwloc_internal_distances_add(topology, NULL, nbobjs, objs, values, v1dist->kind, 0);
	} else {
	  free(objs);
	  free(values);
	}
      }
      free(v1dist->floats);
      free(v1dist);
    }
    data->first_v1dist = data->last_v1dist = NULL;
  }

  /* FIXME:
   * We should check that the existing object sets are consistent:
   * no intersection between objects of a same level,
   * object sets included in parent sets.
   * hwloc never generated such buggy XML, but users could create one.
   *
   * We want to add these checks to the existing core code that
   * adds missing sets and propagates parent/children sets
   * (in case another backend ever generates buggy object sets as well).
   */

  if (data->version_major >= 2) {
    /* v2 must have non-empty nodesets since at least one NUMA node is required */
    if (!root->nodeset) {
      if (hwloc__xml_verbose())
	fprintf(stderr, "%s: invalid root object without nodeset\n",
		data->msgprefix);
      goto err;
    }
    if (hwloc_bitmap_iszero(root->nodeset)) {
      if (hwloc__xml_verbose())
	fprintf(stderr, "%s: invalid root object with empty nodeset\n",
		data->msgprefix);
      goto err;
    }
  } else {
    /* if v1 without nodeset, the core will add a default NUMA node and nodesets */
  }

  /* allocate default cpusets and nodesets if missing, the core will restrict them */
  hwloc_alloc_root_sets(root);

  /* keep the "Backend" information intact */
  /* we could add "BackendSource=XML" to notify that XML was used between the actual backend and here */

  if (!(topology->flags & HWLOC_TOPOLOGY_FLAG_IMPORT_SUPPORT)) {
    topology->support.discovery->pu = 1;
    topology->support.discovery->disallowed_pu = 1;
    if (data->nbnumanodes) {
      topology->support.discovery->numa = 1;
      topology->support.discovery->numa_memory = 1; // FIXME
      topology->support.discovery->disallowed_numa = 1;
    }
  }

  if (data->look_done)
    data->look_done(data, 0);

  hwloc_localeswitch_fini();
  return 0;

 failed:
  if (data->look_done)
    data->look_done(data, -1);
  if (hwloc__xml_verbose())
    fprintf(stderr, "%s: XML component discovery failed.\n",
	    data->msgprefix);
 err:
  hwloc_free_object_siblings_and_children(root->first_child);
  root->first_child = NULL;
  hwloc_free_object_siblings_and_children(root->memory_first_child);
  root->memory_first_child = NULL;
  hwloc_free_object_siblings_and_children(root->io_first_child);
  root->io_first_child = NULL;
  hwloc_free_object_siblings_and_children(root->misc_first_child);
  root->misc_first_child = NULL;

  /* make sure the core will abort */
  if (root->cpuset)
    hwloc_bitmap_zero(root->cpuset);
  if (root->nodeset)
    hwloc_bitmap_zero(root->nodeset);

  hwloc_localeswitch_fini();
  return -1;
}

/* this can be the first XML call */
int
hwloc_topology_diff_load_xml(const char *xmlpath,
			     hwloc_topology_diff_t *firstdiffp, char **refnamep)
{
  struct hwloc__xml_import_state_s state;
  struct hwloc_xml_backend_data_s fakedata; /* only for storing global info during parsing */
  hwloc_localeswitch_declare;
  const char *local_basename;
  int force_nolibxml;
  int ret;

  state.global = &fakedata;

  local_basename = strrchr(xmlpath, '/');
  if (local_basename)
    local_basename++;
  else
    local_basename = xmlpath;
  fakedata.msgprefix = strdup(local_basename);

  hwloc_components_init();
  assert(hwloc_nolibxml_callbacks);

  hwloc_localeswitch_init();

  *firstdiffp = NULL;

  force_nolibxml = hwloc_nolibxml_import();
retry:
  if (!hwloc_libxml_callbacks || (hwloc_nolibxml_callbacks && force_nolibxml))
    ret = hwloc_nolibxml_callbacks->import_diff(&state, xmlpath, NULL, 0, firstdiffp, refnamep);
  else {
    ret = hwloc_libxml_callbacks->import_diff(&state, xmlpath, NULL, 0, firstdiffp, refnamep);
    if (ret < 0 && errno == ENOSYS) {
      hwloc_libxml_callbacks = NULL;
      goto retry;
    }
  }

  hwloc_localeswitch_fini();
  hwloc_components_fini();
  free(fakedata.msgprefix);
  return ret;
}

/* this can be the first XML call */
int
hwloc_topology_diff_load_xmlbuffer(const char *xmlbuffer, int buflen,
				   hwloc_topology_diff_t *firstdiffp, char **refnamep)
{
  struct hwloc__xml_import_state_s state;
  struct hwloc_xml_backend_data_s fakedata; /* only for storing global info during parsing */
  hwloc_localeswitch_declare;
  int force_nolibxml;
  int ret;

  state.global = &fakedata;
  fakedata.msgprefix = strdup("xmldiffbuffer");

  hwloc_components_init();
  assert(hwloc_nolibxml_callbacks);

  hwloc_localeswitch_init();

  *firstdiffp = NULL;

  force_nolibxml = hwloc_nolibxml_import();
 retry:
  if (!hwloc_libxml_callbacks || (hwloc_nolibxml_callbacks && force_nolibxml))
    ret = hwloc_nolibxml_callbacks->import_diff(&state, NULL, xmlbuffer, buflen, firstdiffp, refnamep);
  else {
    ret = hwloc_libxml_callbacks->import_diff(&state, NULL, xmlbuffer, buflen, firstdiffp, refnamep);
    if (ret < 0 && errno == ENOSYS) {
      hwloc_libxml_callbacks = NULL;
      goto retry;
    }
  }

  hwloc_localeswitch_fini();
  hwloc_components_fini();
  free(fakedata.msgprefix);
  return ret;
}

/************************************************
 ********* XML export (common routines) *********
 ************************************************/

#define HWLOC_XML_CHAR_VALID(c) (((c) >= 32 && (c) <= 126) || (c) == '\t' || (c) == '\n' || (c) == '\r')

static int
hwloc__xml_export_check_buffer(const char *buf, size_t length)
{
  unsigned i;
  for(i=0; i<length; i++)
    if (!HWLOC_XML_CHAR_VALID(buf[i]))
      return -1;
  return 0;
}

/* strdup and remove ugly chars from random string */
static char*
hwloc__xml_export_safestrdup(const char *old)
{
  char *new = malloc(strlen(old)+1);
  char *dst = new;
  const char *src = old;
  if (!new)
    return NULL;

  while (*src) {
    if (HWLOC_XML_CHAR_VALID(*src))
      *(dst++) = *src;
    src++;
  }
  *dst = '\0';
  return new;
}

static void
hwloc__xml_export_object_contents (hwloc__xml_export_state_t state, hwloc_topology_t topology, hwloc_obj_t obj, unsigned long flags)
{
  char *setstring = NULL, *setstring2 = NULL;
  char tmp[255];
  int v1export = flags & HWLOC_TOPOLOGY_EXPORT_XML_FLAG_V1;
  unsigned i,j;

  if (v1export && obj->type == HWLOC_OBJ_PACKAGE)
    state->new_prop(state, "type", "Socket");
  else if (v1export && obj->type == HWLOC_OBJ_DIE)
    state->new_prop(state, "type", "Group");
  else if (v1export && hwloc__obj_type_is_cache(obj->type))
    state->new_prop(state, "type", "Cache");
  else
    state->new_prop(state, "type", hwloc_obj_type_string(obj->type));

  if (obj->os_index != HWLOC_UNKNOWN_INDEX) {
    sprintf(tmp, "%u", obj->os_index);
    state->new_prop(state, "os_index", tmp);
  }

  if (obj->cpuset) {
    int empty_cpusets = 0;

    if (v1export && obj->type == HWLOC_OBJ_NUMANODE) {
      /* walk up this memory hierarchy to find-out if we are the first numa node.
       * v1 non-first NUMA nodes have empty cpusets.
       */
      hwloc_obj_t parent = obj;
      while (!hwloc_obj_type_is_normal(parent->type)) {
	if (parent->sibling_rank > 0) {
	  empty_cpusets = 1;
	  break;
	}
	parent = parent->parent;
      }
    }

    if (empty_cpusets) {
      state->new_prop(state, "cpuset", "0x0");
      state->new_prop(state, "online_cpuset", "0x0");
      state->new_prop(state, "complete_cpuset", "0x0");
      state->new_prop(state, "allowed_cpuset", "0x0");

    } else {
      /* normal case */
      hwloc_bitmap_asprintf(&setstring, obj->cpuset);
      state->new_prop(state, "cpuset", setstring);

      hwloc_bitmap_asprintf(&setstring2, obj->complete_cpuset);
      state->new_prop(state, "complete_cpuset", setstring2);
      free(setstring2);

      if (v1export)
	state->new_prop(state, "online_cpuset", setstring);
      free(setstring);

      if (v1export) {
	hwloc_bitmap_t allowed_cpuset = hwloc_bitmap_dup(obj->cpuset);
	hwloc_bitmap_and(allowed_cpuset, allowed_cpuset, topology->allowed_cpuset);
	hwloc_bitmap_asprintf(&setstring, allowed_cpuset);
	state->new_prop(state, "allowed_cpuset", setstring);
	free(setstring);
	hwloc_bitmap_free(allowed_cpuset);
      } else if (!obj->parent) {
	hwloc_bitmap_asprintf(&setstring, topology->allowed_cpuset);
	state->new_prop(state, "allowed_cpuset", setstring);
	free(setstring);
      }
    }

    /* If exporting v1, we should clear second local NUMA bits from nodeset,
     * but the importer will clear them anyway.
     */
    hwloc_bitmap_asprintf(&setstring, obj->nodeset);
    state->new_prop(state, "nodeset", setstring);
    free(setstring);

    hwloc_bitmap_asprintf(&setstring, obj->complete_nodeset);
    state->new_prop(state, "complete_nodeset", setstring);
    free(setstring);

    if (v1export) {
      hwloc_bitmap_t allowed_nodeset = hwloc_bitmap_dup(obj->nodeset);
      hwloc_bitmap_and(allowed_nodeset, allowed_nodeset, topology->allowed_nodeset);
      hwloc_bitmap_asprintf(&setstring, allowed_nodeset);
      state->new_prop(state, "allowed_nodeset", setstring);
      free(setstring);
      hwloc_bitmap_free(allowed_nodeset);
    } else if (!obj->parent) {
      hwloc_bitmap_asprintf(&setstring, topology->allowed_nodeset);
      state->new_prop(state, "allowed_nodeset", setstring);
      free(setstring);
    }
  }

  if (!v1export) {
    sprintf(tmp, "%llu", (unsigned long long) obj->gp_index);
    state->new_prop(state, "gp_index", tmp);
  }

  if (obj->name) {
    char *name = hwloc__xml_export_safestrdup(obj->name);
    if (name) {
      state->new_prop(state, "name", name);
      free(name);
    }
  }
  if (!v1export && obj->subtype) {
    char *subtype = hwloc__xml_export_safestrdup(obj->subtype);
    if (subtype) {
      state->new_prop(state, "subtype", subtype);
      free(subtype);
    }
  }

  switch (obj->type) {
  case HWLOC_OBJ_NUMANODE:
    if (obj->attr->numanode.local_memory) {
      sprintf(tmp, "%llu", (unsigned long long) obj->attr->numanode.local_memory);
      state->new_prop(state, "local_memory", tmp);
    }
    for(i=0; i<obj->attr->numanode.page_types_len; i++) {
      struct hwloc__xml_export_state_s childstate;
      state->new_child(state, &childstate, "page_type");
      sprintf(tmp, "%llu", (unsigned long long) obj->attr->numanode.page_types[i].size);
      childstate.new_prop(&childstate, "size", tmp);
      sprintf(tmp, "%llu", (unsigned long long) obj->attr->numanode.page_types[i].count);
      childstate.new_prop(&childstate, "count", tmp);
      childstate.end_object(&childstate, "page_type");
    }
    break;
  case HWLOC_OBJ_L1CACHE:
  case HWLOC_OBJ_L2CACHE:
  case HWLOC_OBJ_L3CACHE:
  case HWLOC_OBJ_L4CACHE:
  case HWLOC_OBJ_L5CACHE:
  case HWLOC_OBJ_L1ICACHE:
  case HWLOC_OBJ_L2ICACHE:
  case HWLOC_OBJ_L3ICACHE:
  case HWLOC_OBJ_MEMCACHE:
    sprintf(tmp, "%llu", (unsigned long long) obj->attr->cache.size);
    state->new_prop(state, "cache_size", tmp);
    sprintf(tmp, "%u", obj->attr->cache.depth);
    state->new_prop(state, "depth", tmp);
    sprintf(tmp, "%u", (unsigned) obj->attr->cache.linesize);
    state->new_prop(state, "cache_linesize", tmp);
    sprintf(tmp, "%d", obj->attr->cache.associativity);
    state->new_prop(state, "cache_associativity", tmp);
    sprintf(tmp, "%d", (int) obj->attr->cache.type);
    state->new_prop(state, "cache_type", tmp);
    break;
  case HWLOC_OBJ_GROUP:
    if (v1export) {
      sprintf(tmp, "%u", obj->attr->group.depth);
      state->new_prop(state, "depth", tmp);
      if (obj->attr->group.dont_merge)
        state->new_prop(state, "dont_merge", "1");
    } else {
      sprintf(tmp, "%u", obj->attr->group.kind);
      state->new_prop(state, "kind", tmp);
      sprintf(tmp, "%u", obj->attr->group.subkind);
      state->new_prop(state, "subkind", tmp);
      if (obj->attr->group.dont_merge)
        state->new_prop(state, "dont_merge", "1");
    }
    break;
  case HWLOC_OBJ_BRIDGE:
    sprintf(tmp, "%d-%d", (int) obj->attr->bridge.upstream_type, (int) obj->attr->bridge.downstream_type);
    state->new_prop(state, "bridge_type", tmp);
    sprintf(tmp, "%u", obj->attr->bridge.depth);
    state->new_prop(state, "depth", tmp);
    if (obj->attr->bridge.downstream_type == HWLOC_OBJ_BRIDGE_PCI) {
      sprintf(tmp, "%04x:[%02x-%02x]",
	      (unsigned) obj->attr->bridge.downstream.pci.domain,
	      (unsigned) obj->attr->bridge.downstream.pci.secondary_bus,
	      (unsigned) obj->attr->bridge.downstream.pci.subordinate_bus);
      state->new_prop(state, "bridge_pci", tmp);
    }
    if (obj->attr->bridge.upstream_type != HWLOC_OBJ_BRIDGE_PCI)
      break;
    /* FALLTHRU */
  case HWLOC_OBJ_PCI_DEVICE:
    sprintf(tmp, "%04x:%02x:%02x.%01x",
	    (unsigned) obj->attr->pcidev.domain,
	    (unsigned) obj->attr->pcidev.bus,
	    (unsigned) obj->attr->pcidev.dev,
	    (unsigned) obj->attr->pcidev.func);
    state->new_prop(state, "pci_busid", tmp);
    sprintf(tmp, "%04x [%04x:%04x] [%04x:%04x] %02x",
	    (unsigned) obj->attr->pcidev.class_id,
	    (unsigned) obj->attr->pcidev.vendor_id, (unsigned) obj->attr->pcidev.device_id,
	    (unsigned) obj->attr->pcidev.subvendor_id, (unsigned) obj->attr->pcidev.subdevice_id,
	    (unsigned) obj->attr->pcidev.revision);
    state->new_prop(state, "pci_type", tmp);
    sprintf(tmp, "%f", obj->attr->pcidev.linkspeed);
    state->new_prop(state, "pci_link_speed", tmp);
    break;
  case HWLOC_OBJ_OS_DEVICE:
    sprintf(tmp, "%d", (int) obj->attr->osdev.type);
    state->new_prop(state, "osdev_type", tmp);
    break;
  default:
    break;
  }

  for(i=0; i<obj->infos_count; i++) {
    char *name = hwloc__xml_export_safestrdup(obj->infos[i].name);
    char *value = hwloc__xml_export_safestrdup(obj->infos[i].value);
    if (name && value) {
      struct hwloc__xml_export_state_s childstate;
      state->new_child(state, &childstate, "info");
      childstate.new_prop(&childstate, "name", name);
      childstate.new_prop(&childstate, "value", value);
      childstate.end_object(&childstate, "info");
    }
    free(name);
    free(value);
  }
  if (v1export && obj->subtype) {
    char *subtype = hwloc__xml_export_safestrdup(obj->subtype);
    if (subtype) {
      struct hwloc__xml_export_state_s childstate;
      int is_coproctype = (obj->type == HWLOC_OBJ_OS_DEVICE && obj->attr->osdev.type == HWLOC_OBJ_OSDEV_COPROC);
      state->new_child(state, &childstate, "info");
      childstate.new_prop(&childstate, "name", is_coproctype ? "CoProcType" : "Type");
      childstate.new_prop(&childstate, "value", subtype);
      childstate.end_object(&childstate, "info");
      free(subtype);
    }
  }
  if (v1export && obj->type == HWLOC_OBJ_DIE) {
    struct hwloc__xml_export_state_s childstate;
    state->new_child(state, &childstate, "info");
    childstate.new_prop(&childstate, "name", "Type");
    childstate.new_prop(&childstate, "value", "Die");
    childstate.end_object(&childstate, "info");
  }

  if (v1export && !obj->parent) {
    /* only latency matrices covering the entire machine can be exported to v1 */
    struct hwloc_internal_distances_s *dist;
    /* refresh distances since we need objects below */
    hwloc_internal_distances_refresh(topology);
    for(dist = topology->first_dist; dist; dist = dist->next) {
      struct hwloc__xml_export_state_s childstate;
      unsigned nbobjs = dist->nbobjs;
      unsigned *logical_to_v2array;
      int depth;

      if (nbobjs != (unsigned) hwloc_get_nbobjs_by_type(topology, dist->unique_type))
	continue;
      if (!(dist->kind & HWLOC_DISTANCES_KIND_MEANS_LATENCY))
	continue;
      if (dist->kind & HWLOC_DISTANCES_KIND_HETEROGENEOUS_TYPES)
	continue;

      logical_to_v2array = malloc(nbobjs * sizeof(*logical_to_v2array));
      if (!logical_to_v2array) {
	fprintf(stderr, "xml/export/v1: failed to allocated logical_to_v2array\n");
	continue;
      }

      for(i=0; i<nbobjs; i++)
	logical_to_v2array[dist->objs[i]->logical_index] = i;

      /* compute the relative depth */
      if (dist->unique_type == HWLOC_OBJ_NUMANODE) {
	/* for NUMA nodes, use the highest normal-parent depth + 1 */
	depth = -1;
	for(i=0; i<nbobjs; i++) {
	  hwloc_obj_t parent = dist->objs[i]->parent;
	  while (hwloc__obj_type_is_memory(parent->type))
	    parent = parent->parent;
	  if (parent->depth+1 > depth)
	    depth = parent->depth+1;
	}
      } else {
	/* for non-NUMA nodes, increase the object depth if any of them has memory above */
	int parent_with_memory = 0;
	for(i=0; i<nbobjs; i++) {
	  hwloc_obj_t parent = dist->objs[i]->parent;
	  while (parent) {
	    if (parent->memory_first_child) {
	      parent_with_memory = 1;
	      goto done;
	    }
	    parent = parent->parent;
	  }
	}
      done:
	depth = hwloc_get_type_depth(topology, dist->unique_type) + parent_with_memory;
      }

      state->new_child(state, &childstate, "distances");
      sprintf(tmp, "%u", nbobjs);
      childstate.new_prop(&childstate, "nbobjs", tmp);
      sprintf(tmp, "%d", depth);
      childstate.new_prop(&childstate, "relative_depth", tmp);
      sprintf(tmp, "%f", 1.f);
      childstate.new_prop(&childstate, "latency_base", tmp);
      for(i=0; i<nbobjs; i++) {
        for(j=0; j<nbobjs; j++) {
	  /* we should export i*nbobjs+j, we translate using logical_to_v2array[] */
	  unsigned k = logical_to_v2array[i]*nbobjs+logical_to_v2array[j];
	  struct hwloc__xml_export_state_s greatchildstate;
	  childstate.new_child(&childstate, &greatchildstate, "latency");
	  sprintf(tmp, "%f", (float) dist->values[k]);
	  greatchildstate.new_prop(&greatchildstate, "value", tmp);
	  greatchildstate.end_object(&greatchildstate, "latency");
	}
      }
      childstate.end_object(&childstate, "distances");
      free(logical_to_v2array);
    }
  }

  if (obj->userdata && topology->userdata_export_cb)
    topology->userdata_export_cb((void*) state, topology, obj);
}

static void
hwloc__xml_v2export_object (hwloc__xml_export_state_t parentstate, hwloc_topology_t topology, hwloc_obj_t obj, unsigned long flags)
{
  struct hwloc__xml_export_state_s state;
  hwloc_obj_t child;

  parentstate->new_child(parentstate, &state, "object");

  hwloc__xml_export_object_contents(&state, topology, obj, flags);

  for_each_memory_child(child, obj)
    hwloc__xml_v2export_object (&state, topology, child, flags);
  for_each_child(child, obj)
    hwloc__xml_v2export_object (&state, topology, child, flags);
  for_each_io_child(child, obj)
    hwloc__xml_v2export_object (&state, topology, child, flags);
  for_each_misc_child(child, obj)
    hwloc__xml_v2export_object (&state, topology, child, flags);

  state.end_object(&state, "object");
}

static void
hwloc__xml_v1export_object (hwloc__xml_export_state_t parentstate, hwloc_topology_t topology, hwloc_obj_t obj, unsigned long flags);

static hwloc_obj_t
hwloc__xml_v1export_object_next_numanode(hwloc_obj_t obj, hwloc_obj_t cur)
{
  hwloc_obj_t parent;

  if (!cur) {
    /* first numa node is on the very bottom left */
    cur = obj->memory_first_child;
    goto find_first;
  }

  /* walk-up until there's a next sibling */
  parent = cur;
  while (1) {
    if (parent->next_sibling) {
      /* found a next sibling, we'll walk down-left from there */
      cur = parent->next_sibling;
      break;
    }
    parent = parent->parent;
    if (parent == obj)
      return NULL;
  }

 find_first:
  while (cur->type != HWLOC_OBJ_NUMANODE)
    cur = cur->memory_first_child;
  assert(cur);
  return cur;
}

static unsigned
hwloc__xml_v1export_object_list_numanodes(hwloc_obj_t obj, hwloc_obj_t *first_p, hwloc_obj_t **nodes_p)
{
  hwloc_obj_t *nodes, cur;
  int nr;

  if (!obj->memory_first_child) {
    *first_p = NULL;
    *nodes_p = NULL;
    return 0;
  }
  /* we're sure there's at least one numa node */

  nr = hwloc_bitmap_weight(obj->nodeset);
  assert(nr > 0);
  /* these are local nodes, but some of them may be attached above instead of here */

  nodes = calloc(nr, sizeof(*nodes));
  if (!nodes) {
    /* only return the first node */
    cur = hwloc__xml_v1export_object_next_numanode(obj, NULL);
    assert(cur);
    *first_p = cur;
    *nodes_p = NULL;
    return 1;
  }

  nr = 0;
  cur = NULL;
  while (1) {
    cur = hwloc__xml_v1export_object_next_numanode(obj, cur);
    if (!cur)
      break;
    nodes[nr++] = cur;
  }

  *first_p = nodes[0];
  *nodes_p = nodes;
  return nr;
}

static void
hwloc__xml_v1export_object_with_memory(hwloc__xml_export_state_t parentstate, hwloc_topology_t topology, hwloc_obj_t obj, unsigned long flags)
{
  struct hwloc__xml_export_state_s gstate, mstate, ostate, *state = parentstate;
  hwloc_obj_t child;
  unsigned nr_numanodes;
  hwloc_obj_t *numanodes, first_numanode;
  unsigned i;

  nr_numanodes = hwloc__xml_v1export_object_list_numanodes(obj, &first_numanode, &numanodes);

  if (obj->parent->arity > 1 && nr_numanodes > 1 && parentstate->global->v1_memory_group) {
    /* child has sibling, we must add a Group around those memory children */
    hwloc_obj_t group = parentstate->global->v1_memory_group;
    parentstate->new_child(parentstate, &gstate, "object");
    group->cpuset = obj->cpuset;
    group->complete_cpuset = obj->complete_cpuset;
    group->nodeset = obj->nodeset;
    group->complete_nodeset = obj->complete_nodeset;
    hwloc__xml_export_object_contents (&gstate, topology, group, flags);
    group->cpuset = NULL;
    group->complete_cpuset = NULL;
    group->nodeset = NULL;
    group->complete_nodeset = NULL;
    state = &gstate;
  }

  /* export first memory child */
  state->new_child(state, &mstate, "object");
  hwloc__xml_export_object_contents (&mstate, topology, first_numanode, flags);

  /* then the actual object */
  mstate.new_child(&mstate, &ostate, "object");
  hwloc__xml_export_object_contents (&ostate, topology, obj, flags);

  /* then its normal/io/misc children */
  for_each_child(child, obj)
    hwloc__xml_v1export_object (&ostate, topology, child, flags);
  for_each_io_child(child, obj)
    hwloc__xml_v1export_object (&ostate, topology, child, flags);
  for_each_misc_child(child, obj)
    hwloc__xml_v1export_object (&ostate, topology, child, flags);

  /* close object and first memory child */
  ostate.end_object(&ostate, "object");
  mstate.end_object(&mstate, "object");

  /* now other memory children */
  for(i=1; i<nr_numanodes; i++)
    hwloc__xml_v1export_object (state, topology, numanodes[i], flags);

  free(numanodes);

  if (state == &gstate) {
    /* close group if any */
    gstate.end_object(&gstate, "object");
  }
}

static void
hwloc__xml_v1export_object (hwloc__xml_export_state_t parentstate, hwloc_topology_t topology, hwloc_obj_t obj, unsigned long flags)
{
  struct hwloc__xml_export_state_s state;
  hwloc_obj_t child;

  parentstate->new_child(parentstate, &state, "object");

  hwloc__xml_export_object_contents(&state, topology, obj, flags);

  for_each_child(child, obj) {
    if (!child->memory_arity) {
      /* no memory child, just export normally */
      hwloc__xml_v1export_object (&state, topology, child, flags);
    } else {
      hwloc__xml_v1export_object_with_memory(&state, topology, child, flags);
    }
  }

  for_each_io_child(child, obj)
    hwloc__xml_v1export_object (&state, topology, child, flags);
  for_each_misc_child(child, obj)
    hwloc__xml_v1export_object (&state, topology, child, flags);

  state.end_object(&state, "object");
}

#define EXPORT_ARRAY(state, type, nr, values, tagname, format, maxperline) do { \
  unsigned _i = 0; \
  while (_i<(nr)) { \
    char _tmp[255]; /* enough for (snprintf(format)+space) x maxperline */ \
    char _tmp2[16]; \
    size_t _len = 0; \
    unsigned _j; \
    struct hwloc__xml_export_state_s _childstate; \
    (state)->new_child(state, &_childstate, tagname); \
    for(_j=0; \
	_i+_j<(nr) && _j<maxperline; \
	_j++) \
      _len += sprintf(_tmp+_len, format " ", (type) (values)[_i+_j]); \
    _i += _j; \
    sprintf(_tmp2, "%lu", (unsigned long) _len); \
    _childstate.new_prop(&_childstate, "length", _tmp2); \
    _childstate.add_content(&_childstate, _tmp, _len); \
    _childstate.end_object(&_childstate, tagname); \
  } \
} while (0)

#define EXPORT_TYPE_GPINDEX_ARRAY(state, nr, objs, tagname, maxperline) do { \
  unsigned _i = 0; \
  while (_i<(nr)) { \
    char _tmp[255]; /* enough for (snprintf(type+index)+space) x maxperline */ \
    char _tmp2[16]; \
    size_t _len = 0; \
    unsigned _j; \
    struct hwloc__xml_export_state_s _childstate; \
    (state)->new_child(state, &_childstate, tagname); \
    for(_j=0; \
	_i+_j<(nr) && _j<maxperline; \
	_j++) \
      _len += sprintf(_tmp+_len, "%s:%llu ", hwloc_obj_type_string((objs)[_i+_j]->type), (unsigned long long) (objs)[_i+_j]->gp_index); \
    _i += _j; \
    sprintf(_tmp2, "%lu", (unsigned long) _len); \
    _childstate.new_prop(&_childstate, "length", _tmp2); \
    _childstate.add_content(&_childstate, _tmp, _len); \
    _childstate.end_object(&_childstate, tagname); \
  } \
} while (0)

static void
hwloc___xml_v2export_distances(hwloc__xml_export_state_t parentstate, struct hwloc_internal_distances_s *dist)
{
  char tmp[255];
  unsigned nbobjs = dist->nbobjs;
  struct hwloc__xml_export_state_s state;

  if (dist->different_types) {
    parentstate->new_child(parentstate, &state, "distances2hetero");
  } else {
    parentstate->new_child(parentstate, &state, "distances2");
    state.new_prop(&state, "type", hwloc_obj_type_string(dist->unique_type));
  }

  sprintf(tmp, "%u", nbobjs);
  state.new_prop(&state, "nbobjs", tmp);
  sprintf(tmp, "%lu", dist->kind);
  state.new_prop(&state, "kind", tmp);
  if (dist->name)
    state.new_prop(&state, "name", dist->name);

  if (!dist->different_types) {
    state.new_prop(&state, "indexing",
		   HWLOC_DIST_TYPE_USE_OS_INDEX(dist->unique_type) ? "os" : "gp");
  }

  /* TODO don't hardwire 10 below. either snprintf the max to guess it, or just append until the end of the buffer */
  if (dist->different_types) {
    EXPORT_TYPE_GPINDEX_ARRAY(&state, nbobjs, dist->objs, "indexes", 10);
  } else {
    EXPORT_ARRAY(&state, unsigned long long, nbobjs, dist->indexes, "indexes", "%llu", 10);
  }
  EXPORT_ARRAY(&state, unsigned long long, nbobjs*nbobjs, dist->values, "u64values", "%llu", 10);
  state.end_object(&state, dist->different_types ? "distances2hetero" : "distances2");
}

static void
hwloc__xml_v2export_distances(hwloc__xml_export_state_t parentstate, hwloc_topology_t topology)
{
  struct hwloc_internal_distances_s *dist;
  for(dist = topology->first_dist; dist; dist = dist->next)
    if (!dist->different_types)
      hwloc___xml_v2export_distances(parentstate, dist);
  /* export homogeneous distances first in case the importer doesn't support heterogeneous and stops there */
  for(dist = topology->first_dist; dist; dist = dist->next)
    if (dist->different_types)
      hwloc___xml_v2export_distances(parentstate, dist);
}

static void
hwloc__xml_v2export_support(hwloc__xml_export_state_t parentstate, hwloc_topology_t topology)
{
  struct hwloc__xml_export_state_s state;
  char tmp[11];

#ifdef HWLOC_DEBUG
  HWLOC_BUILD_ASSERT(sizeof(struct hwloc_topology_support) == 4*sizeof(void*));
  HWLOC_BUILD_ASSERT(sizeof(struct hwloc_topology_discovery_support) == 6);
  HWLOC_BUILD_ASSERT(sizeof(struct hwloc_topology_cpubind_support) == 11);
  HWLOC_BUILD_ASSERT(sizeof(struct hwloc_topology_membind_support) == 15);
  HWLOC_BUILD_ASSERT(sizeof(struct hwloc_topology_misc_support) == 1);
#endif

#define DO(_cat,_name) do {                                     \
    if (topology->support._cat->_name) {                        \
      parentstate->new_child(parentstate, &state, "support");   \
      state.new_prop(&state, "name", #_cat "." #_name);         \
      if (topology->support._cat->_name != 1) {                 \
        sprintf(tmp, "%u", topology->support._cat->_name); \
        state.new_prop(&state, "value", tmp);                   \
      }                                                         \
      state.end_object(&state, "support");                      \
    }                                                           \
  } while (0)

  DO(discovery,pu);
  DO(discovery,numa);
  DO(discovery,numa_memory);
  DO(discovery,disallowed_pu);
  DO(discovery,disallowed_numa);
  DO(discovery,cpukind_efficiency);
  DO(cpubind,set_thisproc_cpubind);
  DO(cpubind,get_thisproc_cpubind);
  DO(cpubind,set_proc_cpubind);
  DO(cpubind,get_proc_cpubind);
  DO(cpubind,set_thisthread_cpubind);
  DO(cpubind,get_thisthread_cpubind);
  DO(cpubind,set_thread_cpubind);
  DO(cpubind,get_thread_cpubind);
  DO(cpubind,get_thisproc_last_cpu_location);
  DO(cpubind,get_proc_last_cpu_location);
  DO(cpubind,get_thisthread_last_cpu_location);
  DO(membind,set_thisproc_membind);
  DO(membind,get_thisproc_membind);
  DO(membind,set_proc_membind);
  DO(membind,get_proc_membind);
  DO(membind,set_thisthread_membind);
  DO(membind,get_thisthread_membind);
  DO(membind,set_area_membind);
  DO(membind,get_area_membind);
  DO(membind,alloc_membind);
  DO(membind,firsttouch_membind);
  DO(membind,bind_membind);
  DO(membind,interleave_membind);
  DO(membind,nexttouch_membind);
  DO(membind,migrate_membind);
  DO(membind,get_area_memlocation);

  /* misc.imported_support would be meaningless in the remote importer,
   * but the importer needs to know whether we exported support or not
   * (in case there are no support bit set at all),
   * use a custom/fake field to do so.
   */
  parentstate->new_child(parentstate, &state, "support");
  state.new_prop(&state, "name", "custom.exported_support");
  state.end_object(&state, "support");

#undef DO
}

static void
hwloc__xml_export_memattr_target(hwloc__xml_export_state_t state,
                                 struct hwloc_internal_memattr_s *imattr,
                                 struct hwloc_internal_memattr_target_s *imtg)
{
  struct hwloc__xml_export_state_s vstate;
  char tmp[255];

  if (imattr->flags & HWLOC_MEMATTR_FLAG_NEED_INITIATOR) {
    /* export all initiators */
    unsigned k;
    for(k=0; k<imtg->nr_initiators; k++) {
      struct hwloc_internal_memattr_initiator_s *imi = &imtg->initiators[k];
      state->new_child(state, &vstate, "memattr_value");
      vstate.new_prop(&vstate, "target_obj_type", hwloc_obj_type_string(imtg->type));
      snprintf(tmp, sizeof(tmp), "%llu", (unsigned long long) imtg->gp_index);
      vstate.new_prop(&vstate, "target_obj_gp_index", tmp);
      snprintf(tmp, sizeof(tmp), "%llu", (unsigned long long) imi->value);
      vstate.new_prop(&vstate, "value", tmp);
      switch (imi->initiator.type) {
      case HWLOC_LOCATION_TYPE_OBJECT:
        snprintf(tmp, sizeof(tmp), "%llu", (unsigned long long) imi->initiator.location.object.gp_index);
        vstate.new_prop(&vstate, "initiator_obj_gp_index", tmp);
        vstate.new_prop(&vstate, "initiator_obj_type", hwloc_obj_type_string(imi->initiator.location.object.type));
        break;
      case HWLOC_LOCATION_TYPE_CPUSET: {
        char *setstring;
        hwloc_bitmap_asprintf(&setstring, imi->initiator.location.cpuset);
        if (setstring)
          vstate.new_prop(&vstate, "initiator_cpuset", setstring);
        free(setstring);
        break;
      }
      default:
        assert(0);
      }
      vstate.end_object(&vstate, "memattr_value");
    }
  } else {
    /* just export the global value */
    state->new_child(state, &vstate, "memattr_value");
    vstate.new_prop(&vstate, "target_obj_type", hwloc_obj_type_string(imtg->type));
    snprintf(tmp, sizeof(tmp), "%llu", (unsigned long long) imtg->gp_index);
    vstate.new_prop(&vstate, "target_obj_gp_index", tmp);
    snprintf(tmp, sizeof(tmp), "%llu", (unsigned long long) imtg->noinitiator_value);
    vstate.new_prop(&vstate, "value", tmp);
    vstate.end_object(&vstate, "memattr_value");
  }
}

static void
hwloc__xml_export_memattrs(hwloc__xml_export_state_t state, hwloc_topology_t topology)
{
  unsigned id;
  for(id=0; id<topology->nr_memattrs; id++) {
    struct hwloc_internal_memattr_s *imattr;
    struct hwloc__xml_export_state_s mstate;
    char tmp[255];
    unsigned j;

    if (id == HWLOC_MEMATTR_ID_CAPACITY || id == HWLOC_MEMATTR_ID_LOCALITY)
      /* no need to export virtual memattrs */
      continue;

    imattr = &topology->memattrs[id];
    if ((id == HWLOC_MEMATTR_ID_LATENCY || id == HWLOC_MEMATTR_ID_BANDWIDTH)
        && !imattr->nr_targets)
      /* no need to export target-less attributes for initial attributes, no release support attributes without those definitions */
      continue;

    state->new_child(state, &mstate, "memattr");
    mstate.new_prop(&mstate, "name", imattr->name);
    snprintf(tmp, sizeof(tmp), "%lu", imattr->flags);
    mstate.new_prop(&mstate, "flags", tmp);

    for(j=0; j<imattr->nr_targets; j++)
      hwloc__xml_export_memattr_target(&mstate, imattr, &imattr->targets[j]);

    mstate.end_object(&mstate, "memattr");
  }
}

static void
hwloc__xml_export_cpukinds(hwloc__xml_export_state_t state, hwloc_topology_t topology)
{
  unsigned i;
  for(i=0; i<topology->nr_cpukinds; i++) {
    struct hwloc_internal_cpukind_s *kind = &topology->cpukinds[i];
    struct hwloc__xml_export_state_s cstate;
    char *setstring;
    unsigned j;

    state->new_child(state, &cstate, "cpukind");
    hwloc_bitmap_asprintf(&setstring, kind->cpuset);
    cstate.new_prop(&cstate, "cpuset", setstring);
    free(setstring);
    if (kind->forced_efficiency != HWLOC_CPUKIND_EFFICIENCY_UNKNOWN) {
      char tmp[11];
      snprintf(tmp, sizeof(tmp), "%d", kind->forced_efficiency);
      cstate.new_prop(&cstate, "forced_efficiency", tmp);
    }

    for(j=0; j<kind->nr_infos; j++) {
      char *name = hwloc__xml_export_safestrdup(kind->infos[j].name);
      char *value = hwloc__xml_export_safestrdup(kind->infos[j].value);
      struct hwloc__xml_export_state_s istate;
      cstate.new_child(&cstate, &istate, "info");
      istate.new_prop(&istate, "name", name);
      istate.new_prop(&istate, "value", value);
      istate.end_object(&istate, "info");
      free(name);
      free(value);
    }

    cstate.end_object(&cstate, "cpukind");
  }
}

void
hwloc__xml_export_topology(hwloc__xml_export_state_t state, hwloc_topology_t topology, unsigned long flags)
{
  char *env;
  hwloc_obj_t root = hwloc_get_root_obj(topology);

  if (flags & HWLOC_TOPOLOGY_EXPORT_XML_FLAG_V1) {
    hwloc_obj_t *numanodes, first_numanode;
    unsigned nr_numanodes;

    nr_numanodes = hwloc__xml_v1export_object_list_numanodes(root, &first_numanode, &numanodes);

    if (nr_numanodes) {
      /* we don't use hwloc__xml_v1export_object_with_memory() because we want/can keep root above the numa node */
      struct hwloc__xml_export_state_s rstate, mstate;
      hwloc_obj_t child;
      unsigned i;
      /* export the root */
      state->new_child(state, &rstate, "object");
      hwloc__xml_export_object_contents (&rstate, topology, root, flags);
      /* export first memory child */
      rstate.new_child(&rstate, &mstate, "object");
      hwloc__xml_export_object_contents (&mstate, topology, first_numanode, flags);
      /* then its normal/io/misc children */
      for_each_child(child, root)
	hwloc__xml_v1export_object (&mstate, topology, child, flags);
      for_each_io_child(child, root)
	hwloc__xml_v1export_object (&mstate, topology, child, flags);
      for_each_misc_child(child, root)
	hwloc__xml_v1export_object (&mstate, topology, child, flags);
      /* close first memory child */
      mstate.end_object(&mstate, "object");
      /* now other memory children */
      for(i=1; i<nr_numanodes; i++)
	hwloc__xml_v1export_object (&rstate, topology, numanodes[i], flags);
      /* close the root */
      rstate.end_object(&rstate, "object");
    } else {
      hwloc__xml_v1export_object(state, topology, root, flags);
    }

    free(numanodes);

  } else {
    hwloc__xml_v2export_object (state, topology, root, flags);
    hwloc__xml_v2export_distances (state, topology);
    env = getenv("HWLOC_XML_EXPORT_SUPPORT");
    if (!env || atoi(env))
      hwloc__xml_v2export_support(state, topology);
    hwloc__xml_export_memattrs(state, topology);
    hwloc__xml_export_cpukinds(state, topology);
  }
}

void
hwloc__xml_export_diff(hwloc__xml_export_state_t parentstate, hwloc_topology_diff_t diff)
{
  while (diff) {
    struct hwloc__xml_export_state_s state;
    char tmp[255];

    parentstate->new_child(parentstate, &state, "diff");

    sprintf(tmp, "%d", (int) diff->generic.type);
    state.new_prop(&state, "type", tmp);

    switch (diff->generic.type) {
    case HWLOC_TOPOLOGY_DIFF_OBJ_ATTR:
      sprintf(tmp, "%d", diff->obj_attr.obj_depth);
      state.new_prop(&state, "obj_depth", tmp);
      sprintf(tmp, "%u", diff->obj_attr.obj_index);
      state.new_prop(&state, "obj_index", tmp);

      sprintf(tmp, "%d", (int) diff->obj_attr.diff.generic.type);
      state.new_prop(&state, "obj_attr_type", tmp);

      switch (diff->obj_attr.diff.generic.type) {
      case HWLOC_TOPOLOGY_DIFF_OBJ_ATTR_SIZE:
	sprintf(tmp, "%llu", (unsigned long long) diff->obj_attr.diff.uint64.index);
	state.new_prop(&state, "obj_attr_index", tmp);
	sprintf(tmp, "%llu", (unsigned long long) diff->obj_attr.diff.uint64.oldvalue);
	state.new_prop(&state, "obj_attr_oldvalue", tmp);
	sprintf(tmp, "%llu", (unsigned long long) diff->obj_attr.diff.uint64.newvalue);
	state.new_prop(&state, "obj_attr_newvalue", tmp);
	break;
      case HWLOC_TOPOLOGY_DIFF_OBJ_ATTR_NAME:
      case HWLOC_TOPOLOGY_DIFF_OBJ_ATTR_INFO:
	if (diff->obj_attr.diff.string.name)
	  state.new_prop(&state, "obj_attr_name", diff->obj_attr.diff.string.name);
	state.new_prop(&state, "obj_attr_oldvalue", diff->obj_attr.diff.string.oldvalue);
	state.new_prop(&state, "obj_attr_newvalue", diff->obj_attr.diff.string.newvalue);
	break;
      }

      break;
    default:
      assert(0);
    }
    state.end_object(&state, "diff");

    diff = diff->generic.next;
  }
}

/**********************************
 ********* main XML export ********
 **********************************/

/* this can be the first XML call */
int hwloc_topology_export_xml(hwloc_topology_t topology, const char *filename, unsigned long flags)
{
  hwloc_localeswitch_declare;
  struct hwloc__xml_export_data_s edata;
  int force_nolibxml;
  int ret;

  if (!topology->is_loaded) {
    errno = EINVAL;
    return -1;
  }

  assert(hwloc_nolibxml_callbacks); /* the core called components_init() for the topology */

  if (flags & ~HWLOC_TOPOLOGY_EXPORT_XML_FLAG_V1) {
    errno = EINVAL;
    return -1;
  }

  hwloc_internal_distances_refresh(topology);

  hwloc_localeswitch_init();

  edata.v1_memory_group = NULL;
  if (flags & HWLOC_TOPOLOGY_EXPORT_XML_FLAG_V1)
    /* temporary group to be used during v1 export of memory children */
    edata.v1_memory_group = hwloc_alloc_setup_object(topology, HWLOC_OBJ_GROUP, HWLOC_UNKNOWN_INDEX);

  force_nolibxml = hwloc_nolibxml_export();
retry:
  if (!hwloc_libxml_callbacks || (hwloc_nolibxml_callbacks && force_nolibxml))
    ret = hwloc_nolibxml_callbacks->export_file(topology, &edata, filename, flags);
  else {
    ret = hwloc_libxml_callbacks->export_file(topology, &edata, filename, flags);
    if (ret < 0 && errno == ENOSYS) {
      hwloc_libxml_callbacks = NULL;
      goto retry;
    }
  }

  if (edata.v1_memory_group)
    hwloc_free_unlinked_object(edata.v1_memory_group);

  hwloc_localeswitch_fini();
  return ret;
}

/* this can be the first XML call */
int hwloc_topology_export_xmlbuffer(hwloc_topology_t topology, char **xmlbuffer, int *buflen, unsigned long flags)
{
  hwloc_localeswitch_declare;
  struct hwloc__xml_export_data_s edata;
  int force_nolibxml;
  int ret;

  if (!topology->is_loaded) {
    errno = EINVAL;
    return -1;
  }

  assert(hwloc_nolibxml_callbacks); /* the core called components_init() for the topology */

  if (flags & ~HWLOC_TOPOLOGY_EXPORT_XML_FLAG_V1) {
    errno = EINVAL;
    return -1;
  }

  hwloc_internal_distances_refresh(topology);

  hwloc_localeswitch_init();

  edata.v1_memory_group = NULL;
  if (flags & HWLOC_TOPOLOGY_EXPORT_XML_FLAG_V1)
    /* temporary group to be used during v1 export of memory children */
    edata.v1_memory_group = hwloc_alloc_setup_object(topology, HWLOC_OBJ_GROUP, HWLOC_UNKNOWN_INDEX);

  force_nolibxml = hwloc_nolibxml_export();
retry:
  if (!hwloc_libxml_callbacks || (hwloc_nolibxml_callbacks && force_nolibxml))
    ret = hwloc_nolibxml_callbacks->export_buffer(topology, &edata, xmlbuffer, buflen, flags);
  else {
    ret = hwloc_libxml_callbacks->export_buffer(topology, &edata, xmlbuffer, buflen, flags);
    if (ret < 0 && errno == ENOSYS) {
      hwloc_libxml_callbacks = NULL;
      goto retry;
    }
  }

  if (edata.v1_memory_group)
    hwloc_free_unlinked_object(edata.v1_memory_group);

  hwloc_localeswitch_fini();
  return ret;
}

/* this can be the first XML call */
int
hwloc_topology_diff_export_xml(hwloc_topology_diff_t diff, const char *refname,
			       const char *filename)
{
  hwloc_localeswitch_declare;
  hwloc_topology_diff_t tmpdiff;
  int force_nolibxml;
  int ret;

  tmpdiff = diff;
  while (tmpdiff) {
    if (tmpdiff->generic.type == HWLOC_TOPOLOGY_DIFF_TOO_COMPLEX) {
      errno = EINVAL;
      return -1;
    }
    tmpdiff = tmpdiff->generic.next;
  }

  hwloc_components_init();
  assert(hwloc_nolibxml_callbacks);

  hwloc_localeswitch_init();

  force_nolibxml = hwloc_nolibxml_export();
retry:
  if (!hwloc_libxml_callbacks || (hwloc_nolibxml_callbacks && force_nolibxml))
    ret = hwloc_nolibxml_callbacks->export_diff_file(diff, refname, filename);
  else {
    ret = hwloc_libxml_callbacks->export_diff_file(diff, refname, filename);
    if (ret < 0 && errno == ENOSYS) {
      hwloc_libxml_callbacks = NULL;
      goto retry;
    }
  }

  hwloc_localeswitch_fini();
  hwloc_components_fini();
  return ret;
}

/* this can be the first XML call */
int
hwloc_topology_diff_export_xmlbuffer(hwloc_topology_diff_t diff, const char *refname,
				     char **xmlbuffer, int *buflen)
{
  hwloc_localeswitch_declare;
  hwloc_topology_diff_t tmpdiff;
  int force_nolibxml;
  int ret;

  tmpdiff = diff;
  while (tmpdiff) {
    if (tmpdiff->generic.type == HWLOC_TOPOLOGY_DIFF_TOO_COMPLEX) {
      errno = EINVAL;
      return -1;
    }
    tmpdiff = tmpdiff->generic.next;
  }

  hwloc_components_init();
  assert(hwloc_nolibxml_callbacks);

  hwloc_localeswitch_init();

  force_nolibxml = hwloc_nolibxml_export();
retry:
  if (!hwloc_libxml_callbacks || (hwloc_nolibxml_callbacks && force_nolibxml))
    ret = hwloc_nolibxml_callbacks->export_diff_buffer(diff, refname, xmlbuffer, buflen);
  else {
    ret = hwloc_libxml_callbacks->export_diff_buffer(diff, refname, xmlbuffer, buflen);
    if (ret < 0 && errno == ENOSYS) {
      hwloc_libxml_callbacks = NULL;
      goto retry;
    }
  }

  hwloc_localeswitch_fini();
  hwloc_components_fini();
  return ret;
}

void hwloc_free_xmlbuffer(hwloc_topology_t topology __hwloc_attribute_unused, char *xmlbuffer)
{
  int force_nolibxml;

  assert(hwloc_nolibxml_callbacks); /* the core called components_init() for the topology */

  force_nolibxml = hwloc_nolibxml_export();
  if (!hwloc_libxml_callbacks || (hwloc_nolibxml_callbacks && force_nolibxml))
    hwloc_nolibxml_callbacks->free_buffer(xmlbuffer);
  else
    hwloc_libxml_callbacks->free_buffer(xmlbuffer);
}

void
hwloc_topology_set_userdata_export_callback(hwloc_topology_t topology,
					    void (*export)(void *reserved, struct hwloc_topology *topology, struct hwloc_obj *obj))
{
  topology->userdata_export_cb = export;
}

static void
hwloc__export_obj_userdata(hwloc__xml_export_state_t parentstate, int encoded,
			   const char *name, size_t length, const void *buffer, size_t encoded_length)
{
  struct hwloc__xml_export_state_s state;
  char tmp[255];
  parentstate->new_child(parentstate, &state, "userdata");
  if (name)
    state.new_prop(&state, "name", name);
  sprintf(tmp, "%lu", (unsigned long) length);
  state.new_prop(&state, "length", tmp);
  if (encoded)
    state.new_prop(&state, "encoding", "base64");
  if (encoded_length)
    state.add_content(&state, buffer, encoded ? encoded_length : length);
  state.end_object(&state, "userdata");
}

int
hwloc_export_obj_userdata(void *reserved,
			  struct hwloc_topology *topology, struct hwloc_obj *obj __hwloc_attribute_unused,
			  const char *name, const void *buffer, size_t length)
{
  hwloc__xml_export_state_t state = reserved;

  if (!buffer) {
    errno = EINVAL;
    return -1;
  }

  if ((name && hwloc__xml_export_check_buffer(name, strlen(name)) < 0)
      || hwloc__xml_export_check_buffer(buffer, length) < 0) {
    errno = EINVAL;
    return -1;
  }

  if (topology->userdata_not_decoded) {
    int encoded;
    size_t encoded_length;
    const char *realname;
    assert(name);
    if (!strncmp(name, "base64", 6)) {
      encoded = 1;
      encoded_length = BASE64_ENCODED_LENGTH(length);
    } else {
      assert(!strncmp(name, "normal", 6));
      encoded = 0;
      encoded_length = length;
    }
    if (name[6] == ':')
      realname = name+7;
    else {
      assert(!strcmp(name+6, "-anon"));
      realname = NULL;
    }
    hwloc__export_obj_userdata(state, encoded, realname, length, buffer, encoded_length);

  } else
    hwloc__export_obj_userdata(state, 0, name, length, buffer, length);

  return 0;
}

int
hwloc_export_obj_userdata_base64(void *reserved,
				 struct hwloc_topology *topology __hwloc_attribute_unused, struct hwloc_obj *obj __hwloc_attribute_unused,
				 const char *name, const void *buffer, size_t length)
{
  hwloc__xml_export_state_t state = reserved;
  size_t encoded_length;
  char *encoded_buffer;
  int ret __hwloc_attribute_unused;

  if (!buffer) {
    errno = EINVAL;
    return -1;
  }

  assert(!topology->userdata_not_decoded);

  if (name && hwloc__xml_export_check_buffer(name, strlen(name)) < 0) {
    errno = EINVAL;
    return -1;
  }

  encoded_length = BASE64_ENCODED_LENGTH(length);
  encoded_buffer = malloc(encoded_length+1);
  if (!encoded_buffer) {
    errno = ENOMEM;
    return -1;
  }

  ret = hwloc_encode_to_base64(buffer, length, encoded_buffer, encoded_length+1);
  assert(ret == (int) encoded_length);

  hwloc__export_obj_userdata(state, 1, name, length, encoded_buffer, encoded_length);

  free(encoded_buffer);
  return 0;
}

void
hwloc_topology_set_userdata_import_callback(hwloc_topology_t topology,
					    void (*import)(struct hwloc_topology *topology, struct hwloc_obj *obj, const char *name, const void *buffer, size_t length))
{
  topology->userdata_import_cb = import;
}

/***************************************
 ************ XML component ************
 ***************************************/

static void
hwloc_xml_backend_disable(struct hwloc_backend *backend)
{
  struct hwloc_xml_backend_data_s *data = backend->private_data;
  data->backend_exit(data);
  free(data->msgprefix);
  free(data);
}

static struct hwloc_backend *
hwloc_xml_component_instantiate(struct hwloc_topology *topology,
				struct hwloc_disc_component *component,
				unsigned excluded_phases __hwloc_attribute_unused,
				const void *_data1,
				const void *_data2,
				const void *_data3)
{
  struct hwloc_xml_backend_data_s *data;
  struct hwloc_backend *backend;
  const char *env;
  int force_nolibxml;
  const char * xmlpath = (const char *) _data1;
  const char * xmlbuffer = (const char *) _data2;
  int xmlbuflen = (int)(uintptr_t) _data3;
  const char *local_basename;
  int err;

  assert(hwloc_nolibxml_callbacks); /* the core called components_init() for the component's topology */

  if (!xmlpath && !xmlbuffer) {
    env = getenv("HWLOC_XMLFILE");
    if (env) {
      /* 'xml' was given in HWLOC_COMPONENTS without a filename */
      xmlpath = env;
    } else {
      errno = EINVAL;
      goto out;
    }
  }

  backend = hwloc_backend_alloc(topology, component);
  if (!backend)
    goto out;

  data = malloc(sizeof(*data));
  if (!data) {
    errno = ENOMEM;
    goto out_with_backend;
  }

  backend->private_data = data;
  backend->discover = hwloc_look_xml;
  backend->disable = hwloc_xml_backend_disable;
  backend->is_thissystem = 0;

  if (xmlpath) {
    local_basename = strrchr(xmlpath, '/');
    if (local_basename)
      local_basename++;
    else
      local_basename = xmlpath;
  } else {
    local_basename = "xmlbuffer";
  }
  data->msgprefix = strdup(local_basename);

  force_nolibxml = hwloc_nolibxml_import();
retry:
  if (!hwloc_libxml_callbacks || (hwloc_nolibxml_callbacks && force_nolibxml))
    err = hwloc_nolibxml_callbacks->backend_init(data, xmlpath, xmlbuffer, xmlbuflen);
  else {
    err = hwloc_libxml_callbacks->backend_init(data, xmlpath, xmlbuffer, xmlbuflen);
    if (err < 0 && errno == ENOSYS) {
      hwloc_libxml_callbacks = NULL;
      goto retry;
    }
  }
  if (err < 0)
    goto out_with_data;

  return backend;

 out_with_data:
  free(data->msgprefix);
  free(data);
 out_with_backend:
  free(backend);
 out:
  return NULL;
}

static struct hwloc_disc_component hwloc_xml_disc_component = {
  "xml",
  HWLOC_DISC_PHASE_GLOBAL,
  ~0,
  hwloc_xml_component_instantiate,
  30,
  1,
  NULL
};

const struct hwloc_component hwloc_xml_component = {
  HWLOC_COMPONENT_ABI,
  NULL, NULL,
  HWLOC_COMPONENT_TYPE_DISC,
  0,
  &hwloc_xml_disc_component
};
