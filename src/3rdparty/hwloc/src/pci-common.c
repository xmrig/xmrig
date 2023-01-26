/*
 * Copyright © 2009-2022 Inria.  All rights reserved.
 * See COPYING in top-level directory.
 */

#include "private/autogen/config.h"
#include "hwloc.h"
#include "hwloc/plugins.h"
#include "private/private.h"
#include "private/debug.h"
#include "private/misc.h"

#include <fcntl.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#include <sys/stat.h>

#if defined(HWLOC_WIN_SYS) && !defined(__CYGWIN__)
#include <io.h>
#define open _open
#define read _read
#define close _close
#endif


/**************************************
 * Init/Exit and Forced PCI localities
 */

static void
hwloc_pci_forced_locality_parse_one(struct hwloc_topology *topology,
				    const char *string /* must contain a ' ' */,
				    unsigned *allocated)
{
  unsigned nr = topology->pci_forced_locality_nr;
  unsigned domain, bus_first, bus_last, dummy;
  hwloc_bitmap_t set;
  char *tmp;

  if (sscanf(string, "%x:%x-%x %x", &domain, &bus_first, &bus_last, &dummy) == 4) {
    /* fine */
  } else if (sscanf(string, "%x:%x %x", &domain, &bus_first, &dummy) == 3) {
    bus_last = bus_first;
  } else if (sscanf(string, "%x %x", &domain, &dummy) == 2) {
    bus_first = 0;
    bus_last = 255;
  } else
    return;

  tmp = strchr(string, ' ');
  if (!tmp)
    return;
  tmp++;

  set = hwloc_bitmap_alloc();
  hwloc_bitmap_sscanf(set, tmp);

  if (!*allocated) {
    topology->pci_forced_locality = malloc(sizeof(*topology->pci_forced_locality));
    if (!topology->pci_forced_locality)
      goto out_with_set; /* failed to allocate, ignore this forced locality */
    *allocated = 1;
  } else if (nr >= *allocated) {
    struct hwloc_pci_forced_locality_s *tmplocs;
    tmplocs = realloc(topology->pci_forced_locality,
		      2 * *allocated * sizeof(*topology->pci_forced_locality));
    if (!tmplocs)
      goto out_with_set; /* failed to allocate, ignore this forced locality */
    topology->pci_forced_locality = tmplocs;
    *allocated *= 2;
  }

  topology->pci_forced_locality[nr].domain = domain;
  topology->pci_forced_locality[nr].bus_first = bus_first;
  topology->pci_forced_locality[nr].bus_last = bus_last;
  topology->pci_forced_locality[nr].cpuset = set;
  topology->pci_forced_locality_nr++;
  return;

 out_with_set:
  hwloc_bitmap_free(set);
  return;
}

static void
hwloc_pci_forced_locality_parse(struct hwloc_topology *topology, const char *_env)
{
  char *env = strdup(_env);
  unsigned allocated = 0;
  char *tmp = env;

  while (1) {
    size_t len = strcspn(tmp, ";\r\n");
    char *next = NULL;

    if (tmp[len] != '\0') {
      tmp[len] = '\0';
      if (tmp[len+1] != '\0')
	next = &tmp[len]+1;
    }

    hwloc_pci_forced_locality_parse_one(topology, tmp, &allocated);

    if (next)
      tmp = next;
    else
      break;
  }

  free(env);
}

void
hwloc_pci_discovery_init(struct hwloc_topology *topology)
{
  topology->pci_has_forced_locality = 0;
  topology->pci_forced_locality_nr = 0;
  topology->pci_forced_locality = NULL;

  topology->first_pci_locality = topology->last_pci_locality = NULL;

#define HWLOC_PCI_LOCALITY_QUIRK_CRAY_EX235A (1ULL<<0)
#define HWLOC_PCI_LOCALITY_QUIRK_FAKE (1ULL<<62)
  topology->pci_locality_quirks = (uint64_t) -1;
  /* -1 is unknown, 0 is disabled, >0 is bitmask of enabled quirks.
   * bit 63 should remain unused so that -1 is unaccessible as a bitmask.
   */
}

void
hwloc_pci_discovery_prepare(struct hwloc_topology *topology)
{
  char *env;

  env = getenv("HWLOC_PCI_LOCALITY");
  if (env) {
    int fd;

    topology->pci_has_forced_locality = 1;

    fd = open(env, O_RDONLY);
    if (fd >= 0) {
      struct stat st;
      char *buffer;
      int err = fstat(fd, &st);
      if (!err) {
	if (st.st_size <= 64*1024) { /* random limit large enough to store multiple cpusets for thousands of PUs */
	  buffer = malloc(st.st_size+1);
	  if (buffer && read(fd, buffer, st.st_size) == st.st_size) {
	    buffer[st.st_size] = '\0';
	    hwloc_pci_forced_locality_parse(topology, buffer);
	  }
	  free(buffer);
	} else {
          if (HWLOC_SHOW_CRITICAL_ERRORS())
            fprintf(stderr, "hwloc/pci: Ignoring HWLOC_PCI_LOCALITY file `%s' too large (%lu bytes)\n",
                    env, (unsigned long) st.st_size);
	}
      }
      close(fd);
    } else
      hwloc_pci_forced_locality_parse(topology, env);
  }
}

void
hwloc_pci_discovery_exit(struct hwloc_topology *topology)
{
  struct hwloc_pci_locality_s *cur;
  unsigned i;

  for(i=0; i<topology->pci_forced_locality_nr; i++)
    hwloc_bitmap_free(topology->pci_forced_locality[i].cpuset);
  free(topology->pci_forced_locality);

  cur = topology->first_pci_locality;
  while (cur) {
    struct hwloc_pci_locality_s *next = cur->next;
    hwloc_bitmap_free(cur->cpuset);
    free(cur);
    cur = next;
  }

  hwloc_pci_discovery_init(topology);
}


/******************************
 * Inserting in Tree by Bus ID
 */

#ifdef HWLOC_DEBUG
static void
hwloc_pci_traverse_print_cb(void * cbdata __hwloc_attribute_unused,
			    struct hwloc_obj *pcidev)
{
  char busid[14];
  hwloc_obj_t parent;

  /* indent */
  parent = pcidev->parent;
  while (parent) {
    hwloc_debug("%s", "  ");
    parent = parent->parent;
  }

  snprintf(busid, sizeof(busid), "%04x:%02x:%02x.%01x",
           pcidev->attr->pcidev.domain, pcidev->attr->pcidev.bus, pcidev->attr->pcidev.dev, pcidev->attr->pcidev.func);

  if (pcidev->type == HWLOC_OBJ_BRIDGE) {
    if (pcidev->attr->bridge.upstream_type == HWLOC_OBJ_BRIDGE_HOST)
      hwloc_debug("HostBridge");
    else
      hwloc_debug("%s Bridge [%04x:%04x]", busid,
		  pcidev->attr->pcidev.vendor_id, pcidev->attr->pcidev.device_id);
    if (pcidev->attr->bridge.downstream_type == HWLOC_OBJ_BRIDGE_PCI)
      hwloc_debug(" to %04x:[%02x:%02x]\n",
                  pcidev->attr->bridge.downstream.pci.domain, pcidev->attr->bridge.downstream.pci.secondary_bus, pcidev->attr->bridge.downstream.pci.subordinate_bus);
    else
      assert(0);
  } else
    hwloc_debug("%s Device [%04x:%04x (%04x:%04x) rev=%02x class=%04x]\n", busid,
		pcidev->attr->pcidev.vendor_id, pcidev->attr->pcidev.device_id,
		pcidev->attr->pcidev.subvendor_id, pcidev->attr->pcidev.subdevice_id,
		pcidev->attr->pcidev.revision, pcidev->attr->pcidev.class_id);
}

static void
hwloc_pci_traverse(void * cbdata, struct hwloc_obj *tree,
		   void (*cb)(void * cbdata, struct hwloc_obj *))
{
  hwloc_obj_t child;
  cb(cbdata, tree);
  for_each_io_child(child, tree) {
    if (child->type == HWLOC_OBJ_BRIDGE)
      hwloc_pci_traverse(cbdata, child, cb);
  }
}
#endif /* HWLOC_DEBUG */

enum hwloc_pci_busid_comparison_e {
  HWLOC_PCI_BUSID_LOWER,
  HWLOC_PCI_BUSID_HIGHER,
  HWLOC_PCI_BUSID_INCLUDED,
  HWLOC_PCI_BUSID_SUPERSET,
  HWLOC_PCI_BUSID_EQUAL
};

static enum hwloc_pci_busid_comparison_e
hwloc_pci_compare_busids(struct hwloc_obj *a, struct hwloc_obj *b)
{
#ifdef HWLOC_DEBUG
  if (a->type == HWLOC_OBJ_BRIDGE)
    assert(a->attr->bridge.upstream_type == HWLOC_OBJ_BRIDGE_PCI);
  if (b->type == HWLOC_OBJ_BRIDGE)
    assert(b->attr->bridge.upstream_type == HWLOC_OBJ_BRIDGE_PCI);
#endif

  if (a->attr->pcidev.domain < b->attr->pcidev.domain)
    return HWLOC_PCI_BUSID_LOWER;
  if (a->attr->pcidev.domain > b->attr->pcidev.domain)
    return HWLOC_PCI_BUSID_HIGHER;

  if (a->type == HWLOC_OBJ_BRIDGE && a->attr->bridge.downstream_type == HWLOC_OBJ_BRIDGE_PCI
      && b->attr->pcidev.bus >= a->attr->bridge.downstream.pci.secondary_bus
      && b->attr->pcidev.bus <= a->attr->bridge.downstream.pci.subordinate_bus)
    return HWLOC_PCI_BUSID_SUPERSET;
  if (b->type == HWLOC_OBJ_BRIDGE && b->attr->bridge.downstream_type == HWLOC_OBJ_BRIDGE_PCI
      && a->attr->pcidev.bus >= b->attr->bridge.downstream.pci.secondary_bus
      && a->attr->pcidev.bus <= b->attr->bridge.downstream.pci.subordinate_bus)
    return HWLOC_PCI_BUSID_INCLUDED;

  if (a->attr->pcidev.bus < b->attr->pcidev.bus)
    return HWLOC_PCI_BUSID_LOWER;
  if (a->attr->pcidev.bus > b->attr->pcidev.bus)
    return HWLOC_PCI_BUSID_HIGHER;

  if (a->attr->pcidev.dev < b->attr->pcidev.dev)
    return HWLOC_PCI_BUSID_LOWER;
  if (a->attr->pcidev.dev > b->attr->pcidev.dev)
    return HWLOC_PCI_BUSID_HIGHER;

  if (a->attr->pcidev.func < b->attr->pcidev.func)
    return HWLOC_PCI_BUSID_LOWER;
  if (a->attr->pcidev.func > b->attr->pcidev.func)
    return HWLOC_PCI_BUSID_HIGHER;

  /* Should never reach here. */
  return HWLOC_PCI_BUSID_EQUAL;
}

static void
hwloc_pci_add_object(struct hwloc_obj *parent, struct hwloc_obj **parent_io_first_child_p, struct hwloc_obj *new)
{
  struct hwloc_obj **curp, **childp;

  curp = parent_io_first_child_p;
  while (*curp) {
    enum hwloc_pci_busid_comparison_e comp = hwloc_pci_compare_busids(new, *curp);
    switch (comp) {
    case HWLOC_PCI_BUSID_HIGHER:
      /* go further */
      curp = &(*curp)->next_sibling;
      continue;
    case HWLOC_PCI_BUSID_INCLUDED:
      /* insert new below current bridge */
      hwloc_pci_add_object(*curp, &(*curp)->io_first_child, new);
      return;
    case HWLOC_PCI_BUSID_LOWER:
    case HWLOC_PCI_BUSID_SUPERSET: {
      /* insert new before current */
      new->next_sibling = *curp;
      *curp = new;
      new->parent = parent;
      if (new->type == HWLOC_OBJ_BRIDGE && new->attr->bridge.downstream_type == HWLOC_OBJ_BRIDGE_PCI) {
	/* look at remaining siblings and move some below new */
	childp = &new->io_first_child;
	curp = &new->next_sibling;
	while (*curp) {
	  hwloc_obj_t cur = *curp;
	  if (hwloc_pci_compare_busids(new, cur) == HWLOC_PCI_BUSID_LOWER) {
	    /* this sibling remains under root, after new. */
	    if (cur->attr->pcidev.domain > new->attr->pcidev.domain
		|| cur->attr->pcidev.bus > new->attr->bridge.downstream.pci.subordinate_bus)
	      /* this sibling is even above new's subordinate bus, no other sibling could go below new */
	      return;
	    curp = &cur->next_sibling;
	  } else {
	    /* this sibling goes under new */
	    *childp = cur;
	    *curp = cur->next_sibling;
	    (*childp)->parent = new;
	    (*childp)->next_sibling = NULL;
	    childp = &(*childp)->next_sibling;
	  }
	}
      }
      return;
    }
    case HWLOC_PCI_BUSID_EQUAL: {
      static int reported = 0;
      if (!reported && HWLOC_SHOW_CRITICAL_ERRORS()) {
        fprintf(stderr, "*********************************************************\n");
        fprintf(stderr, "* hwloc %s received invalid PCI information.\n", HWLOC_VERSION);
        fprintf(stderr, "*\n");
        fprintf(stderr, "* Trying to insert PCI object %04x:%02x:%02x.%01x at %04x:%02x:%02x.%01x\n",
                new->attr->pcidev.domain, new->attr->pcidev.bus, new->attr->pcidev.dev, new->attr->pcidev.func,
                (*curp)->attr->pcidev.domain, (*curp)->attr->pcidev.bus, (*curp)->attr->pcidev.dev, (*curp)->attr->pcidev.func);
        fprintf(stderr, "*\n");
        fprintf(stderr, "* hwloc will now ignore this object and continue.\n");
        fprintf(stderr, "*********************************************************\n");
        reported = 1;
      }
      hwloc_free_unlinked_object(new);
      return;
    }
    }
  }
  /* add to the end of the list if higher than everybody */
  new->parent = parent;
  new->next_sibling = NULL;
  *curp = new;
}

void
hwloc_pcidisc_tree_insert_by_busid(struct hwloc_obj **treep,
				   struct hwloc_obj *obj)
{
  hwloc_pci_add_object(NULL /* no parent on top of tree */, treep, obj);
}


/**********************
 * Attaching PCI Trees
 */

static struct hwloc_obj *
hwloc_pcidisc_add_hostbridges(struct hwloc_topology *topology,
			      struct hwloc_obj *old_tree)
{
  struct hwloc_obj * new = NULL, **newp = &new;

  /*
   * tree points to all objects connected to any upstream bus in the machine.
   * We now create one real hostbridge object per upstream bus.
   * It's not actually a PCI device so we have to create it.
   */
  while (old_tree) {
    /* start a new host bridge */
    struct hwloc_obj *hostbridge;
    struct hwloc_obj **dstnextp;
    struct hwloc_obj **srcnextp;
    struct hwloc_obj *child;
    unsigned current_domain;
    unsigned char current_bus;
    unsigned char current_subordinate;

    hostbridge = hwloc_alloc_setup_object(topology, HWLOC_OBJ_BRIDGE, HWLOC_UNKNOWN_INDEX);
    if (!hostbridge) {
      /* just queue remaining things without hostbridges and return */
      *newp = old_tree;
      return new;
    }
    dstnextp = &hostbridge->io_first_child;

    srcnextp = &old_tree;
    child = *srcnextp;
    current_domain = child->attr->pcidev.domain;
    current_bus = child->attr->pcidev.bus;
    current_subordinate = current_bus;

    hwloc_debug("Adding new PCI hostbridge %04x:%02x\n", current_domain, current_bus);

  next_child:
    /* remove next child from tree */
    *srcnextp = child->next_sibling;
    /* append it to hostbridge */
    *dstnextp = child;
    child->parent = hostbridge;
    child->next_sibling = NULL;
    dstnextp = &child->next_sibling;

    /* compute hostbridge secondary/subordinate buses */
    if (child->type == HWLOC_OBJ_BRIDGE && child->attr->bridge.downstream_type == HWLOC_OBJ_BRIDGE_PCI
	&& child->attr->bridge.downstream.pci.subordinate_bus > current_subordinate)
      current_subordinate = child->attr->bridge.downstream.pci.subordinate_bus;

    /* use next child if it has the same domains/bus */
    child = *srcnextp;
    if (child
	&& child->attr->pcidev.domain == current_domain
	&& child->attr->pcidev.bus == current_bus)
      goto next_child;

    /* finish setting up this hostbridge */
    hostbridge->attr->bridge.upstream_type = HWLOC_OBJ_BRIDGE_HOST;
    hostbridge->attr->bridge.downstream_type = HWLOC_OBJ_BRIDGE_PCI;
    hostbridge->attr->bridge.downstream.pci.domain = current_domain;
    hostbridge->attr->bridge.downstream.pci.secondary_bus = current_bus;
    hostbridge->attr->bridge.downstream.pci.subordinate_bus = current_subordinate;
    hwloc_debug("  new PCI hostbridge covers %04x:[%02x-%02x]\n",
		current_domain, current_bus, current_subordinate);

    *newp = hostbridge;
    newp = &hostbridge->next_sibling;
  }

  return new;
}

/* return 1 if a quirk was applied */
static int
hwloc__pci_find_busid_parent_quirk(struct hwloc_topology *topology,
                                   struct hwloc_pcidev_attr_s *busid,
                                   hwloc_cpuset_t cpuset)
{
  if (topology->pci_locality_quirks == (uint64_t)-1 /* unknown */) {
    const char *dmi_board_name, *env;

    /* first invokation, detect which quirks are needed */
    topology->pci_locality_quirks = 0; /* no quirk yet */

    dmi_board_name = hwloc_obj_get_info_by_name(hwloc_get_root_obj(topology), "DMIBoardName");
    if (dmi_board_name && !strcmp(dmi_board_name, "HPE CRAY EX235A")) {
      hwloc_debug("enabling for PCI locality quirk for HPE Cray EX235A\n");
      topology->pci_locality_quirks |= HWLOC_PCI_LOCALITY_QUIRK_CRAY_EX235A;
    }

    env = getenv("HWLOC_PCI_LOCALITY_QUIRK_FAKE");
    if (env && atoi(env)) {
      hwloc_debug("enabling for PCI locality fake quirk (attaching everything to last PU)\n");
      topology->pci_locality_quirks |= HWLOC_PCI_LOCALITY_QUIRK_FAKE;
    }
  }

  if (topology->pci_locality_quirks & HWLOC_PCI_LOCALITY_QUIRK_FAKE) {
    unsigned last = hwloc_bitmap_last(hwloc_topology_get_topology_cpuset(topology));
    hwloc_bitmap_set(cpuset, last);
    return 1;
  }

  if (topology->pci_locality_quirks & HWLOC_PCI_LOCALITY_QUIRK_CRAY_EX235A) {
    /* AMD Trento has xGMI ports connected to individual CCDs (8 cores + L3)
     * instead of NUMA nodes (pairs of CCDs within Trento) as is usual in AMD EPYC CPUs.
     * This is not described by the ACPI tables, hence we need to manually hardwire
     * the xGMI locality for the (currently single) server that currently uses that CPU.
     * It's not clear if ACPI tables can/will ever be fixed (would require one initiator
     * proximity domain per CCD), or if Linux can/will work around the issue.
     */
    if (busid->domain == 0) {
      if (busid->bus >= 0xd0 && busid->bus <= 0xd1) {
        hwloc_bitmap_set_range(cpuset, 0, 7);
        hwloc_bitmap_set_range(cpuset, 64, 71);
        return 1;
      }
      if (busid->bus >= 0xd4 && busid->bus <= 0xd6) {
        hwloc_bitmap_set_range(cpuset, 8, 15);
        hwloc_bitmap_set_range(cpuset, 72, 79);
        return 1;
      }
      if (busid->bus >= 0xc8 && busid->bus <= 0xc9) {
        hwloc_bitmap_set_range(cpuset, 16, 23);
        hwloc_bitmap_set_range(cpuset, 80, 87);
        return 1;
      }
      if (busid->bus >= 0xcc && busid->bus <= 0xce) {
        hwloc_bitmap_set_range(cpuset, 24, 31);
        hwloc_bitmap_set_range(cpuset, 88, 95);
        return 1;
      }
      if (busid->bus >= 0xd8 && busid->bus <= 0xd9) {
        hwloc_bitmap_set_range(cpuset, 32, 39);
        hwloc_bitmap_set_range(cpuset, 96, 103);
        return 1;
      }
      if (busid->bus >= 0xdc && busid->bus <= 0xde) {
        hwloc_bitmap_set_range(cpuset, 40, 47);
        hwloc_bitmap_set_range(cpuset, 104, 111);
        return 1;
      }
      if (busid->bus >= 0xc0 && busid->bus <= 0xc1) {
        hwloc_bitmap_set_range(cpuset, 48, 55);
        hwloc_bitmap_set_range(cpuset, 112, 119);
        return 1;
      }
      if (busid->bus >= 0xc4 && busid->bus <= 0xc6) {
        hwloc_bitmap_set_range(cpuset, 56, 63);
        hwloc_bitmap_set_range(cpuset, 120, 127);
        return 1;
      }
    }
  }

  return 0;
}

static struct hwloc_obj *
hwloc__pci_find_busid_parent(struct hwloc_topology *topology, struct hwloc_pcidev_attr_s *busid)
{
  hwloc_bitmap_t cpuset = hwloc_bitmap_alloc();
  hwloc_obj_t parent;
  int forced = 0;
  int noquirks = 0, got_quirked = 0;
  unsigned i;
  int err;

  hwloc_debug("Looking for parent of PCI busid %04x:%02x:%02x.%01x\n",
	      busid->domain, busid->bus, busid->dev, busid->func);

  /* try to match a forced locality */
  if (topology->pci_has_forced_locality) {
    for(i=0; i<topology->pci_forced_locality_nr; i++) {
      if (busid->domain == topology->pci_forced_locality[i].domain
	  && busid->bus >= topology->pci_forced_locality[i].bus_first
	  && busid->bus <= topology->pci_forced_locality[i].bus_last) {
	hwloc_bitmap_copy(cpuset, topology->pci_forced_locality[i].cpuset);
	forced = 1;
	break;
      }
    }
    /* if pci locality was forced, even empty, don't let quirks change what the OS reports */
    noquirks = 1;
  }

  /* deprecated force locality variables */
  if (!forced) {
    const char *env;
    char envname[256];
    /* override the cpuset with the environment if given */
    snprintf(envname, sizeof(envname), "HWLOC_PCI_%04x_%02x_LOCALCPUS",
	     busid->domain, busid->bus);
    env = getenv(envname);
    if (env) {
      static int reported = 0;
      if (!topology->pci_has_forced_locality && !reported) {
        if (HWLOC_SHOW_ALL_ERRORS())
          fprintf(stderr, "hwloc/pci: Environment variable %s is deprecated, please use HWLOC_PCI_LOCALITY instead.\n", env);
	reported = 1;
      }
      if (*env) {
	/* force the cpuset */
	hwloc_debug("Overriding PCI locality using %s in the environment\n", envname);
	hwloc_bitmap_sscanf(cpuset, env);
	forced = 1;
      }
      /* if env exists, even empty, don't let quirks change what the OS reports */
      noquirks = 1;
    }
  }

  if (!forced && !noquirks && topology->pci_locality_quirks /* either quirks are unknown yet, or some are enabled */) {
    err = hwloc__pci_find_busid_parent_quirk(topology, busid, cpuset);
    if (err > 0)
      got_quirked = 1;
  }

  if (!forced && !got_quirked) {
    /* get the cpuset by asking the backend that provides the relevant hook, if any. */
    struct hwloc_backend *backend = topology->get_pci_busid_cpuset_backend;
    if (backend)
      err = backend->get_pci_busid_cpuset(backend, busid, cpuset);
    else
      err = -1;
    if (err < 0)
      /* if we got nothing, assume this PCI bus is attached to the top of hierarchy */
      hwloc_bitmap_copy(cpuset, hwloc_topology_get_topology_cpuset(topology));
  }

  hwloc_debug_bitmap("  will attach PCI bus to cpuset %s\n", cpuset);

  parent = hwloc_find_insert_io_parent_by_complete_cpuset(topology, cpuset);
  if (!parent) {
    /* Fallback to root */
    parent = hwloc_get_root_obj(topology);
  }

  hwloc_bitmap_free(cpuset);
  return parent;
}

int
hwloc_pcidisc_tree_attach(struct hwloc_topology *topology, struct hwloc_obj *tree)
{
  enum hwloc_type_filter_e bfilter;

  if (!tree)
    /* found nothing, exit */
    return 0;

#ifdef HWLOC_DEBUG
  hwloc_debug("%s", "\nPCI hierarchy:\n");
  hwloc_pci_traverse(NULL, tree, hwloc_pci_traverse_print_cb);
  hwloc_debug("%s", "\n");
#endif

  bfilter = topology->type_filter[HWLOC_OBJ_BRIDGE];
  if (bfilter != HWLOC_TYPE_FILTER_KEEP_NONE) {
    tree = hwloc_pcidisc_add_hostbridges(topology, tree);
  }

  while (tree) {
    struct hwloc_obj *obj, *pciobj;
    struct hwloc_obj *parent;
    struct hwloc_pci_locality_s *loc;
    unsigned domain, bus_min, bus_max;

    obj = tree;

    /* hostbridges don't have a PCI busid for looking up locality, use their first child */
    if (obj->type == HWLOC_OBJ_BRIDGE && obj->attr->bridge.upstream_type == HWLOC_OBJ_BRIDGE_HOST)
      pciobj = obj->io_first_child;
    else
      pciobj = obj;
    /* now we have a pci device or a pci bridge */
    assert(pciobj->type == HWLOC_OBJ_PCI_DEVICE
	   || (pciobj->type == HWLOC_OBJ_BRIDGE && pciobj->attr->bridge.upstream_type == HWLOC_OBJ_BRIDGE_PCI));

    if (obj->type == HWLOC_OBJ_BRIDGE && obj->attr->bridge.downstream_type == HWLOC_OBJ_BRIDGE_PCI) {
      domain = obj->attr->bridge.downstream.pci.domain;
      bus_min = obj->attr->bridge.downstream.pci.secondary_bus;
      bus_max = obj->attr->bridge.downstream.pci.subordinate_bus;
    } else {
      domain = pciobj->attr->pcidev.domain;
      bus_min = pciobj->attr->pcidev.bus;
      bus_max = pciobj->attr->pcidev.bus;
    }

    /* find where to attach that PCI bus */
    parent = hwloc__pci_find_busid_parent(topology, &pciobj->attr->pcidev);

    /* reuse the previous locality if possible */
    if (topology->last_pci_locality
	&& parent == topology->last_pci_locality->parent
	&& domain == topology->last_pci_locality->domain
	&& (bus_min == topology->last_pci_locality->bus_max
	    || bus_min == topology->last_pci_locality->bus_max+1)) {
      hwloc_debug("  Reusing PCI locality up to bus %04x:%02x\n",
		  domain, bus_max);
      topology->last_pci_locality->bus_max = bus_max;
      goto done;
    }

    loc = malloc(sizeof(*loc));
    if (!loc) {
      /* fallback to attaching to root */
      parent = hwloc_get_root_obj(topology);
      goto done;
    }

    loc->domain = domain;
    loc->bus_min = bus_min;
    loc->bus_max = bus_max;
    loc->parent = parent;
    loc->cpuset = hwloc_bitmap_dup(parent->cpuset);
    if (!loc->cpuset) {
      /* fallback to attaching to root */
      free(loc);
      parent = hwloc_get_root_obj(topology);
      goto done;
    }

    hwloc_debug("Adding PCI locality %s P#%u for bus %04x:[%02x:%02x]\n",
		hwloc_obj_type_string(parent->type), parent->os_index, loc->domain, loc->bus_min, loc->bus_max);
    if (topology->last_pci_locality) {
      loc->prev = topology->last_pci_locality;
      loc->next = NULL;
      topology->last_pci_locality->next = loc;
      topology->last_pci_locality = loc;
    } else {
      loc->prev = NULL;
      loc->next = NULL;
      topology->first_pci_locality = loc;
      topology->last_pci_locality = loc;
    }

  done:
    /* dequeue this object */
    tree = obj->next_sibling;
    obj->next_sibling = NULL;
    hwloc_insert_object_by_parent(topology, parent, obj);
  }

  return 0;
}


/*********************************
 * Finding PCI objects or parents
 */

struct hwloc_obj *
hwloc_pci_find_parent_by_busid(struct hwloc_topology *topology,
			       unsigned domain, unsigned bus, unsigned dev, unsigned func)
{
  struct hwloc_pcidev_attr_s busid;
  hwloc_obj_t parent;

  /* try to find that exact busid */
  parent = hwloc_pci_find_by_busid(topology, domain, bus, dev, func);
  if (parent)
    return parent;

  /* try to find the locality of that bus instead */
  busid.domain = domain;
  busid.bus = bus;
  busid.dev = dev;
  busid.func = func;
  return hwloc__pci_find_busid_parent(topology, &busid);
}

/* return the smallest object that contains the desired busid */
static struct hwloc_obj *
hwloc__pci_find_by_busid(hwloc_obj_t parent,
			 unsigned domain, unsigned bus, unsigned dev, unsigned func)
{
  hwloc_obj_t child;

  for_each_io_child(child, parent) {
    if (child->type == HWLOC_OBJ_PCI_DEVICE
	|| (child->type == HWLOC_OBJ_BRIDGE
	    && child->attr->bridge.upstream_type == HWLOC_OBJ_BRIDGE_PCI)) {
      if (child->attr->pcidev.domain == domain
	  && child->attr->pcidev.bus == bus
	  && child->attr->pcidev.dev == dev
	  && child->attr->pcidev.func == func)
	/* that's the right bus id */
	return child;
      if (child->attr->pcidev.domain > domain
	  || (child->attr->pcidev.domain == domain
	      && child->attr->pcidev.bus > bus))
	/* bus id too high, won't find anything later, return parent */
	return parent;
      if (child->type == HWLOC_OBJ_BRIDGE
	  && child->attr->bridge.downstream_type == HWLOC_OBJ_BRIDGE_PCI
	  && child->attr->bridge.downstream.pci.domain == domain
	  && child->attr->bridge.downstream.pci.secondary_bus <= bus
	  && child->attr->bridge.downstream.pci.subordinate_bus >= bus)
	/* not the right bus id, but it's included in the bus below that bridge */
	return hwloc__pci_find_by_busid(child, domain, bus, dev, func);

    } else if (child->type == HWLOC_OBJ_BRIDGE
	       && child->attr->bridge.upstream_type != HWLOC_OBJ_BRIDGE_PCI
	       && child->attr->bridge.downstream_type == HWLOC_OBJ_BRIDGE_PCI
	       /* non-PCI to PCI bridge, just look at the subordinate bus */
	       && child->attr->bridge.downstream.pci.domain == domain
	       && child->attr->bridge.downstream.pci.secondary_bus <= bus
	       && child->attr->bridge.downstream.pci.subordinate_bus >= bus) {
      /* contains our bus, recurse */
      return hwloc__pci_find_by_busid(child, domain, bus, dev, func);
    }
  }
  /* didn't find anything, return parent */
  return parent;
}

struct hwloc_obj *
hwloc_pci_find_by_busid(struct hwloc_topology *topology,
			unsigned domain, unsigned bus, unsigned dev, unsigned func)
{
  struct hwloc_pci_locality_s *loc;
  hwloc_obj_t root = hwloc_get_root_obj(topology);
  hwloc_obj_t parent = NULL;

  hwloc_debug("pcidisc looking for bus id %04x:%02x:%02x.%01x\n", domain, bus, dev, func);
  loc = topology->first_pci_locality;
  while (loc) {
    if (loc->domain == domain && loc->bus_min <= bus && loc->bus_max >= bus) {
      parent = loc->parent;
      assert(parent);
      hwloc_debug("  found pci locality for %04x:[%02x:%02x]\n",
		  loc->domain, loc->bus_min, loc->bus_max);
      break;
    }
    loc = loc->next;
  }
  /* if we failed to insert localities, look at root too */
  if (!parent)
    parent = root;

  hwloc_debug("  looking for bus %04x:%02x:%02x.%01x below %s P#%u\n",
	      domain, bus, dev, func,
	      hwloc_obj_type_string(parent->type), parent->os_index);
  parent = hwloc__pci_find_by_busid(parent, domain, bus, dev, func);
  if (parent == root) {
    hwloc_debug("  found nothing better than root object, ignoring\n");
    return NULL;
  } else {
    if (parent->type == HWLOC_OBJ_PCI_DEVICE
	|| (parent->type == HWLOC_OBJ_BRIDGE && parent->attr->bridge.upstream_type == HWLOC_OBJ_BRIDGE_PCI))
      hwloc_debug("  found busid %04x:%02x:%02x.%01x\n",
		  parent->attr->pcidev.domain, parent->attr->pcidev.bus,
		  parent->attr->pcidev.dev, parent->attr->pcidev.func);
    else
      hwloc_debug("  found parent %s P#%u\n",
		  hwloc_obj_type_string(parent->type), parent->os_index);
    return parent;
  }
}


/*******************************
 * Parsing the PCI Config Space
 */

#define HWLOC_PCI_STATUS 0x06
#define HWLOC_PCI_STATUS_CAP_LIST 0x10
#define HWLOC_PCI_CAPABILITY_LIST 0x34
#define HWLOC_PCI_CAP_LIST_ID 0
#define HWLOC_PCI_CAP_LIST_NEXT 1

unsigned
hwloc_pcidisc_find_cap(const unsigned char *config, unsigned cap)
{
  unsigned char seen[256] = { 0 };
  unsigned char ptr; /* unsigned char to make sure we stay within the 256-byte config space */

  if (!(config[HWLOC_PCI_STATUS] & HWLOC_PCI_STATUS_CAP_LIST))
    return 0;

  for (ptr = config[HWLOC_PCI_CAPABILITY_LIST] & ~3;
       ptr; /* exit if next is 0 */
       ptr = config[ptr + HWLOC_PCI_CAP_LIST_NEXT] & ~3) {
    unsigned char id;

    /* Looped around! */
    if (seen[ptr])
      break;
    seen[ptr] = 1;

    id = config[ptr + HWLOC_PCI_CAP_LIST_ID];
    if (id == cap)
      return ptr;
    if (id == 0xff) /* exit if id is 0 or 0xff */
      break;
  }
  return 0;
}

#define HWLOC_PCI_EXP_LNKSTA 0x12
#define HWLOC_PCI_EXP_LNKSTA_SPEED 0x000f
#define HWLOC_PCI_EXP_LNKSTA_WIDTH 0x03f0

int
hwloc_pcidisc_find_linkspeed(const unsigned char *config,
			     unsigned offset, float *linkspeed)
{
  unsigned linksta, speed, width;
  float lanespeed;

  memcpy(&linksta, &config[offset + HWLOC_PCI_EXP_LNKSTA], 4);
  speed = linksta & HWLOC_PCI_EXP_LNKSTA_SPEED; /* PCIe generation */
  width = (linksta & HWLOC_PCI_EXP_LNKSTA_WIDTH) >> 4; /* how many lanes */
  /*
   * These are single-direction bandwidths only.
   *
   * Gen1 used NRZ with 8/10 encoding.
   * PCIe Gen1 = 2.5GT/s signal-rate per lane x 8/10    =  0.25GB/s data-rate per lane
   * PCIe Gen2 = 5  GT/s signal-rate per lane x 8/10    =  0.5 GB/s data-rate per lane
   * Gen3 switched to NRZ with 128/130 encoding.
   * PCIe Gen3 = 8  GT/s signal-rate per lane x 128/130 =  1   GB/s data-rate per lane
   * PCIe Gen4 = 16 GT/s signal-rate per lane x 128/130 =  2   GB/s data-rate per lane
   * PCIe Gen5 = 32 GT/s signal-rate per lane x 128/130 =  4   GB/s data-rate per lane
   * Gen6 switched to PAM with with 242/256 FLIT (242B payload protected by 8B CRC + 6B FEC).
   * PCIe Gen6 = 64 GT/s signal-rate per lane x 242/256 =  8   GB/s data-rate per lane
   * PCIe Gen7 = 128GT/s signal-rate per lane x 242/256 = 16   GB/s data-rate per lane
   */

  /* lanespeed in Gbit/s */
  if (speed <= 2)
    lanespeed = 2.5f * speed * 0.8f;
  else if (speed <= 5)
    lanespeed = 8.0f * (1<<(speed-3)) * 128/130;
  else
    lanespeed = 8.0f * (1<<(speed-3)) * 242/256; /* assume Gen8 will be 256 GT/s and so on */

  /* linkspeed in GB/s */
  *linkspeed = lanespeed * width / 8;
  return 0;
}

#define HWLOC_PCI_HEADER_TYPE 0x0e
#define HWLOC_PCI_HEADER_TYPE_BRIDGE 1
#define HWLOC_PCI_CLASS_BRIDGE_PCI 0x0604

hwloc_obj_type_t
hwloc_pcidisc_check_bridge_type(unsigned device_class, const unsigned char *config)
{
  unsigned char headertype;

  if (device_class != HWLOC_PCI_CLASS_BRIDGE_PCI)
    return HWLOC_OBJ_PCI_DEVICE;

  headertype = config[HWLOC_PCI_HEADER_TYPE] & 0x7f;
  return (headertype == HWLOC_PCI_HEADER_TYPE_BRIDGE)
    ? HWLOC_OBJ_BRIDGE : HWLOC_OBJ_PCI_DEVICE;
}

#define HWLOC_PCI_PRIMARY_BUS 0x18
#define HWLOC_PCI_SECONDARY_BUS 0x19
#define HWLOC_PCI_SUBORDINATE_BUS 0x1a

int
hwloc_pcidisc_find_bridge_buses(unsigned domain, unsigned bus, unsigned dev, unsigned func,
				unsigned *secondary_busp, unsigned *subordinate_busp,
				const unsigned char *config)
{
  unsigned secondary_bus, subordinate_bus;

  if (config[HWLOC_PCI_PRIMARY_BUS] != bus) {
    /* Sometimes the config space contains 00 instead of the actual primary bus number.
     * Always trust the bus ID because it was built by the system which has more information
     * to workaround such problems (e.g. ACPI information about PCI parent/children).
     */
    hwloc_debug("  %04x:%02x:%02x.%01x bridge with (ignored) invalid PCI_PRIMARY_BUS %02x\n",
		domain, bus, dev, func, config[HWLOC_PCI_PRIMARY_BUS]);
  }

  secondary_bus = config[HWLOC_PCI_SECONDARY_BUS];
  subordinate_bus = config[HWLOC_PCI_SUBORDINATE_BUS];

  if (secondary_bus <= bus
      || subordinate_bus <= bus
      || secondary_bus > subordinate_bus) {
    /* This should catch most cases of invalid bridge information
     * (e.g. 00 for secondary and subordinate).
     * Ideally we would also check that [secondary-subordinate] is included
     * in the parent bridge [secondary+1:subordinate]. But that's hard to do
     * because objects may be discovered out of order (especially in the fsroot case).
     */
    hwloc_debug("  %04x:%02x:%02x.%01x bridge has invalid secondary-subordinate buses [%02x-%02x]\n",
		domain, bus, dev, func,
		secondary_bus, subordinate_bus);
    return -1;
  }

  *secondary_busp = secondary_bus;
  *subordinate_busp = subordinate_bus;
  return 0;
}


/****************
 * Class Strings
 */

const char *
hwloc_pci_class_string(unsigned short class_id)
{
  /* See https://pci-ids.ucw.cz/read/PD/ */
  switch ((class_id & 0xff00) >> 8) {
    case 0x00:
      switch (class_id) {
	case 0x0001: return "VGA";
      }
      break;
    case 0x01:
      switch (class_id) {
	case 0x0100: return "SCSI";
	case 0x0101: return "IDE";
	case 0x0102: return "Floppy";
	case 0x0103: return "IPI";
	case 0x0104: return "RAID";
	case 0x0105: return "ATA";
	case 0x0106: return "SATA";
	case 0x0107: return "SAS";
	case 0x0108: return "NVMExp";
      }
      return "Storage";
    case 0x02:
      switch (class_id) {
	case 0x0200: return "Ethernet";
	case 0x0201: return "TokenRing";
	case 0x0202: return "FDDI";
	case 0x0203: return "ATM";
	case 0x0204: return "ISDN";
	case 0x0205: return "WorldFip";
	case 0x0206: return "PICMG";
	case 0x0207: return "InfiniBand";
	case 0x0208: return "Fabric";
      }
      return "Network";
    case 0x03:
      switch (class_id) {
	case 0x0300: return "VGA";
	case 0x0301: return "XGA";
	case 0x0302: return "3D";
      }
      return "Display";
    case 0x04:
      switch (class_id) {
	case 0x0400: return "MultimediaVideo";
	case 0x0401: return "MultimediaAudio";
	case 0x0402: return "Telephony";
	case 0x0403: return "AudioDevice";
      }
      return "Multimedia";
    case 0x05:
      switch (class_id) {
	case 0x0500: return "RAM";
	case 0x0501: return "Flash";
        case 0x0502: return "CXLMem";
      }
      return "Memory";
    case 0x06:
      switch (class_id) {
	case 0x0600: return "HostBridge";
	case 0x0601: return "ISABridge";
	case 0x0602: return "EISABridge";
	case 0x0603: return "MicroChannelBridge";
	case 0x0604: return "PCIBridge";
	case 0x0605: return "PCMCIABridge";
	case 0x0606: return "NubusBridge";
	case 0x0607: return "CardBusBridge";
	case 0x0608: return "RACEwayBridge";
	case 0x0609: return "SemiTransparentPCIBridge";
	case 0x060a: return "InfiniBandPCIHostBridge";
      }
      return "Bridge";
    case 0x07:
      switch (class_id) {
	case 0x0700: return "Serial";
	case 0x0701: return "Parallel";
	case 0x0702: return "MultiportSerial";
	case 0x0703: return "Model";
	case 0x0704: return "GPIB";
	case 0x0705: return "SmartCard";
      }
      return "Communication";
    case 0x08:
      switch (class_id) {
	case 0x0800: return "PIC";
	case 0x0801: return "DMA";
	case 0x0802: return "Timer";
	case 0x0803: return "RTC";
	case 0x0804: return "PCIHotPlug";
	case 0x0805: return "SDHost";
	case 0x0806: return "IOMMU";
      }
      return "SystemPeripheral";
    case 0x09:
      switch (class_id) {
	case 0x0900: return "Keyboard";
	case 0x0901: return "DigitizerPen";
	case 0x0902: return "Mouse";
	case 0x0903: return "Scanern";
	case 0x0904: return "Gameport";
      }
      return "Input";
    case 0x0a:
      return "DockingStation";
    case 0x0b:
      switch (class_id) {
	case 0x0b00: return "386";
	case 0x0b01: return "486";
	case 0x0b02: return "Pentium";
/* 0x0b03 and 0x0b04 might be Pentium and P6 ? */
	case 0x0b10: return "Alpha";
	case 0x0b20: return "PowerPC";
	case 0x0b30: return "MIPS";
	case 0x0b40: return "Co-Processor";
      }
      return "Processor";
    case 0x0c:
      switch (class_id) {
	case 0x0c00: return "FireWire";
	case 0x0c01: return "ACCESS";
	case 0x0c02: return "SSA";
	case 0x0c03: return "USB";
	case 0x0c04: return "FibreChannel";
	case 0x0c05: return "SMBus";
	case 0x0c06: return "InfiniBand";
	case 0x0c07: return "IPMI-SMIC";
	case 0x0c08: return "SERCOS";
	case 0x0c09: return "CANBUS";
      }
      return "SerialBus";
    case 0x0d:
      switch (class_id) {
	case 0x0d00: return "IRDA";
	case 0x0d01: return "ConsumerIR";
	case 0x0d10: return "RF";
	case 0x0d11: return "Bluetooth";
	case 0x0d12: return "Broadband";
	case 0x0d20: return "802.1a";
	case 0x0d21: return "802.1b";
      }
      return "Wireless";
    case 0x0e:
      switch (class_id) {
	case 0x0e00: return "I2O";
      }
      return "Intelligent";
    case 0x0f:
      return "Satellite";
    case 0x10:
      return "Encryption";
    case 0x11:
      return "SignalProcessing";
    case 0x12:
      return "ProcessingAccelerator";
    case 0x13:
      return "Instrumentation";
    case 0x40:
      return "Co-Processor";
  }
  return "Other";
}
