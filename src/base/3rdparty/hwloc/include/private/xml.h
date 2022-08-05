/*
 * Copyright © 2009-2017 Inria.  All rights reserved.
 * See COPYING in top-level directory.
 */

#ifndef PRIVATE_XML_H
#define PRIVATE_XML_H 1

#include "hwloc.h"

#include <sys/types.h>

HWLOC_DECLSPEC int hwloc__xml_verbose(void);

/**************
 * XML import *
 **************/

typedef struct hwloc__xml_import_state_s {
  struct hwloc__xml_import_state_s *parent;

  /* globals shared because the entire stack of states during import */
  struct hwloc_xml_backend_data_s *global;

  /* opaque data used to store backend-specific data.
   * statically allocated to allow stack-allocation by the common code without knowing actual backend needs.
   */
  char data[32];
} * hwloc__xml_import_state_t;

struct hwloc__xml_imported_v1distances_s {
  unsigned long kind;
  unsigned nbobjs;
  float *floats;
  struct hwloc__xml_imported_v1distances_s *prev, *next;
};

HWLOC_DECLSPEC int hwloc__xml_import_diff(hwloc__xml_import_state_t state, hwloc_topology_diff_t *firstdiffp);

struct hwloc_xml_backend_data_s {
  /* xml backend parameters */
  int (*look_init)(struct hwloc_xml_backend_data_s *bdata, struct hwloc__xml_import_state_s *state);
  void (*look_done)(struct hwloc_xml_backend_data_s *bdata, int result);
  void (*backend_exit)(struct hwloc_xml_backend_data_s *bdata);
  int (*next_attr)(struct hwloc__xml_import_state_s * state, char **namep, char **valuep);
  int (*find_child)(struct hwloc__xml_import_state_s * state, struct hwloc__xml_import_state_s * childstate, char **tagp);
  int (*close_tag)(struct hwloc__xml_import_state_s * state); /* look for an explicit closing tag </name> */
  void (*close_child)(struct hwloc__xml_import_state_s * state);
  int (*get_content)(struct hwloc__xml_import_state_s * state, const char **beginp, size_t expected_length); /* return 0 on empty content (and sets beginp to empty string), 1 on actual content, -1 on error or unexpected content length */
  void (*close_content)(struct hwloc__xml_import_state_s * state);
  char * msgprefix;
  void *data; /* libxml2 doc, or nolibxml buffer */
  unsigned version_major, version_minor;
  unsigned nbnumanodes;
  hwloc_obj_t first_numanode, last_numanode; /* temporary cousin-list for handling v1distances */
  struct hwloc__xml_imported_v1distances_s *first_v1dist, *last_v1dist;
};

/**************
 * XML export *
 **************/

typedef struct hwloc__xml_export_state_s {
  struct hwloc__xml_export_state_s *parent;

  void (*new_child)(struct hwloc__xml_export_state_s *parentstate, struct hwloc__xml_export_state_s *state, const char *name);
  void (*new_prop)(struct hwloc__xml_export_state_s *state, const char *name, const char *value);
  void (*add_content)(struct hwloc__xml_export_state_s *state, const char *buffer, size_t length);
  void (*end_object)(struct hwloc__xml_export_state_s *state, const char *name);

  struct hwloc__xml_export_data_s {
    hwloc_obj_t v1_memory_group; /* if we need to insert intermediate group above memory children when exporting to v1 */
  } *global;

  /* opaque data used to store backend-specific data.
   * statically allocated to allow stack-allocation by the common code without knowing actual backend needs.
   */
  char data[40];
} * hwloc__xml_export_state_t;

HWLOC_DECLSPEC void hwloc__xml_export_topology(hwloc__xml_export_state_t parentstate, hwloc_topology_t topology, unsigned long flags);

HWLOC_DECLSPEC void hwloc__xml_export_diff(hwloc__xml_export_state_t parentstate, hwloc_topology_diff_t diff);

/******************
 * XML components *
 ******************/

struct hwloc_xml_callbacks {
  int (*backend_init)(struct hwloc_xml_backend_data_s *bdata, const char *xmlpath, const char *xmlbuffer, int xmlbuflen);
  int (*export_file)(struct hwloc_topology *topology, struct hwloc__xml_export_data_s *edata, const char *filename, unsigned long flags);
  int (*export_buffer)(struct hwloc_topology *topology, struct hwloc__xml_export_data_s *edata, char **xmlbuffer, int *buflen, unsigned long flags);
  void (*free_buffer)(void *xmlbuffer);
  int (*import_diff)(struct hwloc__xml_import_state_s *state, const char *xmlpath, const char *xmlbuffer, int xmlbuflen, hwloc_topology_diff_t *diff, char **refnamep);
  int (*export_diff_file)(union hwloc_topology_diff_u *diff, const char *refname, const char *filename);
  int (*export_diff_buffer)(union hwloc_topology_diff_u *diff, const char *refname, char **xmlbuffer, int *buflen);
};

struct hwloc_xml_component {
  struct hwloc_xml_callbacks *nolibxml_callbacks;
  struct hwloc_xml_callbacks *libxml_callbacks;
};

HWLOC_DECLSPEC void hwloc_xml_callbacks_register(struct hwloc_xml_component *component);
HWLOC_DECLSPEC void hwloc_xml_callbacks_reset(void);

#endif /* PRIVATE_XML_H */
