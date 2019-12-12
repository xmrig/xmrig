/*
 * Copyright © 2013-2019 Inria.  All rights reserved.
 * Copyright © 2016 Cisco Systems, Inc.  All rights reserved.
 * See COPYING in top-level directory.
 */

#ifndef HWLOC_PLUGINS_H
#define HWLOC_PLUGINS_H

/** \file
 * \brief Public interface for building hwloc plugins.
 */

struct hwloc_backend;

#include "hwloc.h"

#ifdef HWLOC_INSIDE_PLUGIN
/* needed for hwloc_plugin_check_namespace() */
#include <ltdl.h>
#endif



/** \defgroup hwlocality_disc_components Components and Plugins: Discovery components
 * @{
 */

/** \brief Discovery component structure
 *
 * This is the major kind of components, taking care of the discovery.
 * They are registered by generic components, either statically-built or as plugins.
 */
struct hwloc_disc_component {
  /** \brief Name.
   * If this component is built as a plugin, this name does not have to match the plugin filename.
   */
  const char *name;

  /** \brief Discovery phases performed by this component.
   * OR'ed set of ::hwloc_disc_phase_t
   */
  unsigned phases;

  /** \brief Component phases to exclude, as an OR'ed set of ::hwloc_disc_phase_t.
   *
   * For a GLOBAL component, this usually includes all other phases (\c ~UL).
   *
   * Other components only exclude types that may bring conflicting
   * topology information. MISC components should likely not be excluded
   * since they usually bring non-primary additional information.
   */
  unsigned excluded_phases;

  /** \brief Instantiate callback to create a backend from the component.
   * Parameters data1, data2, data3 are NULL except for components
   * that have special enabling routines such as hwloc_topology_set_xml(). */
  struct hwloc_backend * (*instantiate)(struct hwloc_topology *topology, struct hwloc_disc_component *component, unsigned excluded_phases, const void *data1, const void *data2, const void *data3);

  /** \brief Component priority.
   * Used to sort topology->components, higher priority first.
   * Also used to decide between two components with the same name.
   *
   * Usual values are
   * 50 for native OS (or platform) components,
   * 45 for x86,
   * 40 for no-OS fallback,
   * 30 for global components (xml, synthetic),
   * 20 for pci,
   * 10 for other misc components (opencl etc.).
   */
  unsigned priority;

  /** \brief Enabled by default.
   * If unset, if will be disabled unless explicitly requested.
   */
  unsigned enabled_by_default;

  /** \private Used internally to list components by priority on topology->components
   * (the component structure is usually read-only,
   *  the core copies it before using this field for queueing)
   */
  struct hwloc_disc_component * next;
};

/** @} */




/** \defgroup hwlocality_disc_backends Components and Plugins: Discovery backends
 * @{
 */

/** \brief Discovery phase */
typedef enum hwloc_disc_phase_e {
  /** \brief xml or synthetic, platform-specific components such as bgq.
   * Discovers everything including CPU, memory, I/O and everything else.
   * A component with a Global phase usually excludes all other phases.
   * \hideinitializer */
  HWLOC_DISC_PHASE_GLOBAL = (1U<<0),

  /** \brief CPU discovery.
   * \hideinitializer */
  HWLOC_DISC_PHASE_CPU = (1U<<1),

  /** \brief Attach memory to existing CPU objects.
   * \hideinitializer */
  HWLOC_DISC_PHASE_MEMORY = (1U<<2),

  /** \brief Attach PCI devices and bridges to existing CPU objects.
   * \hideinitializer */
  HWLOC_DISC_PHASE_PCI = (1U<<3),

  /** \brief I/O discovery that requires PCI devices (OS devices such as OpenCL, CUDA, etc.).
   * \hideinitializer */
  HWLOC_DISC_PHASE_IO = (1U<<4),

  /** \brief Misc objects that gets added below anything else.
   * \hideinitializer */
  HWLOC_DISC_PHASE_MISC = (1U<<5),

  /** \brief Annotating existing objects, adding distances, etc.
   * \hideinitializer */
  HWLOC_DISC_PHASE_ANNOTATE = (1U<<6),

  /** \brief Final tweaks to a ready-to-use topology.
   * This phase runs once the topology is loaded, before it is returned to the topology.
   * Hence it may only use the main hwloc API for modifying the topology,
   * for instance by restricting it, adding info attributes, etc.
   * \hideinitializer */
  HWLOC_DISC_PHASE_TWEAK = (1U<<7)
} hwloc_disc_phase_t;

/** \brief Discovery status flags */
enum hwloc_disc_status_flag_e {
  /** \brief The sets of allowed resources were already retrieved \hideinitializer */
  HWLOC_DISC_STATUS_FLAG_GOT_ALLOWED_RESOURCES = (1UL<<1)
};

/** \brief Discovery status structure
 *
 * Used by the core and backends to inform about what has been/is being done
 * during the discovery process.
 */
struct hwloc_disc_status {
  /** \brief The current discovery phase that is performed.
   * Must match one of the phases in the component phases field.
   */
  hwloc_disc_phase_t phase;

  /** \brief Dynamically excluded phases.
   * If a component decides during discovery that some phases are no longer needed.
   */
  unsigned excluded_phases;

  /** \brief OR'ed set of hwloc_disc_status_flag_e */
  unsigned long flags;
};

/** \brief Discovery backend structure
 *
 * A backend is the instantiation of a discovery component.
 * When a component gets enabled for a topology,
 * its instantiate() callback creates a backend.
 *
 * hwloc_backend_alloc() initializes all fields to default values
 * that the component may change (except "component" and "next")
 * before enabling the backend with hwloc_backend_enable().
 *
 * Most backends assume that the topology is_thissystem flag is
 * set because they talk to the underlying operating system.
 * However they may still be used in topologies without the
 * is_thissystem flag for debugging reasons.
 * In practice, they are usually auto-disabled in such cases
 * (excluded by xml or synthetic backends, or by environment
 *  variables when changing the Linux fsroot or the x86 cpuid path).
 */
struct hwloc_backend {
  /** \private Reserved for the core, set by hwloc_backend_alloc() */
  struct hwloc_disc_component * component;
  /** \private Reserved for the core, set by hwloc_backend_enable() */
  struct hwloc_topology * topology;
  /** \private Reserved for the core. Set to 1 if forced through envvar, 0 otherwise. */
  int envvar_forced;
  /** \private Reserved for the core. Used internally to list backends topology->backends. */
  struct hwloc_backend * next;

  /** \brief Discovery phases performed by this component, possibly without some of them if excluded by other components.
   * OR'ed set of ::hwloc_disc_phase_t
   */
  unsigned phases;

  /** \brief Backend flags, currently always 0. */
  unsigned long flags;

  /** \brief Backend-specific 'is_thissystem' property.
   * Set to 0 if the backend disables the thissystem flag for this topology
   * (e.g. loading from xml or synthetic string,
   *  or using a different fsroot on Linux, or a x86 CPUID dump).
   * Set to -1 if the backend doesn't care (default).
   */
  int is_thissystem;

  /** \brief Backend private data, or NULL if none. */
  void * private_data;
  /** \brief Callback for freeing the private_data.
   * May be NULL.
   */
  void (*disable)(struct hwloc_backend *backend);

  /** \brief Main discovery callback.
   * returns -1 on error, either because it couldn't add its objects ot the existing topology,
   * or because of an actual discovery/gathering failure.
   * May be NULL.
   */
  int (*discover)(struct hwloc_backend *backend, struct hwloc_disc_status *status);

  /** \brief Callback to retrieve the locality of a PCI object.
   * Called by the PCI core when attaching PCI hierarchy to CPU objects.
   * May be NULL.
   */
  int (*get_pci_busid_cpuset)(struct hwloc_backend *backend, struct hwloc_pcidev_attr_s *busid, hwloc_bitmap_t cpuset);
};

/** \brief Allocate a backend structure, set good default values, initialize backend->component and topology, etc.
 * The caller will then modify whatever needed, and call hwloc_backend_enable().
 */
HWLOC_DECLSPEC struct hwloc_backend * hwloc_backend_alloc(struct hwloc_topology *topology, struct hwloc_disc_component *component);

/** \brief Enable a previously allocated and setup backend. */
HWLOC_DECLSPEC int hwloc_backend_enable(struct hwloc_backend *backend);

/** @} */




/** \defgroup hwlocality_generic_components Components and Plugins: Generic components
 * @{
 */

/** \brief Generic component type */
typedef enum hwloc_component_type_e {
  /** \brief The data field must point to a struct hwloc_disc_component. */
  HWLOC_COMPONENT_TYPE_DISC,

  /** \brief The data field must point to a struct hwloc_xml_component. */
  HWLOC_COMPONENT_TYPE_XML
} hwloc_component_type_t;

/** \brief Generic component structure
 *
 * Generic components structure, either statically listed by configure in static-components.h
 * or dynamically loaded as a plugin.
 */
struct hwloc_component {
  /** \brief Component ABI version, set to ::HWLOC_COMPONENT_ABI */
  unsigned abi;

  /** \brief Process-wide component initialization callback.
   *
   * This optional callback is called when the component is registered
   * to the hwloc core (after loading the plugin).
   *
   * When the component is built as a plugin, this callback
   * should call hwloc_check_plugin_namespace()
   * and return an negative error code on error.
   *
   * \p flags is always 0 for now.
   *
   * \return 0 on success, or a negative code on error.
   *
   * \note If the component uses ltdl for loading its own plugins,
   * it should load/unload them only in init() and finalize(),
   * to avoid race conditions with hwloc's use of ltdl.
   */
  int (*init)(unsigned long flags);

  /** \brief Process-wide component termination callback.
   *
   * This optional callback is called after unregistering the component
   * from the hwloc core (before unloading the plugin).
   *
   * \p flags is always 0 for now.
   *
   * \note If the component uses ltdl for loading its own plugins,
   * it should load/unload them only in init() and finalize(),
   * to avoid race conditions with hwloc's use of ltdl.
   */
  void (*finalize)(unsigned long flags);

  /** \brief Component type */
  hwloc_component_type_t type;

  /** \brief Component flags, unused for now */
  unsigned long flags;

  /** \brief Component data, pointing to a struct hwloc_disc_component or struct hwloc_xml_component. */
  void * data;
};

/** @} */




/** \defgroup hwlocality_components_core_funcs Components and Plugins: Core functions to be used by components
 * @{
 */

/** \brief Add an object to the topology.
 *
 * It is sorted along the tree of other objects according to the inclusion of
 * cpusets, to eventually be added as a child of the smallest object including
 * this object.
 *
 * If the cpuset is empty, the type of the object (and maybe some attributes)
 * must be enough to find where to insert the object. This is especially true
 * for NUMA nodes with memory and no CPUs.
 *
 * The given object should not have children.
 *
 * This shall only be called before levels are built.
 *
 * In case of error, hwloc_report_os_error() is called.
 *
 * The caller should check whether the object type is filtered-out before calling this function.
 *
 * The topology cpuset/nodesets will be enlarged to include the object sets.
 *
 * Returns the object on success.
 * Returns NULL and frees obj on error.
 * Returns another object and frees obj if it was merged with an identical pre-existing object.
 */
HWLOC_DECLSPEC struct hwloc_obj *hwloc_insert_object_by_cpuset(struct hwloc_topology *topology, hwloc_obj_t obj);

/** \brief Type of error callbacks during object insertion */
typedef void (*hwloc_report_error_t)(const char * msg, int line);
/** \brief Report an insertion error from a backend */
HWLOC_DECLSPEC void hwloc_report_os_error(const char * msg, int line);
/** \brief Check whether insertion errors are hidden */
HWLOC_DECLSPEC int hwloc_hide_errors(void);

/** \brief Add an object to the topology and specify which error callback to use.
 *
 * This function is similar to hwloc_insert_object_by_cpuset() but it allows specifying
 * where to start insertion from (if \p root is NULL, the topology root object is used),
 * and specifying the error callback.
 */
HWLOC_DECLSPEC struct hwloc_obj *hwloc__insert_object_by_cpuset(struct hwloc_topology *topology, hwloc_obj_t root, hwloc_obj_t obj, hwloc_report_error_t report_error);

/** \brief Insert an object somewhere in the topology.
 *
 * It is added as the last child of the given parent.
 * The cpuset is completely ignored, so strange objects such as I/O devices should
 * preferably be inserted with this.
 *
 * When used for "normal" children with cpusets (when importing from XML
 * when duplicating a topology), the caller should make sure that:
 * - children are inserted in order,
 * - children cpusets do not intersect.
 *
 * The given object may have normal, I/O or Misc children, as long as they are in order as well.
 * These children must have valid parent and next_sibling pointers.
 *
 * The caller should check whether the object type is filtered-out before calling this function.
 */
HWLOC_DECLSPEC void hwloc_insert_object_by_parent(struct hwloc_topology *topology, hwloc_obj_t parent, hwloc_obj_t obj);

/** \brief Allocate and initialize an object of the given type and physical index.
 *
 * If \p os_index is unknown or irrelevant, use \c HWLOC_UNKNOWN_INDEX.
 */
HWLOC_DECLSPEC hwloc_obj_t hwloc_alloc_setup_object(hwloc_topology_t topology, hwloc_obj_type_t type, unsigned os_index);

/** \brief Setup object cpusets/nodesets by OR'ing its children.
 *
 * Used when adding an object late in the topology.
 * Will update the new object by OR'ing all its new children sets.
 *
 * Used when PCI backend adds a hostbridge parent, when distances
 * add a new Group, etc.
 */
HWLOC_DECLSPEC int hwloc_obj_add_children_sets(hwloc_obj_t obj);

/** \brief Request a reconnection of children and levels in the topology.
 *
 * May be used by backends during discovery if they need arrays or lists
 * of object within levels or children to be fully connected.
 *
 * \p flags is currently unused, must 0.
 */
HWLOC_DECLSPEC int hwloc_topology_reconnect(hwloc_topology_t topology, unsigned long flags __hwloc_attribute_unused);

/** \brief Make sure that plugins can lookup core symbols.
 *
 * This is a sanity check to avoid lazy-lookup failures when libhwloc
 * is loaded within a plugin, and later tries to load its own plugins.
 * This may fail (and abort the program) if libhwloc symbols are in a
 * private namespace.
 *
 * \return 0 on success.
 * \return -1 if the plugin cannot be successfully loaded. The caller
 * plugin init() callback should return a negative error code as well.
 *
 * Plugins should call this function in their init() callback to avoid
 * later crashes if lazy symbol resolution is used by the upper layer that
 * loaded hwloc (e.g. OpenCL implementations using dlopen with RTLD_LAZY).
 *
 * \note The build system must define HWLOC_INSIDE_PLUGIN if and only if
 * building the caller as a plugin.
 *
 * \note This function should remain inline so plugins can call it even
 * when they cannot find libhwloc symbols.
 */
static __hwloc_inline int
hwloc_plugin_check_namespace(const char *pluginname __hwloc_attribute_unused, const char *symbol __hwloc_attribute_unused)
{
#ifdef HWLOC_INSIDE_PLUGIN
  lt_dlhandle handle;
  void *sym;
  handle = lt_dlopen(NULL);
  if (!handle)
    /* cannot check, assume things will work */
    return 0;
  sym = lt_dlsym(handle, symbol);
  lt_dlclose(handle);
  if (!sym) {
    static int verboseenv_checked = 0;
    static int verboseenv_value = 0;
    if (!verboseenv_checked) {
      const char *verboseenv = getenv("HWLOC_PLUGINS_VERBOSE");
      verboseenv_value = verboseenv ? atoi(verboseenv) : 0;
      verboseenv_checked = 1;
    }
    if (verboseenv_value)
      fprintf(stderr, "Plugin `%s' disabling itself because it cannot find the `%s' core symbol.\n",
	      pluginname, symbol);
    return -1;
  }
#endif /* HWLOC_INSIDE_PLUGIN */
  return 0;
}

/** @} */




/** \defgroup hwlocality_components_filtering Components and Plugins: Filtering objects
 * @{
 */

/** \brief Check whether the given PCI device classid is important.
 *
 * \return 1 if important, 0 otherwise.
 */
static __hwloc_inline int
hwloc_filter_check_pcidev_subtype_important(unsigned classid)
{
  unsigned baseclass = classid >> 8;
  return (baseclass == 0x03 /* PCI_BASE_CLASS_DISPLAY */
	  || baseclass == 0x02 /* PCI_BASE_CLASS_NETWORK */
	  || baseclass == 0x01 /* PCI_BASE_CLASS_STORAGE */
	  || baseclass == 0x0b /* PCI_BASE_CLASS_PROCESSOR */
	  || classid == 0x0c04 /* PCI_CLASS_SERIAL_FIBER */
	  || classid == 0x0c06 /* PCI_CLASS_SERIAL_INFINIBAND */
	  || baseclass == 0x12 /* Processing Accelerators */);
}

/** \brief Check whether the given OS device subtype is important.
 *
 * \return 1 if important, 0 otherwise.
 */
static __hwloc_inline int
hwloc_filter_check_osdev_subtype_important(hwloc_obj_osdev_type_t subtype)
{
  return (subtype != HWLOC_OBJ_OSDEV_DMA);
}

/** \brief Check whether a non-I/O object type should be filtered-out.
 *
 * Cannot be used for I/O objects.
 *
 * \return 1 if the object type should be kept, 0 otherwise.
 */
static __hwloc_inline int
hwloc_filter_check_keep_object_type(hwloc_topology_t topology, hwloc_obj_type_t type)
{
  enum hwloc_type_filter_e filter = HWLOC_TYPE_FILTER_KEEP_NONE;
  hwloc_topology_get_type_filter(topology, type, &filter);
  assert(filter != HWLOC_TYPE_FILTER_KEEP_IMPORTANT); /* IMPORTANT only used for I/O */
  return filter == HWLOC_TYPE_FILTER_KEEP_NONE ? 0 : 1;
}

/** \brief Check whether the given object should be filtered-out.
 *
 * \return 1 if the object type should be kept, 0 otherwise.
 */
static __hwloc_inline int
hwloc_filter_check_keep_object(hwloc_topology_t topology, hwloc_obj_t obj)
{
  hwloc_obj_type_t type = obj->type;
  enum hwloc_type_filter_e filter = HWLOC_TYPE_FILTER_KEEP_NONE;
  hwloc_topology_get_type_filter(topology, type, &filter);
  if (filter == HWLOC_TYPE_FILTER_KEEP_NONE)
    return 0;
  if (filter == HWLOC_TYPE_FILTER_KEEP_IMPORTANT) {
    if (type == HWLOC_OBJ_PCI_DEVICE)
      return hwloc_filter_check_pcidev_subtype_important(obj->attr->pcidev.class_id);
    if (type == HWLOC_OBJ_OS_DEVICE)
      return hwloc_filter_check_osdev_subtype_important(obj->attr->osdev.type);
  }
  return 1;
}

/** @} */




/** \defgroup hwlocality_components_pcidisc Components and Plugins: helpers for PCI discovery
 * @{
 */

/** \brief Return the offset of the given capability in the PCI config space buffer
 *
 * This function requires a 256-bytes config space. Unknown/unavailable bytes should be set to 0xff.
 */
HWLOC_DECLSPEC unsigned hwloc_pcidisc_find_cap(const unsigned char *config, unsigned cap);

/** \brief Fill linkspeed by reading the PCI config space where PCI_CAP_ID_EXP is at position offset.
 *
 * Needs 20 bytes of EXP capability block starting at offset in the config space
 * for registers up to link status.
 */
HWLOC_DECLSPEC int hwloc_pcidisc_find_linkspeed(const unsigned char *config, unsigned offset, float *linkspeed);

/** \brief Return the hwloc object type (PCI device or Bridge) for the given class and configuration space.
 *
 * This function requires 16 bytes of common configuration header at the beginning of config.
 */
HWLOC_DECLSPEC hwloc_obj_type_t hwloc_pcidisc_check_bridge_type(unsigned device_class, const unsigned char *config);

/** \brief Fills the attributes of the given PCI bridge using the given PCI config space.
 *
 * This function requires 32 bytes of common configuration header at the beginning of config.
 *
 * Returns -1 and destroys /p obj if bridge fields are invalid.
 */
HWLOC_DECLSPEC int hwloc_pcidisc_find_bridge_buses(unsigned domain, unsigned bus, unsigned dev, unsigned func,
						   unsigned *secondary_busp, unsigned *subordinate_busp,
						   const unsigned char *config);

/** \brief Insert a PCI object in the given PCI tree by looking at PCI bus IDs.
 *
 * If \p treep points to \c NULL, the new object is inserted there.
 */
HWLOC_DECLSPEC void hwloc_pcidisc_tree_insert_by_busid(struct hwloc_obj **treep, struct hwloc_obj *obj);

/** \brief Add some hostbridges on top of the given tree of PCI objects and attach them to the topology.
 *
 * Other backends may lookup PCI objects or localities (for instance to attach OS devices)
 * by using hwloc_pcidisc_find_by_busid() or hwloc_pcidisc_find_busid_parent().
 */
HWLOC_DECLSPEC int hwloc_pcidisc_tree_attach(struct hwloc_topology *topology, struct hwloc_obj *tree);

/** @} */




/** \defgroup hwlocality_components_pcifind Components and Plugins: finding PCI objects during other discoveries
 * @{
 */

/** \brief Find the normal parent of a PCI bus ID.
 *
 * Look at PCI affinity to find out where the given PCI bus ID should be attached.
 *
 * This function should be used to attach an I/O device under the corresponding
 * PCI object (if any), or under a normal (non-I/O) object with same locality.
 */
HWLOC_DECLSPEC struct hwloc_obj * hwloc_pci_find_parent_by_busid(struct hwloc_topology *topology, unsigned domain, unsigned bus, unsigned dev, unsigned func);

/** @} */




#endif /* HWLOC_PLUGINS_H */
