/*
 * Copyright © 2013-2024 Inria.  All rights reserved.
 * See COPYING in top-level directory.
 */

/** \file
 * \brief Topology differences.
 */

#ifndef HWLOC_DIFF_H
#define HWLOC_DIFF_H

#ifndef HWLOC_H
#error Please include the main hwloc.h instead
#endif


#ifdef __cplusplus
extern "C" {
#elif 0
}
#endif


/** \defgroup hwlocality_diff Topology differences
 *
 * Applications that manipulate many similar topologies, for instance
 * one for each node of a homogeneous cluster, may want to compress
 * topologies to reduce the memory footprint.
 *
 * This file offers a way to manipulate the difference between topologies
 * and export/import it to/from XML.
 * Compression may therefore be achieved by storing one topology
 * entirely while the others are only described by their differences
 * with the former.
 * The actual topology can be reconstructed when actually needed by
 * applying the precomputed difference to the reference topology.
 *
 * This interface targets very similar nodes.
 * Only very simple differences between topologies are actually
 * supported, for instance a change in the memory size, the name
 * of the object, or some info attribute.
 * More complex differences such as adding or removing objects cannot
 * be represented in the difference structures and therefore return
 * errors.
 * Differences between object sets or topology-wide allowed sets,
 * cannot be represented either.
 *
 * It means that there is no need to apply the difference when
 * looking at the tree organization (how many levels, how many
 * objects per level, what kind of objects, CPU and node sets, etc)
 * and when binding to objects.
 * However the difference must be applied when looking at object
 * attributes such as the name, the memory size or info attributes.
 *
 * @{
 */


/** \brief Type of one object attribute difference.
 */
typedef enum hwloc_topology_diff_obj_attr_type_e {
  /** \brief The object local memory is modified.
   * The union is a hwloc_topology_diff_obj_attr_u::hwloc_topology_diff_obj_attr_uint64_s
   * (and the index field is ignored).
   */
  HWLOC_TOPOLOGY_DIFF_OBJ_ATTR_SIZE,

  /** \brief The object name is modified.
   * The union is a hwloc_topology_diff_obj_attr_u::hwloc_topology_diff_obj_attr_string_s
   * (and the name field is ignored).
   */

  HWLOC_TOPOLOGY_DIFF_OBJ_ATTR_NAME,
  /** \brief the value of an info attribute is modified.
   * The union is a hwloc_topology_diff_obj_attr_u::hwloc_topology_diff_obj_attr_string_s.
   */
  HWLOC_TOPOLOGY_DIFF_OBJ_ATTR_INFO
} hwloc_topology_diff_obj_attr_type_t;

/** \brief One object attribute difference.
 */
union hwloc_topology_diff_obj_attr_u {
  struct hwloc_topology_diff_obj_attr_generic_s {
    /* each part of the union must start with these */
    hwloc_topology_diff_obj_attr_type_t type;
  } generic;

  /** \brief Integer attribute modification with an optional index. */
  struct hwloc_topology_diff_obj_attr_uint64_s {
    /* used for storing integer attributes */
    hwloc_topology_diff_obj_attr_type_t type;
    hwloc_uint64_t index; /* not used for SIZE */
    hwloc_uint64_t oldvalue;
    hwloc_uint64_t newvalue;
  } uint64;

  /** \brief String attribute modification with an optional name */
  struct hwloc_topology_diff_obj_attr_string_s {
    /* used for storing name and info pairs */
    hwloc_topology_diff_obj_attr_type_t type;
    char *name; /* not used for NAME */
    char *oldvalue;
    char *newvalue;
  } string;
};


/** \brief Type of one element of a difference list.
 */
typedef enum hwloc_topology_diff_type_e {
  /** \brief An object attribute was changed.
   * The union is a hwloc_topology_diff_u::hwloc_topology_diff_obj_attr_s.
   */
  HWLOC_TOPOLOGY_DIFF_OBJ_ATTR,

  /** \brief The difference is too complex,
   * it cannot be represented. The difference below
   * this object has not been checked.
   * hwloc_topology_diff_build() will return 1.
   *
   * The union is a hwloc_topology_diff_u::hwloc_topology_diff_too_complex_s.
   */
  HWLOC_TOPOLOGY_DIFF_TOO_COMPLEX
} hwloc_topology_diff_type_t;

/** \brief One element of a difference list between two topologies.
 */
typedef union hwloc_topology_diff_u {
  struct hwloc_topology_diff_generic_s {
    /* each part of the union must start with these */
    hwloc_topology_diff_type_t type;
    union hwloc_topology_diff_u * next; /* pointer to the next element of the list, or NULL */
  } generic;

  /* A difference in an object attribute. */
  struct hwloc_topology_diff_obj_attr_s {
    hwloc_topology_diff_type_t type; /* must be ::HWLOC_TOPOLOGY_DIFF_OBJ_ATTR */
    union hwloc_topology_diff_u * next;
    /* List of attribute differences for a single object */
    int obj_depth;
    unsigned obj_index;
    union hwloc_topology_diff_obj_attr_u diff;
  } obj_attr;

  /* A difference that is too complex. */
  struct hwloc_topology_diff_too_complex_s {
    hwloc_topology_diff_type_t type; /* must be ::HWLOC_TOPOLOGY_DIFF_TOO_COMPLEX */
    union hwloc_topology_diff_u * next;
    /* Where we had to stop computing the diff in the first topology */
    int obj_depth;
    unsigned obj_index;
  } too_complex;
} * hwloc_topology_diff_t;


/** \brief Compute the difference between 2 topologies.
 *
 * The difference is stored as a list of ::hwloc_topology_diff_t entries
 * starting at \p diff.
 * It is computed by doing a depth-first traversal of both topology trees
 * simultaneously.
 *
 * If the difference between 2 objects is too complex to be represented
 * (for instance if some objects have different types, or different numbers
 * of children), a special diff entry of type ::HWLOC_TOPOLOGY_DIFF_TOO_COMPLEX
 * is queued.
 * The computation of the diff does not continue below these objects.
 * So each such diff entry means that the difference between two subtrees
 * could not be computed.
 *
 * \return 0 if the difference can be represented properly.
 *
 * \return 0 with \p diff pointing to NULL if there is no difference
 * between the topologies.
 *
 * \return 1 if the difference is too complex (see above). Some entries in
 * the list will be of type ::HWLOC_TOPOLOGY_DIFF_TOO_COMPLEX.
 *
 * \return -1 on any other error.
 *
 * \note \p flags is currently not used. It should be 0.
 *
 * \note The output diff has to be freed with hwloc_topology_diff_destroy().
 *
 * \note The output diff can only be exported to XML or passed to
 * hwloc_topology_diff_apply() if 0 was returned, i.e. if no entry of type
 * ::HWLOC_TOPOLOGY_DIFF_TOO_COMPLEX is listed.
 *
 * \note The output diff may be modified by removing some entries from
 * the list. The removed entries should be freed by passing them to
 * to hwloc_topology_diff_destroy() (possible as another list).
*/
HWLOC_DECLSPEC int hwloc_topology_diff_build(hwloc_topology_t topology, hwloc_topology_t newtopology, unsigned long flags, hwloc_topology_diff_t *diff);

/** \brief Flags to be given to hwloc_topology_diff_apply().
 */
enum hwloc_topology_diff_apply_flags_e {
  /** \brief Apply topology diff in reverse direction.
   * \hideinitializer
   */
  HWLOC_TOPOLOGY_DIFF_APPLY_REVERSE = (1UL<<0)
};

/** \brief Apply a topology diff to an existing topology.
 *
 * \p flags is an OR'ed set of ::hwloc_topology_diff_apply_flags_e.
 *
 * The new topology is modified in place. hwloc_topology_dup()
 * may be used to duplicate it before patching.
 *
 * If the difference cannot be applied entirely, all previous applied
 * elements are unapplied before returning.
 *
 * \return 0 on success.
 *
 * \return -N if applying the difference failed while trying
 * to apply the N-th part of the difference. For instance -1
 * is returned if the very first difference element could not
 * be applied.
 */
HWLOC_DECLSPEC int hwloc_topology_diff_apply(hwloc_topology_t topology, hwloc_topology_diff_t diff, unsigned long flags);

/** \brief Destroy a list of topology differences.
 *
 * \return 0.
 */
HWLOC_DECLSPEC int hwloc_topology_diff_destroy(hwloc_topology_diff_t diff);

/** \brief Load a list of topology differences from a XML file.
 *
 * If not \c NULL, \p refname will be filled with the identifier
 * string of the reference topology for the difference file,
 * if any was specified in the XML file.
 * This identifier is usually the name of the other XML file
 * that contains the reference topology.
 *
 * \return 0 on success, -1 on error.
 *
 * \note the pointer returned in refname should later be freed
 * by the caller.
 */
HWLOC_DECLSPEC int hwloc_topology_diff_load_xml(const char *xmlpath, hwloc_topology_diff_t *diff, char **refname);

/** \brief Export a list of topology differences to a XML file.
 *
 * If not \c NULL, \p refname defines an identifier string
 * for the reference topology which was used as a base when
 * computing this difference.
 * This identifier is usually the name of the other XML file
 * that contains the reference topology.
 * This attribute is given back when reading the diff from XML.
 *
 * \return 0 on success, -1 on error.
 */
HWLOC_DECLSPEC int hwloc_topology_diff_export_xml(hwloc_topology_diff_t diff, const char *refname, const char *xmlpath);

/** \brief Load a list of topology differences from a XML buffer.
 *
 * Build a list of differences from the XML memory buffer given
 * at \p xmlbuffer and of length \p buflen (including an ending \c \0).
 * This buffer may have been filled earlier with
 * hwloc_topology_diff_export_xmlbuffer().
 *
 * If not \c NULL, \p refname will be filled with the identifier
 * string of the reference topology for the difference file,
 * if any was specified in the XML file.
 * This identifier is usually the name of the other XML file
 * that contains the reference topology.
 *
 * \return 0 on success, -1 on error.
 *
 * \note the pointer returned in refname should later be freed
 * by the caller.
  */
HWLOC_DECLSPEC int hwloc_topology_diff_load_xmlbuffer(const char *xmlbuffer, int buflen, hwloc_topology_diff_t *diff, char **refname);

/** \brief Export a list of topology differences to a XML buffer.
 *
 * If not \c NULL, \p refname defines an identifier string
 * for the reference topology which was used as a base when
 * computing this difference.
 * This identifier is usually the name of the other XML file
 * that contains the reference topology.
 * This attribute is given back when reading the diff from XML.
 *
 * The returned buffer ends with a \c \0 that is included in the returned
 * length.
 *
 * \return 0 on success, -1 on error.
 *
 * \note The XML buffer should later be freed with hwloc_free_xmlbuffer().
 */
HWLOC_DECLSPEC int hwloc_topology_diff_export_xmlbuffer(hwloc_topology_diff_t diff, const char *refname, char **xmlbuffer, int *buflen);

/** @} */


#ifdef __cplusplus
} /* extern "C" */
#endif


#endif /* HWLOC_DIFF_H */
