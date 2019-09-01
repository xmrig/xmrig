/*
 * Copyright © 2009-2018 Inria.  All rights reserved.
 * Copyright © 2009-2012 Université Bordeaux
 * Copyright © 2009-2011 Cisco Systems, Inc.  All rights reserved.
 * See COPYING in top-level directory.
 */

/** \file
 * \brief Exporting Topologies to XML or to Synthetic strings.
 */

#ifndef HWLOC_EXPORT_H
#define HWLOC_EXPORT_H

#ifndef HWLOC_H
#error Please include the main hwloc.h instead
#endif


#ifdef __cplusplus
extern "C" {
#elif 0
}
#endif


/** \defgroup hwlocality_xmlexport Exporting Topologies to XML
 * @{
 */

/** \brief Flags for exporting XML topologies.
 *
 * Flags to be given as a OR'ed set to hwloc_topology_export_xml().
 */
enum hwloc_topology_export_xml_flags_e {
 /** \brief Export XML that is loadable by hwloc v1.x.
  * However, the export may miss some details about the topology.
  * \hideinitializer
  */
 HWLOC_TOPOLOGY_EXPORT_XML_FLAG_V1 = (1UL<<0)
};

/** \brief Export the topology into an XML file.
 *
 * This file may be loaded later through hwloc_topology_set_xml().
 *
 * By default, the latest export format is used, which means older hwloc
 * releases (e.g. v1.x) will not be able to import it.
 * Exporting to v1.x specific XML format is possible using flag
 * ::HWLOC_TOPOLOGY_EXPORT_XML_FLAG_V1 but it may miss some details
 * about the topology.
 * If there is any chance that the exported file may ever be imported
 * back by a process using hwloc 1.x, one should consider detecting
 * it at runtime and using the corresponding export format.
 *
 * \p flags is a OR'ed set of ::hwloc_topology_export_xml_flags_e.
 *
 * \return -1 if a failure occured.
 *
 * \note See also hwloc_topology_set_userdata_export_callback()
 * for exporting application-specific object userdata.
 *
 * \note The topology-specific userdata pointer is ignored when exporting to XML.
 *
 * \note Only printable characters may be exported to XML string attributes.
 * Any other character, especially any non-ASCII character, will be silently
 * dropped.
 *
 * \note If \p name is "-", the XML output is sent to the standard output.
 */
HWLOC_DECLSPEC int hwloc_topology_export_xml(hwloc_topology_t topology, const char *xmlpath, unsigned long flags);

/** \brief Export the topology into a newly-allocated XML memory buffer.
 *
 * \p xmlbuffer is allocated by the callee and should be freed with
 * hwloc_free_xmlbuffer() later in the caller.
 *
 * This memory buffer may be loaded later through hwloc_topology_set_xmlbuffer().
 *
 * By default, the latest export format is used, which means older hwloc
 * releases (e.g. v1.x) will not be able to import it.
 * Exporting to v1.x specific XML format is possible using flag
 * ::HWLOC_TOPOLOGY_EXPORT_XML_FLAG_V1 but it may miss some details
 * about the topology.
 * If there is any chance that the exported buffer may ever be imported
 * back by a process using hwloc 1.x, one should consider detecting
 * it at runtime and using the corresponding export format.
 *
 * The returned buffer ends with a \0 that is included in the returned
 * length.
 *
 * \p flags is a OR'ed set of ::hwloc_topology_export_xml_flags_e.
 *
 * \return -1 if a failure occured.
 *
 * \note See also hwloc_topology_set_userdata_export_callback()
 * for exporting application-specific object userdata.
 *
 * \note The topology-specific userdata pointer is ignored when exporting to XML.
 *
 * \note Only printable characters may be exported to XML string attributes.
 * Any other character, especially any non-ASCII character, will be silently
 * dropped.
 */
HWLOC_DECLSPEC int hwloc_topology_export_xmlbuffer(hwloc_topology_t topology, char **xmlbuffer, int *buflen, unsigned long flags);

/** \brief Free a buffer allocated by hwloc_topology_export_xmlbuffer() */
HWLOC_DECLSPEC void hwloc_free_xmlbuffer(hwloc_topology_t topology, char *xmlbuffer);

/** \brief Set the application-specific callback for exporting object userdata
 *
 * The object userdata pointer is not exported to XML by default because hwloc
 * does not know what it contains.
 *
 * This function lets applications set \p export_cb to a callback function
 * that converts this opaque userdata into an exportable string.
 *
 * \p export_cb is invoked during XML export for each object whose
 * \p userdata pointer is not \c NULL.
 * The callback should use hwloc_export_obj_userdata() or
 * hwloc_export_obj_userdata_base64() to actually export
 * something to XML (possibly multiple times per object).
 *
 * \p export_cb may be set to \c NULL if userdata should not be exported to XML.
 *
 * \note The topology-specific userdata pointer is ignored when exporting to XML.
 */
HWLOC_DECLSPEC void hwloc_topology_set_userdata_export_callback(hwloc_topology_t topology,
								void (*export_cb)(void *reserved, hwloc_topology_t topology, hwloc_obj_t obj));

/** \brief Export some object userdata to XML
 *
 * This function may only be called from within the export() callback passed
 * to hwloc_topology_set_userdata_export_callback().
 * It may be invoked one of multiple times to export some userdata to XML.
 * The \p buffer content of length \p length is stored with optional name
 * \p name.
 *
 * When importing this XML file, the import() callback (if set) will be
 * called exactly as many times as hwloc_export_obj_userdata() was called
 * during export(). It will receive the corresponding \p name, \p buffer
 * and \p length arguments.
 *
 * \p reserved, \p topology and \p obj must be the first three parameters
 * that were given to the export callback.
 *
 * Only printable characters may be exported to XML string attributes.
 * If a non-printable character is passed in \p name or \p buffer,
 * the function returns -1 with errno set to EINVAL.
 *
 * If exporting binary data, the application should first encode into
 * printable characters only (or use hwloc_export_obj_userdata_base64()).
 * It should also take care of portability issues if the export may
 * be reimported on a different architecture.
 */
HWLOC_DECLSPEC int hwloc_export_obj_userdata(void *reserved, hwloc_topology_t topology, hwloc_obj_t obj, const char *name, const void *buffer, size_t length);

/** \brief Encode and export some object userdata to XML
 *
 * This function is similar to hwloc_export_obj_userdata() but it encodes
 * the input buffer into printable characters before exporting.
 * On import, decoding is automatically performed before the data is given
 * to the import() callback if any.
 *
 * This function may only be called from within the export() callback passed
 * to hwloc_topology_set_userdata_export_callback().
 *
 * The function does not take care of portability issues if the export
 * may be reimported on a different architecture.
 */
HWLOC_DECLSPEC int hwloc_export_obj_userdata_base64(void *reserved, hwloc_topology_t topology, hwloc_obj_t obj, const char *name, const void *buffer, size_t length);

/** \brief Set the application-specific callback for importing userdata
 *
 * On XML import, userdata is ignored by default because hwloc does not know
 * how to store it in memory.
 *
 * This function lets applications set \p import_cb to a callback function
 * that will get the XML-stored userdata and store it in the object as expected
 * by the application.
 *
 * \p import_cb is called during hwloc_topology_load() as many times as
 * hwloc_export_obj_userdata() was called during export. The topology
 * is not entirely setup yet. Object attributes are ready to consult,
 * but links between objects are not.
 *
 * \p import_cb may be \c NULL if userdata should be ignored during import.
 *
 * \note \p buffer contains \p length characters followed by a null byte ('\0').
 *
 * \note This function should be called before hwloc_topology_load().
 *
 * \note The topology-specific userdata pointer is ignored when importing from XML.
 */
HWLOC_DECLSPEC void hwloc_topology_set_userdata_import_callback(hwloc_topology_t topology,
								void (*import_cb)(hwloc_topology_t topology, hwloc_obj_t obj, const char *name, const void *buffer, size_t length));

/** @} */


/** \defgroup hwlocality_syntheticexport Exporting Topologies to Synthetic
 * @{
 */

/** \brief Flags for exporting synthetic topologies.
 *
 * Flags to be given as a OR'ed set to hwloc_topology_export_synthetic().
 */
enum hwloc_topology_export_synthetic_flags_e {
 /** \brief Export extended types such as L2dcache as basic types such as Cache.
  *
  * This is required if loading the synthetic description with hwloc < 1.9.
  * \hideinitializer
  */
 HWLOC_TOPOLOGY_EXPORT_SYNTHETIC_FLAG_NO_EXTENDED_TYPES = (1UL<<0),

 /** \brief Do not export level attributes.
  *
  * Ignore level attributes such as memory/cache sizes or PU indexes.
  * This is required if loading the synthetic description with hwloc < 1.10.
  * \hideinitializer
  */
 HWLOC_TOPOLOGY_EXPORT_SYNTHETIC_FLAG_NO_ATTRS = (1UL<<1),

 /** \brief Export the memory hierarchy as expected in hwloc 1.x.
  *
  * Instead of attaching memory children to levels, export single NUMA node child
  * as normal intermediate levels, when possible.
  * This is required if loading the synthetic description with hwloc 1.x.
  * However this may fail if some objects have multiple local NUMA nodes.
  * \hideinitializer
  */
 HWLOC_TOPOLOGY_EXPORT_SYNTHETIC_FLAG_V1 = (1UL<<2),

 /** \brief Do not export memory information.
  *
  * Only export the actual hierarchy of normal CPU-side objects and ignore
  * where memory is attached.
  * This is useful for when the hierarchy of CPUs is what really matters,
  * but it behaves as if there was a single machine-wide NUMA node.
  * \hideinitializer
  */
 HWLOC_TOPOLOGY_EXPORT_SYNTHETIC_FLAG_IGNORE_MEMORY = (1UL<<3)
};

/** \brief Export the topology as a synthetic string.
 *
 * At most \p buflen characters will be written in \p buffer,
 * including the terminating \0.
 *
 * This exported string may be given back to hwloc_topology_set_synthetic().
 *
 * \p flags is a OR'ed set of ::hwloc_topology_export_synthetic_flags_e.
 *
 * \return The number of characters that were written,
 * not including the terminating \0.
 *
 * \return -1 if the topology could not be exported,
 * for instance if it is not symmetric.
 *
 * \note I/O and Misc children are ignored, the synthetic string only
 * describes normal children.
 *
 * \note A 1024-byte buffer should be large enough for exporting
 * topologies in the vast majority of cases.
 */
  HWLOC_DECLSPEC int hwloc_topology_export_synthetic(hwloc_topology_t topology, char *buffer, size_t buflen, unsigned long flags);

/** @} */



#ifdef __cplusplus
} /* extern "C" */
#endif


#endif /* HWLOC_EXPORT_H */
