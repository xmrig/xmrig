/*
 * Copyright © 2018-2020 Inria.  All rights reserved.
 *
 * See COPYING in top-level directory.
 */

/* List of components defined inside hwloc */

#ifndef PRIVATE_INTERNAL_COMPONENTS_H
#define PRIVATE_INTERNAL_COMPONENTS_H

/* global discovery */
HWLOC_DECLSPEC extern const struct hwloc_component hwloc_xml_component;
HWLOC_DECLSPEC extern const struct hwloc_component hwloc_synthetic_component;

/* CPU discovery */
HWLOC_DECLSPEC extern const struct hwloc_component hwloc_aix_component;
HWLOC_DECLSPEC extern const struct hwloc_component hwloc_bgq_component;
HWLOC_DECLSPEC extern const struct hwloc_component hwloc_darwin_component;
HWLOC_DECLSPEC extern const struct hwloc_component hwloc_freebsd_component;
HWLOC_DECLSPEC extern const struct hwloc_component hwloc_hpux_component;
HWLOC_DECLSPEC extern const struct hwloc_component hwloc_linux_component;
HWLOC_DECLSPEC extern const struct hwloc_component hwloc_netbsd_component;
HWLOC_DECLSPEC extern const struct hwloc_component hwloc_noos_component;
HWLOC_DECLSPEC extern const struct hwloc_component hwloc_solaris_component;
HWLOC_DECLSPEC extern const struct hwloc_component hwloc_windows_component;
HWLOC_DECLSPEC extern const struct hwloc_component hwloc_x86_component;

/* I/O discovery */
HWLOC_DECLSPEC extern const struct hwloc_component hwloc_cuda_component;
HWLOC_DECLSPEC extern const struct hwloc_component hwloc_gl_component;
HWLOC_DECLSPEC extern const struct hwloc_component hwloc_nvml_component;
HWLOC_DECLSPEC extern const struct hwloc_component hwloc_rsmi_component;
HWLOC_DECLSPEC extern const struct hwloc_component hwloc_levelzero_component;
HWLOC_DECLSPEC extern const struct hwloc_component hwloc_opencl_component;
HWLOC_DECLSPEC extern const struct hwloc_component hwloc_pci_component;

/* XML backend */
HWLOC_DECLSPEC extern const struct hwloc_component hwloc_xml_nolibxml_component;
HWLOC_DECLSPEC extern const struct hwloc_component hwloc_xml_libxml_component;

#endif /* PRIVATE_INTERNAL_COMPONENTS_H */
