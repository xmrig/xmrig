/*
 * Copyright © 2012 Blue Brain Project, EPFL. All rights reserved.
 * Copyright © 2012-2013 Inria.  All rights reserved.
 * See COPYING in top-level directory.
 */

/** \file
 * \brief Macros to help interaction between hwloc and OpenGL displays.
 *
 * Applications that use both hwloc and OpenGL may want to include
 * this file so as to get topology information for OpenGL displays.
 */

#ifndef HWLOC_GL_H
#define HWLOC_GL_H

#include "hwloc.h"

#include <stdio.h>
#include <string.h>


#ifdef __cplusplus
extern "C" {
#endif


/** \defgroup hwlocality_gl Interoperability with OpenGL displays
 *
 * This interface offers ways to retrieve topology information about
 * OpenGL displays.
 *
 * Only the NVIDIA display locality information is currently available,
 * using the NV-CONTROL X11 extension and the NVCtrl library.
 *
 * @{
 */

/** \brief Get the hwloc OS device object corresponding to the
 * OpenGL display given by port and device index.
 *
 * Return the OS device object describing the OpenGL display
 * whose port (server) is \p port and device (screen) is \p device.
 * Return NULL if there is none.
 *
 * The topology \p topology does not necessarily have to match the current
 * machine. For instance the topology may be an XML import of a remote host.
 * I/O devices detection and the GL component must be enabled in the topology.
 *
 * \note The corresponding PCI device object can be obtained by looking
 * at the OS device parent object (unless PCI devices are filtered out).
 */
static __hwloc_inline hwloc_obj_t
hwloc_gl_get_display_osdev_by_port_device(hwloc_topology_t topology,
					  unsigned port, unsigned device)
{
        unsigned x = (unsigned) -1, y = (unsigned) -1;
        hwloc_obj_t osdev = NULL;
        while ((osdev = hwloc_get_next_osdev(topology, osdev)) != NULL) {
                if (HWLOC_OBJ_OSDEV_GPU == osdev->attr->osdev.type
                    && osdev->name
                    && sscanf(osdev->name, ":%u.%u", &x, &y) == 2
                    && port == x && device == y)
                        return osdev;
        }
	errno = EINVAL;
        return NULL;
}

/** \brief Get the hwloc OS device object corresponding to the
 * OpenGL display given by name.
 *
 * Return the OS device object describing the OpenGL display
 * whose name is \p name, built as ":port.device" such as ":0.0" .
 * Return NULL if there is none.
 *
 * The topology \p topology does not necessarily have to match the current
 * machine. For instance the topology may be an XML import of a remote host.
 * I/O devices detection and the GL component must be enabled in the topology.
 *
 * \note The corresponding PCI device object can be obtained by looking
 * at the OS device parent object (unless PCI devices are filtered out).
 */
static __hwloc_inline hwloc_obj_t
hwloc_gl_get_display_osdev_by_name(hwloc_topology_t topology,
				   const char *name)
{
        hwloc_obj_t osdev = NULL;
        while ((osdev = hwloc_get_next_osdev(topology, osdev)) != NULL) {
                if (HWLOC_OBJ_OSDEV_GPU == osdev->attr->osdev.type
                    && osdev->name
                    && !strcmp(name, osdev->name))
                        return osdev;
        }
	errno = EINVAL;
        return NULL;
}

/** \brief Get the OpenGL display port and device corresponding
 * to the given hwloc OS object.
 *
 * Return the OpenGL display port (server) in \p port and device (screen)
 * in \p screen that correspond to the given hwloc OS device object.
 * Return \c -1 if there is none.
 *
 * The topology \p topology does not necessarily have to match the current
 * machine. For instance the topology may be an XML import of a remote host.
 * I/O devices detection and the GL component must be enabled in the topology.
 */
static __hwloc_inline int
hwloc_gl_get_display_by_osdev(hwloc_topology_t topology __hwloc_attribute_unused,
			      hwloc_obj_t osdev,
			      unsigned *port, unsigned *device)
{
	unsigned x = -1, y = -1;
	if (HWLOC_OBJ_OSDEV_GPU == osdev->attr->osdev.type
	    && sscanf(osdev->name, ":%u.%u", &x, &y) == 2) {
		*port = x;
		*device = y;
		return 0;
	}
	errno = EINVAL;
	return -1;
}

/** @} */


#ifdef __cplusplus
} /* extern "C" */
#endif


#endif /* HWLOC_GL_H */

