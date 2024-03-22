This is a truncated and poorly-formatted version of the documentation main page.
See https://www.open-mpi.org/projects/hwloc/doc/ for more.


hwloc Overview

The Hardware Locality (hwloc) software project aims at easing the process of
discovering hardware resources in parallel architectures. It offers
command-line tools and a C API for consulting these resources, their locality,
attributes, and interconnection. hwloc primarily aims at helping
high-performance computing (HPC) applications, but is also applicable to any
project seeking to exploit code and/or data locality on modern computing
platforms.

hwloc provides command line tools and a C API to obtain the hierarchical map of
key computing elements within a node, such as: NUMA memory nodes, shared
caches, processor packages, dies and cores, processing units (logical
processors or "threads") and even I/O devices. hwloc also gathers various
attributes such as cache and memory information, and is portable across a
variety of different operating systems and platforms.

hwloc primarily aims at helping high-performance computing (HPC) applications,
but is also applicable to any project seeking to exploit code and/or data
locality on modern computing platforms.

hwloc supports the following operating systems:

  * Linux (with knowledge of cgroups and cpusets, memory targets/initiators,
 etc.) on all supported hardware, including Intel Xeon Phi, ScaleMP vSMP,
 and NumaScale NumaConnect.
  * Solaris (with support for processor sets and logical domains)
  * AIX
  * Darwin / OS X
  * FreeBSD and its variants (such as kFreeBSD/GNU)
  * NetBSD
  * HP-UX
  * Microsoft Windows
  * IBM BlueGene/Q Compute Node Kernel (CNK)

Since it uses standard Operating System information, hwloc's support is mostly
independant from the processor type (x86, powerpc, ...) and just relies on the
Operating System support. The main exception is BSD operating systems (NetBSD,
FreeBSD, etc.) because they do not provide support topology information, hence
hwloc uses an x86-only CPUID-based backend (which can be used for other OSes
too, see the Components and plugins section).

To check whether hwloc works on a particular machine, just try to build it and
run lstopo or lstopo-no-graphics. If some things do not look right (e.g. bogus
or missing cache information), see Questions and Bugs.

hwloc only reports the number of processors on unsupported operating systems;
no topology information is available.

For development and debugging purposes, hwloc also offers the ability to work
on "fake" topologies:

  * Symmetrical tree of resources generated from a list of level arities, see
 Synthetic topologies.
  * Remote machine simulation through the gathering of topology as XML files,
 see Importing and exporting topologies from/to XML files.

hwloc can display the topology in a human-readable format, either in graphical
mode (X11), or by exporting in one of several different formats, including:
plain text, LaTeX tikzpicture, PDF, PNG, and FIG (see Command-line Examples
below). Note that some of the export formats require additional support
libraries.

hwloc offers a programming interface for manipulating topologies and objects.
It also brings a powerful CPU bitmap API that is used to describe topology
objects location on physical/logical processors. See the Programming Interface
below. It may also be used to binding applications onto certain cores or memory
nodes. Several utility programs are also provided to ease command-line
manipulation of topology objects, binding of processes, and so on.

Bindings for several other languages are available from the project website.

Command-line Examples

On a 4-package 2-core machine with hyper-threading, the lstopo tool may show
the following graphical output:

[dudley]

Here's the equivalent output in textual form:

Machine
  NUMANode L#0 (P#0)
  Package L#0 + L3 L#0 (4096KB)
 L2 L#0 (1024KB) + L1 L#0 (16KB) + Core L#0
   PU L#0 (P#0)
   PU L#1 (P#8)
 L2 L#1 (1024KB) + L1 L#1 (16KB) + Core L#1
   PU L#2 (P#4)
   PU L#3 (P#12)
  Package L#1 + L3 L#1 (4096KB)
 L2 L#2 (1024KB) + L1 L#2 (16KB) + Core L#2
   PU L#4 (P#1)
   PU L#5 (P#9)
 L2 L#3 (1024KB) + L1 L#3 (16KB) + Core L#3
   PU L#6 (P#5)
   PU L#7 (P#13)
  Package L#2 + L3 L#2 (4096KB)
 L2 L#4 (1024KB) + L1 L#4 (16KB) + Core L#4
   PU L#8 (P#2)
   PU L#9 (P#10)
 L2 L#5 (1024KB) + L1 L#5 (16KB) + Core L#5
   PU L#10 (P#6)
   PU L#11 (P#14)
  Package L#3 + L3 L#3 (4096KB)
 L2 L#6 (1024KB) + L1 L#6 (16KB) + Core L#6
   PU L#12 (P#3)
   PU L#13 (P#11)
 L2 L#7 (1024KB) + L1 L#7 (16KB) + Core L#7
   PU L#14 (P#7)
   PU L#15 (P#15)

Note that there is also an equivalent output in XML that is meant for exporting
/importing topologies but it is hardly readable to human-beings (see Importing
and exporting topologies from/to XML files for details).

On a 4-package 2-core Opteron NUMA machine (with two core cores disallowed by
the administrator), the lstopo tool may show the following graphical output
(with --disallowed for displaying disallowed objects):

[hagrid]

Here's the equivalent output in textual form:

Machine (32GB total)
  Package L#0
 NUMANode L#0 (P#0 8190MB)
 L2 L#0 (1024KB) + L1 L#0 (64KB) + Core L#0 + PU L#0 (P#0)
 L2 L#1 (1024KB) + L1 L#1 (64KB) + Core L#1 + PU L#1 (P#1)
  Package L#1
 NUMANode L#1 (P#1 8192MB)
 L2 L#2 (1024KB) + L1 L#2 (64KB) + Core L#2 + PU L#2 (P#2)
 L2 L#3 (1024KB) + L1 L#3 (64KB) + Core L#3 + PU L#3 (P#3)
  Package L#2
 NUMANode L#2 (P#2 8192MB)
 L2 L#4 (1024KB) + L1 L#4 (64KB) + Core L#4 + PU L#4 (P#4)
 L2 L#5 (1024KB) + L1 L#5 (64KB) + Core L#5 + PU L#5 (P#5)
  Package L#3
 NUMANode L#3 (P#3 8192MB)
 L2 L#6 (1024KB) + L1 L#6 (64KB) + Core L#6 + PU L#6 (P#6)
 L2 L#7 (1024KB) + L1 L#7 (64KB) + Core L#7 + PU L#7 (P#7)

On a 2-package quad-core Xeon (pre-Nehalem, with 2 dual-core dies into each
package):

[emmett]

Here's the same output in textual form:

Machine (total 16GB)
  NUMANode L#0 (P#0 16GB)
  Package L#0
 L2 L#0 (4096KB)
   L1 L#0 (32KB) + Core L#0 + PU L#0 (P#0)
   L1 L#1 (32KB) + Core L#1 + PU L#1 (P#4)
 L2 L#1 (4096KB)
   L1 L#2 (32KB) + Core L#2 + PU L#2 (P#2)
   L1 L#3 (32KB) + Core L#3 + PU L#3 (P#6)
  Package L#1
 L2 L#2 (4096KB)
   L1 L#4 (32KB) + Core L#4 + PU L#4 (P#1)
   L1 L#5 (32KB) + Core L#5 + PU L#5 (P#5)
 L2 L#3 (4096KB)
   L1 L#6 (32KB) + Core L#6 + PU L#6 (P#3)
   L1 L#7 (32KB) + Core L#7 + PU L#7 (P#7)

Programming Interface

The basic interface is available in hwloc.h. Some higher-level functions are
available in hwloc/helper.h to reduce the need to manually manipulate objects
and follow links between them. Documentation for all these is provided later in
this document. Developers may also want to look at hwloc/inlines.h which
contains the actual inline code of some hwloc.h routines, and at this document,
which provides good higher-level topology traversal examples.

To precisely define the vocabulary used by hwloc, a Terms and Definitions
section is available and should probably be read first.

Each hwloc object contains a cpuset describing the list of processing units
that it contains. These bitmaps may be used for CPU binding and Memory binding.
hwloc offers an extensive bitmap manipulation interface in hwloc/bitmap.h.

Moreover, hwloc also comes with additional helpers for interoperability with
several commonly used environments. See the Interoperability With Other
Software section for details.

The complete API documentation is available in a full set of HTML pages, man
pages, and self-contained PDF files (formatted for both both US letter and A4
formats) in the source tarball in doc/doxygen-doc/.

NOTE: If you are building the documentation from a Git clone, you will need to
have Doxygen and pdflatex installed -- the documentation will be built during
the normal "make" process. The documentation is installed during "make install"
to $prefix/share/doc/hwloc/ and your systems default man page tree (under
$prefix, of course).

Portability

Operating System have varying support for CPU and memory binding, e.g. while
some Operating Systems provide interfaces for all kinds of CPU and memory
bindings, some others provide only interfaces for a limited number of kinds of
CPU and memory binding, and some do not provide any binding interface at all.
Hwloc's binding functions would then simply return the ENOSYS error (Function
not implemented), meaning that the underlying Operating System does not provide
any interface for them. CPU binding and Memory binding provide more information
on which hwloc binding functions should be preferred because interfaces for
them are usually available on the supported Operating Systems.

Similarly, the ability of reporting topology information varies from one
platform to another. As shown in Command-line Examples, hwloc can obtain
information on a wide variety of hardware topologies. However, some platforms
and/or operating system versions will only report a subset of this information.
For example, on an PPC64-based system with 8 cores (each with 2 hardware
threads) running a default 2.6.18-based kernel from RHEL 5.4, hwloc is only
able to glean information about NUMA nodes and processor units (PUs). No
information about caches, packages, or cores is available.

Here's the graphical output from lstopo on this platform when Simultaneous
Multi-Threading (SMT) is enabled:

[ppc64-with]

And here's the graphical output from lstopo on this platform when SMT is
disabled:

[ppc64-with]

Notice that hwloc only sees half the PUs when SMT is disabled. PU L#6, for
example, seems to change location from NUMA node #0 to #1. In reality, no PUs
"moved" -- they were simply re-numbered when hwloc only saw half as many (see
also Logical index in Indexes and Sets). Hence, PU L#6 in the SMT-disabled
picture probably corresponds to PU L#12 in the SMT-enabled picture.

This same "PUs have disappeared" effect can be seen on other platforms -- even
platforms / OSs that provide much more information than the above PPC64 system.
This is an unfortunate side-effect of how operating systems report information
to hwloc.

Note that upgrading the Linux kernel on the same PPC64 system mentioned above
to 2.6.34, hwloc is able to discover all the topology information. The
following picture shows the entire topology layout when SMT is enabled:

[ppc64-full]

Developers using the hwloc API or XML output for portable applications should
therefore be extremely careful to not make any assumptions about the structure
of data that is returned. For example, per the above reported PPC topology, it
is not safe to assume that PUs will always be descendants of cores.

Additionally, future hardware may insert new topology elements that are not
available in this version of hwloc. Long-lived applications that are meant to
span multiple different hardware platforms should also be careful about making
structure assumptions. For example, a new element may someday exist between a
core and a PU.

API Example

The following small C example (available in the source tree as ``doc/examples/
hwloc-hello.c'') prints the topology of the machine and performs some thread
and memory binding. More examples are available in the doc/examples/ directory
of the source tree.

/* Example hwloc API program.
*
* See other examples under doc/examples/ in the source tree
* for more details.
*
* Copyright (c) 2009-2016 Inria. All rights reserved.
* Copyright (c) 2009-2011 Universit?eacute; Bordeaux
* Copyright (c) 2009-2010 Cisco Systems, Inc. All rights reserved.
* See COPYING in top-level directory.
*
* hwloc-hello.c
*/
#include "hwloc.h"
#include <errno.h>
#include <stdio.h>
#include <string.h>
static void print_children(hwloc_topology_t topology, hwloc_obj_t obj,
int depth)
{
char type[32], attr[1024];
unsigned i;
hwloc_obj_type_snprintf(type, sizeof(type), obj, 0);
printf("%*s%s", 2*depth, "", type);
if (obj->os_index != (unsigned) -1)
printf("#%u", obj->os_index);
hwloc_obj_attr_snprintf(attr, sizeof(attr), obj, " ", 0);
if (*attr)
printf("(%s)", attr);
printf("\n");
for (i = 0; i < obj->arity; i++) {
print_children(topology, obj->children[i], depth + 1);
}
}
int main(void)
{
int depth;
unsigned i, n;
unsigned long size;
int levels;
char string[128];
int topodepth;
void *m;
hwloc_topology_t topology;
hwloc_cpuset_t cpuset;
hwloc_obj_t obj;
/* Allocate and initialize topology object. */
hwloc_topology_init(&topology);
/* ... Optionally, put detection configuration here to ignore
some objects types, define a synthetic topology, etc....
The default is to detect all the objects of the machine that
the caller is allowed to access. See Configure Topology
Detection. */
/* Perform the topology detection. */
hwloc_topology_load(topology);
/* Optionally, get some additional topology information
in case we need the topology depth later. */
topodepth = hwloc_topology_get_depth(topology);
/*****************************************************************
* First example:
* Walk the topology with an array style, from level 0 (always
* the system level) to the lowest level (always the proc level).
*****************************************************************/
for (depth = 0; depth < topodepth; depth++) {
printf("*** Objects at level %d\n", depth);
for (i = 0; i < hwloc_get_nbobjs_by_depth(topology, depth);
i++) {
hwloc_obj_type_snprintf(string, sizeof(string),
hwloc_get_obj_by_depth(topology, depth, i), 0);
printf("Index %u: %s\n", i, string);
}
}
/*****************************************************************
* Second example:
* Walk the topology with a tree style.
*****************************************************************/
printf("*** Printing overall tree\n");
print_children(topology, hwloc_get_root_obj(topology), 0);
/*****************************************************************
* Third example:
* Print the number of packages.
*****************************************************************/
depth = hwloc_get_type_depth(topology, HWLOC_OBJ_PACKAGE);
if (depth == HWLOC_TYPE_DEPTH_UNKNOWN) {
printf("*** The number of packages is unknown\n");
} else {
printf("*** %u package(s)\n",
hwloc_get_nbobjs_by_depth(topology, depth));
}
/*****************************************************************
* Fourth example:
* Compute the amount of cache that the first logical processor
* has above it.
*****************************************************************/
levels = 0;
size = 0;
for (obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PU, 0);
obj;
obj = obj->parent)
if (hwloc_obj_type_is_cache(obj->type)) {
levels++;
size += obj->attr->cache.size;
}
printf("*** Logical processor 0 has %d caches totaling %luKB\n",
levels, size / 1024);
/*****************************************************************
* Fifth example:
* Bind to only one thread of the last core of the machine.
*
* First find out where cores are, or else smaller sets of CPUs if
* the OS doesn't have the notion of a "core".
*****************************************************************/
depth = hwloc_get_type_or_below_depth(topology, HWLOC_OBJ_CORE);
/* Get last core. */
obj = hwloc_get_obj_by_depth(topology, depth,
hwloc_get_nbobjs_by_depth(topology, depth) - 1);
if (obj) {
/* Get a copy of its cpuset that we may modify. */
cpuset = hwloc_bitmap_dup(obj->cpuset);
/* Get only one logical processor (in case the core is
SMT/hyper-threaded). */
hwloc_bitmap_singlify(cpuset);
/* And try to bind ourself there. */
if (hwloc_set_cpubind(topology, cpuset, 0)) {
char *str;
int error = errno;
hwloc_bitmap_asprintf(&str, obj->cpuset);
printf("Couldn't bind to cpuset %s: %s\n", str, strerror(error));
free(str);
}
/* Free our cpuset copy */
hwloc_bitmap_free(cpuset);
}
/*****************************************************************
* Sixth example:
* Allocate some memory on the last NUMA node, bind some existing
* memory to the last NUMA node.
*****************************************************************/
/* Get last node. There's always at least one. */
n = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_NUMANODE);
obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_NUMANODE, n - 1);
size = 1024*1024;
m = hwloc_alloc_membind(topology, size, obj->nodeset,
HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_BYNODESET);
hwloc_free(topology, m, size);
m = malloc(size);
hwloc_set_area_membind(topology, m, size, obj->nodeset,
HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_BYNODESET);
free(m);
/* Destroy topology object. */
hwloc_topology_destroy(topology);
return 0;
}

hwloc provides a pkg-config executable to obtain relevant compiler and linker
flags. For example, it can be used thusly to compile applications that utilize
the hwloc library (assuming GNU Make):

CFLAGS += $(shell pkg-config --cflags hwloc)
LDLIBS += $(shell pkg-config --libs hwloc)

hwloc-hello: hwloc-hello.c
     $(CC) hwloc-hello.c $(CFLAGS) -o hwloc-hello $(LDLIBS)

On a machine 2 processor packages -- each package of which has two processing
cores -- the output from running hwloc-hello could be something like the
following:

shell$ ./hwloc-hello
*** Objects at level 0
Index 0: Machine
*** Objects at level 1
Index 0: Package#0
Index 1: Package#1
*** Objects at level 2
Index 0: Core#0
Index 1: Core#1
Index 2: Core#3
Index 3: Core#2
*** Objects at level 3
Index 0: PU#0
Index 1: PU#1
Index 2: PU#2
Index 3: PU#3
*** Printing overall tree
Machine
  Package#0
 Core#0
   PU#0
 Core#1
   PU#1
  Package#1
 Core#3
   PU#2
 Core#2
   PU#3
*** 2 package(s)
*** Logical processor 0 has 0 caches totaling 0KB
shell$

Questions and Bugs

Bugs should be reported in the tracker (https://github.com/open-mpi/hwloc/
issues). Opening a new issue automatically displays lots of hints about how to
debug and report issues.

Questions may be sent to the users or developers mailing lists (https://
www.open-mpi.org/community/lists/hwloc.php).

There is also a #hwloc IRC channel on Libera Chat (irc.libera.chat).

History / Credits

hwloc is the evolution and merger of the libtopology project and the Portable
Linux Processor Affinity (PLPA) (https://www.open-mpi.org/projects/plpa/)
project. Because of functional and ideological overlap, these two code bases
and ideas were merged and released under the name "hwloc" as an Open MPI
sub-project.

libtopology was initially developed by the Inria Runtime Team-Project. PLPA was
initially developed by the Open MPI development team as a sub-project. Both are
now deprecated in favor of hwloc, which is distributed as an Open MPI
sub-project.



See https://www.open-mpi.org/projects/hwloc/doc/ for more hwloc documentation,
actual links to related pages, images, etc.
