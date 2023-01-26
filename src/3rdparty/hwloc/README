Introduction

The Hardware Locality (hwloc) software project aims at easing the process of
discovering hardware resources in parallel architectures. It offers
command-line tools and a C API for consulting these resources, their locality,
attributes, and interconnection. hwloc primarily aims at helping
high-performance computing (HPC) applications, but is also applicable to any
project seeking to exploit code and/or data locality on modern computing
platforms.

hwloc is actually made of two subprojects distributed together:

  * The original hwloc project for describing the internals of computing nodes.
 It is described in details starting at section Hardware Locality (hwloc)
 Introduction.
  * The network-oriented companion called netloc (Network Locality), described
 in details starting with section Network Locality (netloc).

See also the Related pages tab above for links to other sections.

Netloc may be disabled, but the original hwloc cannot. Both hwloc and netloc
APIs are documented after these sections.

Installation

hwloc (https://www.open-mpi.org/projects/hwloc/) is available under the BSD
license. It is hosted as a sub-project of the overall Open MPI project (https:/
/www.open-mpi.org/). Note that hwloc does not require any functionality from
Open MPI -- it is a wholly separate (and much smaller!) project and code base.
It just happens to be hosted as part of the overall Open MPI project.

Basic Installation

Installation is the fairly common GNU-based process:

shell$ ./configure --prefix=...
shell$ make
shell$ make install

hwloc- and netloc-specific configure options and requirements are documented in
sections hwloc Installation and Netloc Installation respectively.

Also note that if you install supplemental libraries in non-standard locations,
hwloc's configure script may not be able to find them without some help. You
may need to specify additional CPPFLAGS, LDFLAGS, or PKG_CONFIG_PATH values on
the configure command line.

For example, if libpciaccess was installed into /opt/pciaccess, hwloc's
configure script may not find it be default. Try adding PKG_CONFIG_PATH to the
./configure command line, like this:

./configure PKG_CONFIG_PATH=/opt/pciaccess/lib/pkgconfig ...

Running the "lstopo" tool is a good way to check as a graphical output whether
hwloc properly detected the architecture of your node. Netloc command-line
tools can be used to display the network topology interconnecting your nodes.

Installing from a Git clone

Additionally, the code can be directly cloned from Git:

shell$ git clone https://github.com/open-mpi/hwloc.git
shell$ cd hwloc
shell$ ./autogen.sh

Note that GNU Autoconf >=2.63, Automake >=1.11 and Libtool >=2.2.6 are required
when building from a Git clone.

Nightly development snapshots are available on the web site, they can be
configured and built without any need for Git or GNU Autotools.

Questions and Bugs

Bugs should be reported in the tracker (https://github.com/open-mpi/hwloc/
issues). Opening a new issue automatically displays lots of hints about how to
debug and report issues.

Questions may be sent to the users or developers mailing lists (https://
www.open-mpi.org/community/lists/hwloc.php).

There is also a #hwloc IRC channel on Libera Chat (irc.libera.chat).



See https://www.open-mpi.org/projects/hwloc/doc/ for more hwloc documentation.
