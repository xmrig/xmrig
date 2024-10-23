/*
 * Copyright © 2009, 2011, 2012 CNRS.  All rights reserved.
 * Copyright © 2009-2020 Inria.  All rights reserved.
 * Copyright © 2009, 2011, 2012, 2015 Université Bordeaux.  All rights reserved.
 * Copyright © 2009-2020 Cisco Systems, Inc.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef HWLOC_CONFIGURE_H
#define HWLOC_CONFIGURE_H

#define DECLSPEC_EXPORTS

#define HWLOC_HAVE_MSVC_CPUIDEX 1

/* #undef HAVE_MKSTEMP */

#define HWLOC_HAVE_X86_CPUID 1

/* Define to 1 if the system has the type `CACHE_DESCRIPTOR'. */
#define HAVE_CACHE_DESCRIPTOR 0

/* Define to 1 if the system has the type `CACHE_RELATIONSHIP'. */
#define HAVE_CACHE_RELATIONSHIP 0

/* Define to 1 if you have the `clz' function. */
/* #undef HAVE_CLZ */

/* Define to 1 if you have the `clzl' function. */
/* #undef HAVE_CLZL */

/* Define to 1 if you have the <CL/cl_ext.h> header file. */
/* #undef HAVE_CL_CL_EXT_H */

/* Define to 1 if you have the `cpuset_setaffinity' function. */
/* #undef HAVE_CPUSET_SETAFFINITY */

/* Define to 1 if you have the `cpuset_setid' function. */
/* #undef HAVE_CPUSET_SETID */

/* Define to 1 if we have -lcuda */
/* #undef HAVE_CUDA */

/* Define to 1 if you have the <cuda.h> header file. */
/* #undef HAVE_CUDA_H */

/* Define to 1 if you have the <cuda_runtime_api.h> header file. */
/* #undef HAVE_CUDA_RUNTIME_API_H */

/* Define to 1 if you have the declaration of `CL_DEVICE_TOPOLOGY_AMD', and to
   0 if you don't. */
/* #undef HAVE_DECL_CL_DEVICE_TOPOLOGY_AMD */

/* Define to 1 if you have the declaration of `CTL_HW', and to 0 if you don't.
   */
/* #undef HAVE_DECL_CTL_HW */

/* Define to 1 if you have the declaration of `fabsf', and to 0 if you don't.
   */
#define HAVE_DECL_FABSF 1

/* Define to 1 if you have the declaration of `modff', and to 0 if you don't.
   */
#define HAVE_DECL_MODFF 1

/* Define to 1 if you have the declaration of `HW_NCPU', and to 0 if you
   don't. */
/* #undef HAVE_DECL_HW_NCPU */

/* Define to 1 if you have the declaration of
   `nvmlDeviceGetMaxPcieLinkGeneration', and to 0 if you don't. */
/* #undef HAVE_DECL_NVMLDEVICEGETMAXPCIELINKGENERATION */

/* Define to 1 if you have the declaration of `pthread_getaffinity_np', and to
   0 if you don't. */
#define HAVE_DECL_PTHREAD_GETAFFINITY_NP 0

/* Define to 1 if you have the declaration of `pthread_setaffinity_np', and to
   0 if you don't. */
#define HAVE_DECL_PTHREAD_SETAFFINITY_NP 0

/* Define to 1 if you have the declaration of `strtoull', and to 0 if you
   don't. */
#define HAVE_DECL_STRTOULL 0

/* Define to 1 if you have the declaration of `strcasecmp', and to 0 if you
   don't. */
/* #undef HWLOC_HAVE_DECL_STRCASECMP */

/* Define to 1 if you have the declaration of `snprintf', and to 0 if you
   don't. */
#define HAVE_DECL_SNPRINTF 0

/* Define to 1 if you have the declaration of `_strdup', and to 0 if you
   don't. */
#define HAVE_DECL__STRDUP 1

/* Define to 1 if you have the declaration of `_putenv', and to 0 if you
   don't. */
#define HAVE_DECL__PUTENV 1

/* Define to 1 if you have the declaration of `_SC_LARGE_PAGESIZE', and to 0
   if you don't. */
#define HAVE_DECL__SC_LARGE_PAGESIZE 0

/* Define to 1 if you have the declaration of `_SC_NPROCESSORS_CONF', and to 0
   if you don't. */
#define HAVE_DECL__SC_NPROCESSORS_CONF 0

/* Define to 1 if you have the declaration of `_SC_NPROCESSORS_ONLN', and to 0
   if you don't. */
#define HAVE_DECL__SC_NPROCESSORS_ONLN 0

/* Define to 1 if you have the declaration of `_SC_NPROC_CONF', and to 0 if
   you don't. */
#define HAVE_DECL__SC_NPROC_CONF 0

/* Define to 1 if you have the declaration of `_SC_NPROC_ONLN', and to 0 if
   you don't. */
#define HAVE_DECL__SC_NPROC_ONLN 0

/* Define to 1 if you have the declaration of `_SC_PAGESIZE', and to 0 if you
   don't. */
#define HAVE_DECL__SC_PAGESIZE 0

/* Define to 1 if you have the declaration of `_SC_PAGE_SIZE', and to 0 if you
   don't. */
#define HAVE_DECL__SC_PAGE_SIZE 0

/* Define to 1 if you have the <dirent.h> header file. */
/* #undef HAVE_DIRENT_H */

/* Define to 1 if you have the <dlfcn.h> header file. */
/* #undef HAVE_DLFCN_H */

/* Define to 1 if you have the `ffs' function. */
/* #undef HAVE_FFS */

/* Define to 1 if you have the `ffsl' function. */
/* #undef HAVE_FFSL */

/* Define to 1 if you have the `fls' function. */
/* #undef HAVE_FLS */

/* Define to 1 if you have the `flsl' function. */
/* #undef HAVE_FLSL */

/* Define to 1 if you have the `getpagesize' function. */
#define HAVE_GETPAGESIZE 1

/* Define to 1 if the system has the type `GROUP_AFFINITY'. */
#define HAVE_GROUP_AFFINITY 1

/* Define to 1 if the system has the type `GROUP_RELATIONSHIP'. */
#define HAVE_GROUP_RELATIONSHIP 1

/* Define to 1 if you have the `host_info' function. */
/* #undef HAVE_HOST_INFO */

/* Define to 1 if you have the <infiniband/verbs.h> header file. */
/* #undef HAVE_INFINIBAND_VERBS_H */

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define to 1 if the system has the type `KAFFINITY'. */
#define HAVE_KAFFINITY 1

/* Define to 1 if you have the <kstat.h> header file. */
/* #undef HAVE_KSTAT_H */

/* Define to 1 if you have the <langinfo.h> header file. */
/* #undef HAVE_LANGINFO_H */

/* Define to 1 if we have -lgdi32 */
#define HAVE_LIBGDI32 1

/* Define to 1 if we have -libverbs */
/* #undef HAVE_LIBIBVERBS */

/* Define to 1 if we have -lkstat */
/* #undef HAVE_LIBKSTAT */

/* Define to 1 if we have -llgrp */
/* #undef HAVE_LIBLGRP */

/* Define to 1 if you have the <locale.h> header file. */
#define HAVE_LOCALE_H 1

/* Define to 1 if the system has the type `LOGICAL_PROCESSOR_RELATIONSHIP'. */
#define HAVE_LOGICAL_PROCESSOR_RELATIONSHIP 1

/* Define to 1 if you have the <mach/mach_host.h> header file. */
/* #undef HAVE_MACH_MACH_HOST_H */

/* Define to 1 if you have the <mach/mach_init.h> header file. */
/* #undef HAVE_MACH_MACH_INIT_H */

/* Define to 1 if you have the <malloc.h> header file. */
#define HAVE_MALLOC_H 1

/* Define to 1 if you have the `memalign' function. */
/* #undef HAVE_MEMALIGN */

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* Define to 1 if you have the `nl_langinfo' function. */
/* #undef HAVE_NL_LANGINFO */

/* Define to 1 if you have the <numaif.h> header file. */
/* #undef HAVE_NUMAIF_H */

/* Define to 1 if the system has the type `NUMA_NODE_RELATIONSHIP'. */
#define HAVE_NUMA_NODE_RELATIONSHIP 1

/* Define to 1 if you have the <NVCtrl/NVCtrl.h> header file. */
/* #undef HAVE_NVCTRL_NVCTRL_H */

/* Define to 1 if you have the <nvml.h> header file. */
/* #undef HAVE_NVML_H */

/* Define to 1 if you have the `openat' function. */
/* #undef HAVE_OPENAT */

/* Define to 1 if you have the <picl.h> header file. */
/* #undef HAVE_PICL_H */

/* Define to 1 if you have the `posix_memalign' function. */
/* #undef HAVE_POSIX_MEMALIGN */

/* Define to 1 if the system has the type `PROCESSOR_CACHE_TYPE'. */
#define HAVE_PROCESSOR_CACHE_TYPE 1

/* Define to 1 if the system has the type `PROCESSOR_GROUP_INFO'. */
#define HAVE_PROCESSOR_GROUP_INFO 1

/* Define to 1 if the system has the type `PROCESSOR_RELATIONSHIP'. */
#define HAVE_PROCESSOR_RELATIONSHIP 1

/* Define to 1 if the system has the type `PSAPI_WORKING_SET_EX_BLOCK'. */
/* #undef HAVE_PSAPI_WORKING_SET_EX_BLOCK */

/* Define to 1 if the system has the type `PSAPI_WORKING_SET_EX_INFORMATION'.
   */
/* #undef HAVE_PSAPI_WORKING_SET_EX_INFORMATION */

/* Define to 1 if the system has the type `PROCESSOR_NUMBER'. */
#define HAVE_PROCESSOR_NUMBER 1

/* Define to 1 if you have the <pthread_np.h> header file. */
/* #undef HAVE_PTHREAD_NP_H */

/* Define to 1 if the system has the type `pthread_t'. */
/* #undef HAVE_PTHREAD_T */
#undef HAVE_PTHREAD_T

/* Define to 1 if you have the `putwc' function. */
#define HAVE_PUTWC 1

/* Define to 1 if the system has the type `RelationProcessorPackage'. */
/* #undef HAVE_RELATIONPROCESSORPACKAGE */

/* Define to 1 if you have the `setlocale' function. */
#define HAVE_SETLOCALE 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the `strftime' function. */
#define HAVE_STRFTIME 1

/* Define to 1 if you have the <strings.h> header file. */
/* #define HAVE_STRINGS_H 1*/
#undef HAVE_STRINGS_H

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the `strncasecmp' function. */
/* #undef HAVE_STRNCASECMP */

/* Define to '1' if sysctl is present and usable */
/* #undef HAVE_SYSCTL */

/* Define to '1' if sysctlbyname is present and usable */
/* #undef HAVE_SYSCTLBYNAME */

/* Define to 1 if the system has the type
   `SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX'. */
#define HAVE_SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX 1

/* Define to 1 if you have the <sys/cpuset.h> header file. */
/* #undef HAVE_SYS_CPUSET_H */

/* Define to 1 if you have the <sys/lgrp_user.h> header file. */
/* #undef HAVE_SYS_LGRP_USER_H */

/* Define to 1 if you have the <sys/mman.h> header file. */
/* #undef HAVE_SYS_MMAN_H */

/* Define to 1 if you have the <sys/param.h> header file. */
/* #define HAVE_SYS_PARAM_H 1 */
#undef HAVE_SYS_PARAM_H

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/sysctl.h> header file. */
/* #undef HAVE_SYS_SYSCTL_H */

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <sys/utsname.h> header file. */
/* #undef HAVE_SYS_UTSNAME_H */

/* Define to 1 if you have the `uname' function. */
/* #undef HAVE_UNAME */

/* Define to 1 if you have the <unistd.h> header file. */
/* #undef HAVE_UNISTD_H */

/* Define to 1 if you have the `uselocale' function. */
/* #undef HAVE_USELOCALE */

/* Define to 1 if the system has the type `wchar_t'. */
#define HAVE_WCHAR_T 1

/* Define to 1 if you have the <X11/keysym.h> header file. */
/* #undef HAVE_X11_KEYSYM_H */

/* Define to 1 if you have the <X11/Xlib.h> header file. */
/* #undef HAVE_X11_XLIB_H */

/* Define to 1 if you have the <X11/Xutil.h> header file. */
/* #undef HAVE_X11_XUTIL_H */

/* Define to 1 if you have the <xlocale.h> header file. */
/* #undef HAVE_XLOCALE_H */

/* Define to 1 on AIX */
/* #undef HWLOC_AIX_SYS */

/* Define to 1 on BlueGene/Q */
/* #undef HWLOC_BGQ_SYS */

/* Whether C compiler supports symbol visibility or not */
#define HWLOC_C_HAVE_VISIBILITY 0

/* Define to 1 on Darwin */
/* #undef HWLOC_DARWIN_SYS */

/* Whether we are in debugging mode or not */
/* #undef HWLOC_DEBUG */

/* Define to 1 on *FREEBSD */
/* #undef HWLOC_FREEBSD_SYS */

/* Whether your compiler has __attribute__ or not */
/* #define HWLOC_HAVE_ATTRIBUTE 1 */
#undef HWLOC_HAVE_ATTRIBUTE

/* Whether your compiler has __attribute__ aligned or not */
/* #define HWLOC_HAVE_ATTRIBUTE_ALIGNED 1 */

/* Whether your compiler has __attribute__ always_inline or not */
/* #define HWLOC_HAVE_ATTRIBUTE_ALWAYS_INLINE 1 */

/* Whether your compiler has __attribute__ cold or not */
/* #define HWLOC_HAVE_ATTRIBUTE_COLD 1 */

/* Whether your compiler has __attribute__ const or not */
/* #define HWLOC_HAVE_ATTRIBUTE_CONST 1 */

/* Whether your compiler has __attribute__ deprecated or not */
/* #define HWLOC_HAVE_ATTRIBUTE_DEPRECATED 1 */

/* Whether your compiler has __attribute__ format or not */
/* #define HWLOC_HAVE_ATTRIBUTE_FORMAT 1 */

/* Whether your compiler has __attribute__ hot or not */
/* #define HWLOC_HAVE_ATTRIBUTE_HOT 1 */

/* Whether your compiler has __attribute__ malloc or not */
/* #define HWLOC_HAVE_ATTRIBUTE_MALLOC 1 */

/* Whether your compiler has __attribute__ may_alias or not */
/* #define HWLOC_HAVE_ATTRIBUTE_MAY_ALIAS 1 */

/* Whether your compiler has __attribute__ nonnull or not */
/* #define HWLOC_HAVE_ATTRIBUTE_NONNULL 1 */

/* Whether your compiler has __attribute__ noreturn or not */
/* #define HWLOC_HAVE_ATTRIBUTE_NORETURN 1 */

/* Whether your compiler has __attribute__ no_instrument_function or not */
/* #define HWLOC_HAVE_ATTRIBUTE_NO_INSTRUMENT_FUNCTION 1 */

/* Whether your compiler has __attribute__ packed or not */
/* #define HWLOC_HAVE_ATTRIBUTE_PACKED 1 */

/* Whether your compiler has __attribute__ pure or not */
/* #define HWLOC_HAVE_ATTRIBUTE_PURE 1 */

/* Whether your compiler has __attribute__ sentinel or not */
/* #define HWLOC_HAVE_ATTRIBUTE_SENTINEL 1 */

/* Whether your compiler has __attribute__ unused or not */
/* #define HWLOC_HAVE_ATTRIBUTE_UNUSED 1 */

/* Whether your compiler has __attribute__ warn unused result or not */
/* #define HWLOC_HAVE_ATTRIBUTE_WARN_UNUSED_RESULT 1 */

/* Whether your compiler has __attribute__ weak alias or not */
/* #define HWLOC_HAVE_ATTRIBUTE_WEAK_ALIAS 1 */

/* Define to 1 if your `ffs' function is known to be broken. */
/* #undef HWLOC_HAVE_BROKEN_FFS */

/* Define to 1 if you have the `cairo' library. */
/* #undef HWLOC_HAVE_CAIRO */

/* Define to 1 if you have the `clz' function. */
/* #undef HWLOC_HAVE_CLZ */

/* Define to 1 if you have the `clzl' function. */
/* #undef HWLOC_HAVE_CLZL */

/* Define to 1 if you have cpuid */
/* #undef HWLOC_HAVE_CPUID */

/* Define to 1 if the CPU_SET macro works */
/* #undef HWLOC_HAVE_CPU_SET */

/* Define to 1 if the CPU_SET_S macro works */
/* #undef HWLOC_HAVE_CPU_SET_S */

/* Define to 1 if you have the `cudart' SDK. */
/* #undef HWLOC_HAVE_CUDART */

/* Define to 1 if function `clz' is declared by system headers */
/* #undef HWLOC_HAVE_DECL_CLZ */

/* Define to 1 if function `clzl' is declared by system headers */
/* #undef HWLOC_HAVE_DECL_CLZL */

/* Define to 1 if function `ffs' is declared by system headers */
/* #undef HWLOC_HAVE_DECL_FFS */

/* Define to 1 if function `ffsl' is declared by system headers */
/* #undef HWLOC_HAVE_DECL_FFSL */

/* Define to 1 if function `fls' is declared by system headers */
/* #undef HWLOC_HAVE_DECL_FLS */

/* Define to 1 if function `flsl' is declared by system headers */
/* #undef HWLOC_HAVE_DECL_FLSL */

/* Define to 1 if you have the `ffs' function. */
/* #undef HWLOC_HAVE_FFS */

/* Define to 1 if you have the `ffsl' function. */
/* #undef HWLOC_HAVE_FFSL */

/* Define to 1 if you have the `fls' function. */
/* #undef HWLOC_HAVE_FLS */

/* Define to 1 if you have the `flsl' function. */
/* #undef HWLOC_HAVE_FLSL */

/* Define to 1 if you have the GL module components. */
/* #undef HWLOC_HAVE_GL */

/* Define to 1 if you have a library providing the termcap interface */
/* #undef HWLOC_HAVE_LIBTERMCAP */

/* Define to 1 if you have the `libxml2' library. */
/* #undef HWLOC_HAVE_LIBXML2 */

/* Define to 1 if building the Linux PCI component */
/* #undef HWLOC_HAVE_LINUXPCI */

/* Define to 1 if you have the `NVML' library. */
/* #undef HWLOC_HAVE_NVML */

/* Define to 1 if glibc provides the old prototype (without length) of
   sched_setaffinity() */
/* #undef HWLOC_HAVE_OLD_SCHED_SETAFFINITY */

/* Define to 1 if you have the `OpenCL' library. */
/* #undef HWLOC_HAVE_OPENCL */

/* Define to 1 if the hwloc library should support dynamically-loaded plugins
   */
/* #undef HWLOC_HAVE_PLUGINS */

/* `Define to 1 if you have pthread_getthrds_np' */
/* #undef HWLOC_HAVE_PTHREAD_GETTHRDS_NP */

/* Define to 1 if pthread mutexes are available */
/* #undef HWLOC_HAVE_PTHREAD_MUTEX */

/* Define to 1 if glibc provides a prototype of sched_setaffinity() */
#define HWLOC_HAVE_SCHED_SETAFFINITY 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HWLOC_HAVE_STDINT_H 1

/* Define to 1 if you have the `windows.h' header. */
#define HWLOC_HAVE_WINDOWS_H 1

/* Define to 1 if X11 headers including Xutil.h and keysym.h are available. */
/* #undef HWLOC_HAVE_X11_KEYSYM */

/* Define to 1 if function `syscall' is available */
/* #undef HWLOC_HAVE_SYSCALL */

/* Define to 1 on HP-UX */
/* #undef HWLOC_HPUX_SYS */

/* Define to 1 on Linux */
/* #undef HWLOC_LINUX_SYS */

/* Define to 1 on *NETBSD */
/* #undef HWLOC_NETBSD_SYS */

/* The size of `unsigned int', as computed by sizeof */
#define HWLOC_SIZEOF_UNSIGNED_INT 4

/* The size of `unsigned long', as computed by sizeof */
#define HWLOC_SIZEOF_UNSIGNED_LONG 4

/* Define to 1 on Solaris */
/* #undef HWLOC_SOLARIS_SYS */

/* The hwloc symbol prefix */
#define HWLOC_SYM_PREFIX hwloc_

/* The hwloc symbol prefix in all caps */
#define HWLOC_SYM_PREFIX_CAPS HWLOC_

/* Whether we need to re-define all the hwloc public symbols or not */
#define HWLOC_SYM_TRANSFORM 0

/* Define to 1 on unsupported systems */
/* #undef HWLOC_UNSUPPORTED_SYS */

/* Define to 1 if ncurses works, preferred over curses */
/* #undef HWLOC_USE_NCURSES */

/* Define to 1 on WINDOWS */
#define HWLOC_WIN_SYS 1

/* Define to 1 on x86_32 */
/* #undef HWLOC_X86_32_ARCH */

/* Define to 1 on x86_64 */
#define HWLOC_X86_64_ARCH 1

/* Define to the sub-directory in which libtool stores uninstalled libraries.
   */
#define LT_OBJDIR ".libs/"

/* Name of package */
#define PACKAGE "hwloc"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "https://www.open-mpi.org/projects/hwloc/"

/* Define to the full name of this package. */
#define PACKAGE_NAME "hwloc"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "hwloc"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "hwloc"

/* Define to the home page for this package. */
#define PACKAGE_URL ""

/* Define to the version of this package. */
#define PACKAGE_VERSION HWLOC_VERSION

/* The size of `unsigned int', as computed by sizeof. */
#define SIZEOF_UNSIGNED_INT 4

/* The size of `unsigned long', as computed by sizeof. */
#define SIZEOF_UNSIGNED_LONG 4

/* The size of `void *', as computed by sizeof. */
#define SIZEOF_VOID_P 8

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* Enable extensions on HP-UX. */
#ifndef _HPUX_SOURCE
# define _HPUX_SOURCE 1
#endif


/* Enable extensions on AIX 3, Interix.  */
/*
#ifndef _ALL_SOURCE
# define _ALL_SOURCE 1
#endif
*/

/* Enable GNU extensions on systems that have them.  */
/*
#ifndef _GNU_SOURCE
# define _GNU_SOURCE 1
#endif
*/
/* Enable threading extensions on Solaris.  */
/*
#ifndef _POSIX_PTHREAD_SEMANTICS
# define _POSIX_PTHREAD_SEMANTICS 1
#endif
*/
/* Enable extensions on HP NonStop.  */
/*
#ifndef _TANDEM_SOURCE
# define _TANDEM_SOURCE 1
#endif
*/
/* Enable general extensions on Solaris.  */
/*
#ifndef __EXTENSIONS__
# define __EXTENSIONS__ 1
#endif
*/


/* Version number of package */
#define VERSION HWLOC_VERSION

/* Define to 1 if the X Window System is missing or not being used. */
#define X_DISPLAY_MISSING 1

/* Define to 1 if on MINIX. */
/* #undef _MINIX */

/* Define to 2 if the system does not provide POSIX.1 features except with
   this defined. */
/* #undef _POSIX_1_SOURCE */

/* Define to 1 if you need to in order for `stat' and other things to work. */
/* #undef _POSIX_SOURCE */

/* Define this to the process ID type */
#define hwloc_pid_t HANDLE

/* Define this to either strncasecmp or strncmp */
/* #undef hwloc_strncasecmp */

/* Define this to the thread ID type */
#define hwloc_thread_t HANDLE

/* Define to 1 if you have the declaration of `GetModuleFileName', and to 0 if
   you don't. */
#define HAVE_DECL_GETMODULEFILENAME 1


#endif /* HWLOC_CONFIGURE_H */
