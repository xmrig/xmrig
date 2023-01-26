/*
 * Copyright © 2009-2022 Inria.  All rights reserved.
 * Copyright © 2012 Université Bordeaux
 * See COPYING in top-level directory.
 */

#include "private/autogen/config.h"
#include "hwloc.h"
#include "private/private.h"
#include "private/xml.h"
#include "private/misc.h"

#define HWLOC_COMPONENT_STOP_NAME "stop"
#define HWLOC_COMPONENT_EXCLUDE_CHAR '-'
#define HWLOC_COMPONENT_SEPS ","
#define HWLOC_COMPONENT_PHASESEP_CHAR ':'

/* list of all registered discovery components, sorted by priority, higher priority first.
 * noos is last because its priority is 0.
 * others' priority is 10.
 */
static struct hwloc_disc_component * hwloc_disc_components = NULL;

static unsigned hwloc_components_users = 0; /* first one initializes, last ones destroys */

static int hwloc_components_verbose = 0;
#ifdef HWLOC_HAVE_PLUGINS
static int hwloc_plugins_verbose = 0;
static const char * hwloc_plugins_blacklist = NULL;
#endif

/* hwloc_components_mutex serializes:
 * - loading/unloading plugins, and modifications of the hwloc_plugins list
 * - calls to ltdl, including in hwloc_check_plugin_namespace()
 * - registration of components with hwloc_disc_component_register()
 *   and hwloc_xml_callbacks_register()
 */
#ifdef HWLOC_WIN_SYS
/* Basic mutex on top of InterlockedCompareExchange() on windows,
 * Far from perfect, but easy to maintain, and way enough given that this code will never be needed for real. */
#include <windows.h>
static LONG hwloc_components_mutex = 0;
#define HWLOC_COMPONENTS_LOCK() do {						\
  while (InterlockedCompareExchange(&hwloc_components_mutex, 1, 0) != 0)	\
    SwitchToThread();								\
} while (0)
#define HWLOC_COMPONENTS_UNLOCK() do {						\
  assert(hwloc_components_mutex == 1);						\
  hwloc_components_mutex = 0;							\
} while (0)

#elif defined HWLOC_HAVE_PTHREAD_MUTEX
/* pthread mutex if available (except on windows) */
#include <pthread.h>
static pthread_mutex_t hwloc_components_mutex = PTHREAD_MUTEX_INITIALIZER;
#define HWLOC_COMPONENTS_LOCK() pthread_mutex_lock(&hwloc_components_mutex)
#define HWLOC_COMPONENTS_UNLOCK() pthread_mutex_unlock(&hwloc_components_mutex)

#else /* HWLOC_WIN_SYS || HWLOC_HAVE_PTHREAD_MUTEX */
#error No mutex implementation available
#endif


#ifdef HWLOC_HAVE_PLUGINS

#ifdef HWLOC_HAVE_LTDL
/* ltdl-based plugin load */
#include <ltdl.h>
typedef lt_dlhandle hwloc_dlhandle;
#define hwloc_dlinit lt_dlinit
#define hwloc_dlexit lt_dlexit
#define hwloc_dlopenext lt_dlopenext
#define hwloc_dlclose lt_dlclose
#define hwloc_dlerror lt_dlerror
#define hwloc_dlsym lt_dlsym
#define hwloc_dlforeachfile lt_dlforeachfile

#else /* !HWLOC_HAVE_LTDL */
/* no-ltdl plugin load relies on less portable libdl */
#include <dlfcn.h>
typedef void * hwloc_dlhandle;
static __hwloc_inline int hwloc_dlinit(void) { return 0; }
static __hwloc_inline int hwloc_dlexit(void) { return 0; }
#define hwloc_dlclose dlclose
#define hwloc_dlerror dlerror
#define hwloc_dlsym dlsym

#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>

static hwloc_dlhandle hwloc_dlopenext(const char *_filename)
{
  hwloc_dlhandle handle;
  char *filename = NULL;
  (void) asprintf(&filename, "%s.so", _filename);
  if (!filename)
    return NULL;
  handle = dlopen(filename, RTLD_NOW|RTLD_LOCAL);
  free(filename);
  return handle;
}

static int
hwloc_dlforeachfile(const char *_paths,
		    int (*func)(const char *filename, void *data),
		    void *data)
{
  char *paths = NULL, *path;

  paths = strdup(_paths);
  if (!paths)
    return -1;

  path = paths;
  while (*path) {
    char *colon;
    DIR *dir;
    struct dirent *dirent;

    colon = strchr(path, ':');
    if (colon)
      *colon = '\0';

    if (hwloc_plugins_verbose)
      fprintf(stderr, "hwloc:  Looking under %s\n", path);

    dir = opendir(path);
    if (!dir)
      goto next;

    while ((dirent = readdir(dir)) != NULL) {
      char *abs_name, *suffix;
      struct stat stbuf;
      int err;

      err = asprintf(&abs_name, "%s/%s", path, dirent->d_name);
      if (err < 0)
	continue;

      err = stat(abs_name, &stbuf);
      if (err < 0) {
	free(abs_name);
        continue;
      }
      if (!S_ISREG(stbuf.st_mode)) {
	free(abs_name);
	continue;
      }

      /* Only keep .so files, and remove that suffix to get the component basename */
      suffix = strrchr(abs_name, '.');
      if (!suffix || strcmp(suffix, ".so")) {
	free(abs_name);
	continue;
      }
      *suffix = '\0';

      err = func(abs_name, data);
      if (err) {
	free(abs_name);
	continue;
      }

      free(abs_name);
    }

    closedir(dir);

  next:
    if (!colon)
      break;
    path = colon+1;
  }

  free(paths);
  return 0;
}
#endif /* !HWLOC_HAVE_LTDL */

/* array of pointers to dynamically loaded plugins */
static struct hwloc__plugin_desc {
  char *name;
  struct hwloc_component *component;
  char *filename;
  hwloc_dlhandle handle;
  struct hwloc__plugin_desc *next;
} *hwloc_plugins = NULL;

static int
hwloc__dlforeach_cb(const char *filename, void *_data __hwloc_attribute_unused)
{
  const char *basename;
  hwloc_dlhandle handle;
  struct hwloc_component *component;
  struct hwloc__plugin_desc *desc, **prevdesc;
  char *componentsymbolname;

  if (hwloc_plugins_verbose)
    fprintf(stderr, "hwloc: Plugin dlforeach found `%s'\n", filename);

  basename = strrchr(filename, '/');
  if (!basename)
    basename = filename;
  else
    basename++;

  if (hwloc_plugins_blacklist && strstr(hwloc_plugins_blacklist, basename)) {
    if (hwloc_plugins_verbose)
      fprintf(stderr, "hwloc: Plugin `%s' is blacklisted in the environment\n", basename);
    goto out;
  }

  /* dlopen and get the component structure */
  handle = hwloc_dlopenext(filename);
  if (!handle) {
    if (hwloc_plugins_verbose)
      fprintf(stderr, "hwloc: Failed to load plugin: %s\n", hwloc_dlerror());
    goto out;
  }

  componentsymbolname = malloc(strlen(basename)+10+1);
  if (!componentsymbolname) {
    if (hwloc_plugins_verbose)
      fprintf(stderr, "hwloc: Failed to allocation component `%s' symbol\n",
	      basename);
    goto out_with_handle;
  }
  sprintf(componentsymbolname, "%s_component", basename);
  component = hwloc_dlsym(handle, componentsymbolname);
  if (!component) {
    if (hwloc_plugins_verbose)
      fprintf(stderr, "hwloc: Failed to find component symbol `%s'\n",
	      componentsymbolname);
    free(componentsymbolname);
    goto out_with_handle;
  }
  if (component->abi != HWLOC_COMPONENT_ABI) {
    if (hwloc_plugins_verbose)
      fprintf(stderr, "hwloc: Plugin symbol ABI %u instead of %d\n",
	      component->abi, HWLOC_COMPONENT_ABI);
    free(componentsymbolname);
    goto out_with_handle;
  }
  if (hwloc_plugins_verbose)
    fprintf(stderr, "hwloc: Plugin contains expected symbol `%s'\n",
	    componentsymbolname);
  free(componentsymbolname);

  if (HWLOC_COMPONENT_TYPE_DISC == component->type) {
    if (strncmp(basename, "hwloc_", 6)) {
      if (hwloc_plugins_verbose)
	fprintf(stderr, "hwloc: Plugin name `%s' doesn't match its type DISCOVERY\n", basename);
      goto out_with_handle;
    }
  } else if (HWLOC_COMPONENT_TYPE_XML == component->type) {
    if (strncmp(basename, "hwloc_xml_", 10)) {
      if (hwloc_plugins_verbose)
	fprintf(stderr, "hwloc: Plugin name `%s' doesn't match its type XML\n", basename);
      goto out_with_handle;
    }
  } else {
    if (hwloc_plugins_verbose)
      fprintf(stderr, "hwloc: Plugin name `%s' has invalid type %u\n",
	      basename, (unsigned) component->type);
    goto out_with_handle;
  }

  /* allocate a plugin_desc and queue it */
  desc = malloc(sizeof(*desc));
  if (!desc)
    goto out_with_handle;
  desc->name = strdup(basename);
  desc->filename = strdup(filename);
  desc->component = component;
  desc->handle = handle;
  desc->next = NULL;
  if (hwloc_plugins_verbose)
    fprintf(stderr, "hwloc: Plugin descriptor `%s' ready\n", basename);

  /* append to the list */
  prevdesc = &hwloc_plugins;
  while (*prevdesc)
    prevdesc = &((*prevdesc)->next);
  *prevdesc = desc;
  if (hwloc_plugins_verbose)
    fprintf(stderr, "hwloc: Plugin descriptor `%s' queued\n", basename);
  return 0;

 out_with_handle:
  hwloc_dlclose(handle);
 out:
  return 0;
}

static void
hwloc_plugins_exit(void)
{
  struct hwloc__plugin_desc *desc, *next;

  if (hwloc_plugins_verbose)
    fprintf(stderr, "hwloc: Closing all plugins\n");

  desc = hwloc_plugins;
  while (desc) {
    next = desc->next;
    hwloc_dlclose(desc->handle);
    free(desc->name);
    free(desc->filename);
    free(desc);
    desc = next;
  }
  hwloc_plugins = NULL;

  hwloc_dlexit();
}

static int
hwloc_plugins_init(void)
{
  const char *verboseenv;
  const char *path = HWLOC_PLUGINS_PATH;
  const char *env;
  int err;

  verboseenv = getenv("HWLOC_PLUGINS_VERBOSE");
  hwloc_plugins_verbose = verboseenv ? atoi(verboseenv) : 0;

  hwloc_plugins_blacklist = getenv("HWLOC_PLUGINS_BLACKLIST");

  err = hwloc_dlinit();
  if (err)
    goto out;

  env = getenv("HWLOC_PLUGINS_PATH");
  if (env)
    path = env;

  hwloc_plugins = NULL;

  if (hwloc_plugins_verbose)
    fprintf(stderr, "hwloc: Starting plugin dlforeach in %s\n", path);
  err = hwloc_dlforeachfile(path, hwloc__dlforeach_cb, NULL);
  if (err)
    goto out_with_init;

  return 0;

 out_with_init:
  hwloc_plugins_exit();
 out:
  return -1;
}

#endif /* HWLOC_HAVE_PLUGINS */

static int
hwloc_disc_component_register(struct hwloc_disc_component *component,
			      const char *filename)
{
  struct hwloc_disc_component **prev;

  /* check that the component name is valid */
  if (!strcmp(component->name, HWLOC_COMPONENT_STOP_NAME)) {
    if (hwloc_components_verbose)
      fprintf(stderr, "hwloc: Cannot register discovery component with reserved name `" HWLOC_COMPONENT_STOP_NAME "'\n");
    return -1;
  }
  if (strchr(component->name, HWLOC_COMPONENT_EXCLUDE_CHAR)
      || strchr(component->name, HWLOC_COMPONENT_PHASESEP_CHAR)
      || strcspn(component->name, HWLOC_COMPONENT_SEPS) != strlen(component->name)) {
    if (hwloc_components_verbose)
      fprintf(stderr, "hwloc: Cannot register discovery component with name `%s' containing reserved characters `%c" HWLOC_COMPONENT_SEPS "'\n",
	      component->name, HWLOC_COMPONENT_EXCLUDE_CHAR);
    return -1;
  }

  /* check that the component phases are valid */
  if (!component->phases
      || (component->phases != HWLOC_DISC_PHASE_GLOBAL
	  && component->phases & ~(HWLOC_DISC_PHASE_CPU
				   |HWLOC_DISC_PHASE_MEMORY
				   |HWLOC_DISC_PHASE_PCI
				   |HWLOC_DISC_PHASE_IO
				   |HWLOC_DISC_PHASE_MISC
				   |HWLOC_DISC_PHASE_ANNOTATE
				   |HWLOC_DISC_PHASE_TWEAK))) {
    if (HWLOC_SHOW_CRITICAL_ERRORS())
      fprintf(stderr, "hwloc: Cannot register discovery component `%s' with invalid phases 0x%x\n",
              component->name, component->phases);
    return -1;
  }

  prev = &hwloc_disc_components;
  while (NULL != *prev) {
    if (!strcmp((*prev)->name, component->name)) {
      /* if two components have the same name, only keep the highest priority one */
      if ((*prev)->priority < component->priority) {
	/* drop the existing component */
	if (hwloc_components_verbose)
	  fprintf(stderr, "hwloc: Dropping previously registered discovery component `%s', priority %u lower than new one %u\n",
		  (*prev)->name, (*prev)->priority, component->priority);
	*prev = (*prev)->next;
      } else {
	/* drop the new one */
	if (hwloc_components_verbose)
	  fprintf(stderr, "hwloc: Ignoring new discovery component `%s', priority %u lower than previously registered one %u\n",
		  component->name, component->priority, (*prev)->priority);
	return -1;
      }
    }
    prev = &((*prev)->next);
  }
  if (hwloc_components_verbose)
    fprintf(stderr, "hwloc: Registered discovery component `%s' phases 0x%x with priority %u (%s%s)\n",
	    component->name, component->phases, component->priority,
	    filename ? "from plugin " : "statically build", filename ? filename : "");

  prev = &hwloc_disc_components;
  while (NULL != *prev) {
    if ((*prev)->priority < component->priority)
      break;
    prev = &((*prev)->next);
  }
  component->next = *prev;
  *prev = component;
  return 0;
}

#include "static-components.h"

static void (**hwloc_component_finalize_cbs)(unsigned long);
static unsigned hwloc_component_finalize_cb_count;

void
hwloc_components_init(void)
{
#ifdef HWLOC_HAVE_PLUGINS
  struct hwloc__plugin_desc *desc;
#endif
  const char *verboseenv;
  unsigned i;

  HWLOC_COMPONENTS_LOCK();
  assert((unsigned) -1 != hwloc_components_users);
  if (0 != hwloc_components_users++) {
    HWLOC_COMPONENTS_UNLOCK();
    return;
  }

  verboseenv = getenv("HWLOC_COMPONENTS_VERBOSE");
  hwloc_components_verbose = verboseenv ? atoi(verboseenv) : 0;

#ifdef HWLOC_HAVE_PLUGINS
  hwloc_plugins_init();
#endif

  hwloc_component_finalize_cbs = NULL;
  hwloc_component_finalize_cb_count = 0;
  /* count the max number of finalize callbacks */
  for(i=0; NULL != hwloc_static_components[i]; i++)
    hwloc_component_finalize_cb_count++;
#ifdef HWLOC_HAVE_PLUGINS
  for(desc = hwloc_plugins; NULL != desc; desc = desc->next)
    hwloc_component_finalize_cb_count++;
#endif
  if (hwloc_component_finalize_cb_count) {
    hwloc_component_finalize_cbs = calloc(hwloc_component_finalize_cb_count,
					  sizeof(*hwloc_component_finalize_cbs));
    assert(hwloc_component_finalize_cbs);
    /* forget that max number and recompute the real one below */
    hwloc_component_finalize_cb_count = 0;
  }

  /* hwloc_static_components is created by configure in static-components.h */
  for(i=0; NULL != hwloc_static_components[i]; i++) {
    if (hwloc_static_components[i]->flags) {
      if (HWLOC_SHOW_CRITICAL_ERRORS())
        fprintf(stderr, "hwloc: Ignoring static component with invalid flags %lx\n",
                hwloc_static_components[i]->flags);
      continue;
    }

    /* initialize the component */
    if (hwloc_static_components[i]->init && hwloc_static_components[i]->init(0) < 0) {
      if (hwloc_components_verbose)
	fprintf(stderr, "hwloc: Ignoring static component, failed to initialize\n");
      continue;
    }
    /* queue ->finalize() callback if any */
    if (hwloc_static_components[i]->finalize)
      hwloc_component_finalize_cbs[hwloc_component_finalize_cb_count++] = hwloc_static_components[i]->finalize;

    /* register for real now */
    if (HWLOC_COMPONENT_TYPE_DISC == hwloc_static_components[i]->type)
      hwloc_disc_component_register(hwloc_static_components[i]->data, NULL);
    else if (HWLOC_COMPONENT_TYPE_XML == hwloc_static_components[i]->type)
      hwloc_xml_callbacks_register(hwloc_static_components[i]->data);
    else
      assert(0);
  }

  /* dynamic plugins */
#ifdef HWLOC_HAVE_PLUGINS
  for(desc = hwloc_plugins; NULL != desc; desc = desc->next) {
    if (desc->component->flags) {
      if (HWLOC_SHOW_CRITICAL_ERRORS())
        fprintf(stderr, "hwloc: Ignoring plugin `%s' component with invalid flags %lx\n",
                desc->name, desc->component->flags);
      continue;
    }

    /* initialize the component */
    if (desc->component->init && desc->component->init(0) < 0) {
      if (hwloc_components_verbose)
	fprintf(stderr, "hwloc: Ignoring plugin `%s', failed to initialize\n", desc->name);
      continue;
    }
    /* queue ->finalize() callback if any */
    if (desc->component->finalize)
      hwloc_component_finalize_cbs[hwloc_component_finalize_cb_count++] = desc->component->finalize;

    /* register for real now */
    if (HWLOC_COMPONENT_TYPE_DISC == desc->component->type)
      hwloc_disc_component_register(desc->component->data, desc->filename);
    else if (HWLOC_COMPONENT_TYPE_XML == desc->component->type)
      hwloc_xml_callbacks_register(desc->component->data);
    else
      assert(0);
  }
#endif

  HWLOC_COMPONENTS_UNLOCK();
}

void
hwloc_topology_components_init(struct hwloc_topology *topology)
{
  topology->nr_blacklisted_components = 0;
  topology->blacklisted_components = NULL;

  topology->backends = NULL;
  topology->backend_phases = 0;
  topology->backend_excluded_phases = 0;
}

/* look for name among components, ignoring things after `:' */
static struct hwloc_disc_component *
hwloc_disc_component_find(const char *name, const char **endp)
{
  struct hwloc_disc_component *comp;
  size_t length;
  const char *end = strchr(name, HWLOC_COMPONENT_PHASESEP_CHAR);
  if (end) {
    length = end-name;
    if (endp)
      *endp = end+1;
  } else {
    length = strlen(name);
    if (endp)
      *endp = NULL;
  }

  comp = hwloc_disc_components;
  while (NULL != comp) {
    if (!strncmp(name, comp->name, length))
      return comp;
    comp = comp->next;
  }
  return NULL;
}

static unsigned
hwloc_phases_from_string(const char *s)
{
  if (!s)
    return ~0U;
  if (s[0]<'0' || s[0]>'9') {
    if (!strcasecmp(s, "global"))
      return HWLOC_DISC_PHASE_GLOBAL;
    else if (!strcasecmp(s, "cpu"))
      return HWLOC_DISC_PHASE_CPU;
    if (!strcasecmp(s, "memory"))
      return HWLOC_DISC_PHASE_MEMORY;
    if (!strcasecmp(s, "pci"))
      return HWLOC_DISC_PHASE_PCI;
    if (!strcasecmp(s, "io"))
      return HWLOC_DISC_PHASE_IO;
    if (!strcasecmp(s, "misc"))
      return HWLOC_DISC_PHASE_MISC;
    if (!strcasecmp(s, "annotate"))
      return HWLOC_DISC_PHASE_ANNOTATE;
    if (!strcasecmp(s, "tweak"))
      return HWLOC_DISC_PHASE_TWEAK;
    return 0;
  }
  return (unsigned) strtoul(s, NULL, 0);
}

static int
hwloc_disc_component_blacklist_one(struct hwloc_topology *topology,
				   const char *name)
{
  struct hwloc_topology_forced_component_s *blacklisted;
  struct hwloc_disc_component *comp;
  unsigned phases;
  unsigned i;

  if (!strcmp(name, "linuxpci") || !strcmp(name, "linuxio")) {
    /* replace linuxpci and linuxio with linux (with IO phases)
     * for backward compatibility with pre-v2.0 and v2.0 respectively */
    if (hwloc_components_verbose)
      fprintf(stderr, "hwloc: Replacing deprecated component `%s' with `linux' IO phases in blacklisting\n", name);
    comp = hwloc_disc_component_find("linux", NULL);
    phases = HWLOC_DISC_PHASE_PCI | HWLOC_DISC_PHASE_IO | HWLOC_DISC_PHASE_MISC | HWLOC_DISC_PHASE_ANNOTATE;

  } else {
    /* normal lookup */
    const char *end;
    comp = hwloc_disc_component_find(name, &end);
    phases = hwloc_phases_from_string(end);
  }
  if (!comp) {
    errno = EINVAL;
    return -1;
  }

  if (hwloc_components_verbose)
    fprintf(stderr, "hwloc: Blacklisting component `%s` phases 0x%x\n", comp->name, phases);

  for(i=0; i<topology->nr_blacklisted_components; i++) {
    if (topology->blacklisted_components[i].component == comp) {
      topology->blacklisted_components[i].phases |= phases;
      return 0;
    }
  }

  blacklisted = realloc(topology->blacklisted_components, (topology->nr_blacklisted_components+1)*sizeof(*blacklisted));
  if (!blacklisted)
    return -1;

  blacklisted[topology->nr_blacklisted_components].component = comp;
  blacklisted[topology->nr_blacklisted_components].phases = phases;
  topology->blacklisted_components = blacklisted;
  topology->nr_blacklisted_components++;
  return 0;
}

int
hwloc_topology_set_components(struct hwloc_topology *topology,
			      unsigned long flags,
			      const char *name)
{
  if (topology->is_loaded) {
    errno = EBUSY;
    return -1;
  }

  if (flags & ~HWLOC_TOPOLOGY_COMPONENTS_FLAG_BLACKLIST) {
    errno = EINVAL;
    return -1;
  }

  /* this flag is strictly required for now */
  if (flags != HWLOC_TOPOLOGY_COMPONENTS_FLAG_BLACKLIST) {
    errno = EINVAL;
    return -1;
  }

  if (!strncmp(name, "all", 3) && name[3] == HWLOC_COMPONENT_PHASESEP_CHAR) {
    topology->backend_excluded_phases = hwloc_phases_from_string(name+4);
    return 0;
  }

  return hwloc_disc_component_blacklist_one(topology, name);
}

/* used by set_xml(), set_synthetic(), ... environment variables, ... to force the first backend */
int
hwloc_disc_component_force_enable(struct hwloc_topology *topology,
				  int envvar_forced,
				  const char *name,
				  const void *data1, const void *data2, const void *data3)
{
  struct hwloc_disc_component *comp;
  struct hwloc_backend *backend;

  if (topology->is_loaded) {
    errno = EBUSY;
    return -1;
  }

  comp = hwloc_disc_component_find(name, NULL);
  if (!comp) {
    errno = ENOSYS;
    return -1;
  }

  backend = comp->instantiate(topology, comp, 0U /* force-enabled don't get any phase blacklisting */,
			      data1, data2, data3);
  if (backend) {
    int err;
    backend->envvar_forced = envvar_forced;
    if (topology->backends)
      hwloc_backends_disable_all(topology);
    err = hwloc_backend_enable(backend);

    if (comp->phases == HWLOC_DISC_PHASE_GLOBAL) {
      char *env = getenv("HWLOC_ANNOTATE_GLOBAL_COMPONENTS");
      if (env && atoi(env))
	topology->backend_excluded_phases &= ~HWLOC_DISC_PHASE_ANNOTATE;
    }

    return err;
  } else
    return -1;
}

static int
hwloc_disc_component_try_enable(struct hwloc_topology *topology,
				struct hwloc_disc_component *comp,
				int envvar_forced,
				unsigned blacklisted_phases)
{
  struct hwloc_backend *backend;

  if (!(comp->phases & ~(topology->backend_excluded_phases | blacklisted_phases))) {
    /* all this backend phases are already excluded, exclude the backend entirely */
    if (hwloc_components_verbose)
      /* do not warn if envvar_forced since system-wide HWLOC_COMPONENTS must be silently ignored after set_xml() etc.
       */
      fprintf(stderr, "hwloc: Excluding discovery component `%s' phases 0x%x, conflicts with excludes 0x%x\n",
	      comp->name, comp->phases, topology->backend_excluded_phases);
    return -1;
  }

  backend = comp->instantiate(topology, comp, topology->backend_excluded_phases | blacklisted_phases,
			      NULL, NULL, NULL);
  if (!backend) {
    if (hwloc_components_verbose || (envvar_forced && HWLOC_SHOW_CRITICAL_ERRORS()))
      fprintf(stderr, "hwloc: Failed to instantiate discovery component `%s'\n", comp->name);
    return -1;
  }

  backend->phases &= ~blacklisted_phases;
  backend->envvar_forced = envvar_forced;
  return hwloc_backend_enable(backend);
}

void
hwloc_disc_components_enable_others(struct hwloc_topology *topology)
{
  struct hwloc_disc_component *comp;
  struct hwloc_backend *backend;
  int tryall = 1;
  const char *_env;
  char *env; /* we'll to modify the env value, so duplicate it */
  unsigned i;

  _env = getenv("HWLOC_COMPONENTS");
  env = _env ? strdup(_env) : NULL;

  /* blacklist disabled components */
  if (env) {
    char *curenv = env;
    size_t s;

    while (*curenv) {
      s = strcspn(curenv, HWLOC_COMPONENT_SEPS);
      if (s) {
	char c;

	if (curenv[0] != HWLOC_COMPONENT_EXCLUDE_CHAR)
	  goto nextname;

	/* save the last char and replace with \0 */
	c = curenv[s];
	curenv[s] = '\0';

	/* blacklist it, and just ignore failures to allocate */
	hwloc_disc_component_blacklist_one(topology, curenv+1);

	/* remove that blacklisted name from the string */
	for(i=0; i<s; i++)
	  curenv[i] = *HWLOC_COMPONENT_SEPS;

	/* restore chars (the second loop below needs env to be unmodified) */
	curenv[s] = c;
      }

    nextname:
      curenv += s;
      if (*curenv)
	/* Skip comma */
	curenv++;
    }
  }

  /* enable explicitly listed components */
  if (env) {
    char *curenv = env;
    size_t s;

    while (*curenv) {
      s = strcspn(curenv, HWLOC_COMPONENT_SEPS);
      if (s) {
	char c;
	const char *name;

	if (!strncmp(curenv, HWLOC_COMPONENT_STOP_NAME, s)) {
	  tryall = 0;
	  break;
	}

	/* save the last char and replace with \0 */
	c = curenv[s];
	curenv[s] = '\0';

	name = curenv;
	if (!strcmp(name, "linuxpci") || !strcmp(name, "linuxio")) {
	  if (hwloc_components_verbose)
	    fprintf(stderr, "hwloc: Replacing deprecated component `%s' with `linux' in envvar forcing\n", name);
	  name = "linux";
	}

	comp = hwloc_disc_component_find(name, NULL /* we enable the entire component, phases must be blacklisted separately */);
	if (comp) {
	  unsigned blacklisted_phases = 0U;
	  for(i=0; i<topology->nr_blacklisted_components; i++)
	    if (comp == topology->blacklisted_components[i].component) {
	      blacklisted_phases = topology->blacklisted_components[i].phases;
	      break;
	    }
	  if (comp->phases & ~blacklisted_phases)
	    hwloc_disc_component_try_enable(topology, comp, 1 /* envvar forced */, blacklisted_phases);
	} else {
          if (HWLOC_SHOW_CRITICAL_ERRORS())
            fprintf(stderr, "hwloc: Cannot find discovery component `%s'\n", name);
	}

	/* restore chars (the second loop below needs env to be unmodified) */
	curenv[s] = c;
      }

      curenv += s;
      if (*curenv)
	/* Skip comma */
	curenv++;
    }
  }

  /* env is still the same, the above loop didn't modify it */

  /* now enable remaining components (except the explicitly '-'-listed ones) */
  if (tryall) {
    comp = hwloc_disc_components;
    while (NULL != comp) {
      unsigned blacklisted_phases = 0U;
      if (!comp->enabled_by_default)
	goto nextcomp;
      /* check if this component was blacklisted by the application */
      for(i=0; i<topology->nr_blacklisted_components; i++)
	if (comp == topology->blacklisted_components[i].component) {
	  blacklisted_phases = topology->blacklisted_components[i].phases;
	  break;
	}

      if (!(comp->phases & ~blacklisted_phases)) {
	if (hwloc_components_verbose)
	  fprintf(stderr, "hwloc: Excluding blacklisted discovery component `%s' phases 0x%x\n",
		  comp->name, comp->phases);
	goto nextcomp;
      }

      hwloc_disc_component_try_enable(topology, comp, 0 /* defaults, not envvar forced */, blacklisted_phases);
nextcomp:
      comp = comp->next;
    }
  }

  if (hwloc_components_verbose) {
    /* print a summary */
    int first = 1;
    backend = topology->backends;
    fprintf(stderr, "hwloc: Final list of enabled discovery components: ");
    while (backend != NULL) {
      fprintf(stderr, "%s%s(0x%x)", first ? "" : ",", backend->component->name, backend->phases);
      backend = backend->next;
      first = 0;
    }
    fprintf(stderr, "\n");
  }

  free(env);
}

void
hwloc_components_fini(void)
{
  unsigned i;

  HWLOC_COMPONENTS_LOCK();
  assert(0 != hwloc_components_users);
  if (0 != --hwloc_components_users) {
    HWLOC_COMPONENTS_UNLOCK();
    return;
  }

  for(i=0; i<hwloc_component_finalize_cb_count; i++)
    hwloc_component_finalize_cbs[hwloc_component_finalize_cb_count-i-1](0);
  free(hwloc_component_finalize_cbs);
  hwloc_component_finalize_cbs = NULL;
  hwloc_component_finalize_cb_count = 0;

  /* no need to unlink/free the list of components, they'll be unloaded below */

  hwloc_disc_components = NULL;
  hwloc_xml_callbacks_reset();

#ifdef HWLOC_HAVE_PLUGINS
  hwloc_plugins_exit();
#endif

  HWLOC_COMPONENTS_UNLOCK();
}

struct hwloc_backend *
hwloc_backend_alloc(struct hwloc_topology *topology,
		    struct hwloc_disc_component *component)
{
  struct hwloc_backend * backend = malloc(sizeof(*backend));
  if (!backend) {
    errno = ENOMEM;
    return NULL;
  }
  backend->component = component;
  backend->topology = topology;
  /* filter-out component phases that are excluded */
  backend->phases = component->phases & ~topology->backend_excluded_phases;
  if (backend->phases != component->phases && hwloc_components_verbose)
    fprintf(stderr, "hwloc: Trying discovery component `%s' with phases 0x%x instead of 0x%x\n",
	    component->name, backend->phases, component->phases);
  backend->flags = 0;
  backend->discover = NULL;
  backend->get_pci_busid_cpuset = NULL;
  backend->disable = NULL;
  backend->is_thissystem = -1;
  backend->next = NULL;
  backend->envvar_forced = 0;
  return backend;
}

static void
hwloc_backend_disable(struct hwloc_backend *backend)
{
  if (backend->disable)
    backend->disable(backend);
  free(backend);
}

int
hwloc_backend_enable(struct hwloc_backend *backend)
{
  struct hwloc_topology *topology = backend->topology;
  struct hwloc_backend **pprev;

  /* check backend flags */
  if (backend->flags) {
    if (HWLOC_SHOW_CRITICAL_ERRORS())
      fprintf(stderr, "hwloc: Cannot enable discovery component `%s' phases 0x%x with unknown flags %lx\n",
              backend->component->name, backend->component->phases, backend->flags);
    return -1;
  }

  /* make sure we didn't already enable this backend, we don't want duplicates */
  pprev = &topology->backends;
  while (NULL != *pprev) {
    if ((*pprev)->component == backend->component) {
      if (hwloc_components_verbose)
	fprintf(stderr, "hwloc: Cannot enable  discovery component `%s' phases 0x%x twice\n",
		backend->component->name, backend->component->phases);
      hwloc_backend_disable(backend);
      errno = EBUSY;
      return -1;
    }
    pprev = &((*pprev)->next);
  }

  if (hwloc_components_verbose)
    fprintf(stderr, "hwloc: Enabling discovery component `%s' with phases 0x%x (among 0x%x)\n",
	    backend->component->name, backend->phases, backend->component->phases);

  /* enqueue at the end */
  pprev = &topology->backends;
  while (NULL != *pprev)
    pprev = &((*pprev)->next);
  backend->next = *pprev;
  *pprev = backend;

  topology->backend_phases |= backend->component->phases;
  topology->backend_excluded_phases |= backend->component->excluded_phases;
  return 0;
}

void
hwloc_backends_is_thissystem(struct hwloc_topology *topology)
{
  struct hwloc_backend *backend;
  const char *local_env;

  /*
   * If the application changed the backend with set_foo(),
   * it may use set_flags() update the is_thissystem flag here.
   * If it changes the backend with environment variables below,
   * it may use HWLOC_THISSYSTEM envvar below as well.
   */

  topology->is_thissystem = 1;

  /* apply thissystem from normally-given backends (envvar_forced=0, either set_foo() or defaults) */
  backend = topology->backends;
  while (backend != NULL) {
    if (backend->envvar_forced == 0 && backend->is_thissystem != -1) {
      assert(backend->is_thissystem == 0);
      topology->is_thissystem = 0;
    }
    backend = backend->next;
  }

  /* override set_foo() with flags */
  if (topology->flags & HWLOC_TOPOLOGY_FLAG_IS_THISSYSTEM)
    topology->is_thissystem = 1;

  /* now apply envvar-forced backend (envvar_forced=1) */
  backend = topology->backends;
  while (backend != NULL) {
    if (backend->envvar_forced == 1 && backend->is_thissystem != -1) {
      assert(backend->is_thissystem == 0);
      topology->is_thissystem = 0;
    }
    backend = backend->next;
  }

  /* override with envvar-given flag */
  local_env = getenv("HWLOC_THISSYSTEM");
  if (local_env)
    topology->is_thissystem = atoi(local_env);
}

void
hwloc_backends_find_callbacks(struct hwloc_topology *topology)
{
  struct hwloc_backend *backend = topology->backends;
  /* use the first backend's get_pci_busid_cpuset callback */
  topology->get_pci_busid_cpuset_backend = NULL;
  while (backend != NULL) {
    if (backend->get_pci_busid_cpuset) {
      topology->get_pci_busid_cpuset_backend = backend;
      return;
    }
    backend = backend->next;
  }
  return;
}

void
hwloc_backends_disable_all(struct hwloc_topology *topology)
{
  struct hwloc_backend *backend;

  while (NULL != (backend = topology->backends)) {
    struct hwloc_backend *next = backend->next;
    if (hwloc_components_verbose)
      fprintf(stderr, "hwloc: Disabling discovery component `%s'\n",
	      backend->component->name);
    hwloc_backend_disable(backend);
    topology->backends = next;
  }
  topology->backends = NULL;
  topology->backend_excluded_phases = 0;
}

void
hwloc_topology_components_fini(struct hwloc_topology *topology)
{
  /* hwloc_backends_disable_all() must have been called earlier */
  assert(!topology->backends);

  free(topology->blacklisted_components);
}
