/*
 * Copyright © 2009 CNRS
 * Copyright © 2009-2019 Inria.  All rights reserved.
 * Copyright © 2009-2011 Université Bordeaux
 * Copyright © 2009-2011 Cisco Systems, Inc.  All rights reserved.
 * See COPYING in top-level directory.
 */

#include "private/autogen/config.h"
#include "hwloc.h"
#include "hwloc/plugins.h"
#include "private/private.h"
#include "private/misc.h"
#include "private/xml.h"
#include "private/debug.h"

#include <string.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

/*******************
 * Import routines *
 *******************/

struct hwloc__nolibxml_backend_data_s {
  size_t buflen; /* size of both buffer, set during backend_init() */
  char *buffer; /* allocated and filled during backend_init() */
};

typedef struct hwloc__nolibxml_import_state_data_s {
  char *tagbuffer; /* buffer containing the next tag */
  char *attrbuffer; /* buffer containing the next attribute of the current node */
  char *tagname; /* tag name of the current node */
  int closed; /* set if the current node is auto-closing */
} __hwloc_attribute_may_alias * hwloc__nolibxml_import_state_data_t;

static char *
hwloc__nolibxml_import_ignore_spaces(char *buffer)
{
  return buffer + strspn(buffer, " \t\n");
}

static int
hwloc__nolibxml_import_next_attr(hwloc__xml_import_state_t state, char **namep, char **valuep)
{
  hwloc__nolibxml_import_state_data_t nstate = (void*) state->data;
  size_t namelen;
  size_t len, escaped;
  char *buffer, *value, *end;

  if (!nstate->attrbuffer)
    return -1;

  /* find the beginning of an attribute */
  buffer = hwloc__nolibxml_import_ignore_spaces(nstate->attrbuffer);
  namelen = strspn(buffer, "abcdefghijklmnopqrstuvwxyz_");
  if (buffer[namelen] != '=' || buffer[namelen+1] != '\"')
    return -1;
  buffer[namelen] = '\0';
  *namep = buffer;

  /* find the beginning of its value, and unescape it */
  *valuep = value = buffer+namelen+2;
  len = 0; escaped = 0;
  while (value[len+escaped] != '\"') {
    if (value[len+escaped] == '&') {
      if (!strncmp(&value[1+len+escaped], "#10;", 4)) {
	escaped += 4;
	value[len] = '\n';
      } else if (!strncmp(&value[1+len+escaped], "#13;", 4)) {
	escaped += 4;
	value[len] = '\r';
      } else if (!strncmp(&value[1+len+escaped], "#9;", 3)) {
	escaped += 3;
	value[len] = '\t';
      } else if (!strncmp(&value[1+len+escaped], "quot;", 5)) {
	escaped += 5;
	value[len] = '\"';
      } else if (!strncmp(&value[1+len+escaped], "lt;", 3)) {
	escaped += 3;
	value[len] = '<';
      } else if (!strncmp(&value[1+len+escaped], "gt;", 3)) {
	escaped += 3;
	value[len] = '>';
      } else if (!strncmp(&value[1+len+escaped], "amp;", 4)) {
	escaped += 4;
	value[len] = '&';
      } else {
	return -1;
      }
    } else {
      value[len] = value[len+escaped];
    }
    len++;
    if (value[len+escaped] == '\0')
      return -1;
  }
  value[len] = '\0';

  /* find next attribute */
  end = &value[len+escaped+1]; /* skip the ending " */
  nstate->attrbuffer = hwloc__nolibxml_import_ignore_spaces(end);
  return 0;
}

static int
hwloc__nolibxml_import_find_child(hwloc__xml_import_state_t state,
				  hwloc__xml_import_state_t childstate,
				  char **tagp)
{
  hwloc__nolibxml_import_state_data_t nstate = (void*) state->data;
  hwloc__nolibxml_import_state_data_t nchildstate = (void*) childstate->data;
  char *buffer = nstate->tagbuffer;
  char *end;
  char *tag;
  size_t namelen;

  childstate->parent = state;
  childstate->global = state->global;

  /* auto-closed tags have no children */
  if (nstate->closed)
    return 0;

  /* find the beginning of the tag */
  buffer = hwloc__nolibxml_import_ignore_spaces(buffer);
  if (buffer[0] != '<')
    return -1;
  buffer++;

  /* if closing tag, return nothing and do not advance */
  if (buffer[0] == '/')
    return 0;

  /* normal tag */
  tag = nchildstate->tagname = buffer;

  /* find the end, mark it and return it */
  end = strchr(buffer, '>');
  if (!end)
    return -1;
  end[0] = '\0';
  nchildstate->tagbuffer = end+1;

  /* handle auto-closing tags */
  if (end[-1] == '/') {
    nchildstate->closed = 1;
    end[-1] = '\0';
  } else
    nchildstate->closed = 0;

  /* find attributes */
  namelen = strspn(buffer, "abcdefghijklmnopqrstuvwxyz1234567890_");

  if (buffer[namelen] == '\0') {
    /* no attributes */
    nchildstate->attrbuffer = NULL;
    *tagp = tag;
    return 1;
  }

  if (buffer[namelen] != ' ')
    return -1;

  /* found a space, likely starting attributes */
  buffer[namelen] = '\0';
  nchildstate->attrbuffer = buffer+namelen+1;
  *tagp = tag;
  return 1;
}

static int
hwloc__nolibxml_import_close_tag(hwloc__xml_import_state_t state)
{
  hwloc__nolibxml_import_state_data_t nstate = (void*) state->data;
  char *buffer = nstate->tagbuffer;
  char *end;

  /* auto-closed tags need nothing */
  if (nstate->closed)
    return 0;

  /* find the beginning of the tag */
  buffer = hwloc__nolibxml_import_ignore_spaces(buffer);
  if (buffer[0] != '<')
    return -1;
  buffer++;

  /* find the end, mark it and return it to the parent */
  end = strchr(buffer, '>');
  if (!end)
    return -1;
  end[0] = '\0';
  nstate->tagbuffer = end+1;

  /* if closing tag, return nothing */
  if (buffer[0] != '/' || strcmp(buffer+1, nstate->tagname) )
    return -1;
  return 0;
}

static void
hwloc__nolibxml_import_close_child(hwloc__xml_import_state_t state)
{
  hwloc__nolibxml_import_state_data_t nstate = (void*) state->data;
  hwloc__nolibxml_import_state_data_t nparent = (void*) state->parent->data;
  nparent->tagbuffer = nstate->tagbuffer;
}

static int
hwloc__nolibxml_import_get_content(hwloc__xml_import_state_t state,
				   char **beginp, size_t expected_length)
{
  hwloc__nolibxml_import_state_data_t nstate = (void*) state->data;
  char *buffer = nstate->tagbuffer;
  size_t length;
  char *end;

  /* auto-closed tags have no content */
  if (nstate->closed) {
    if (expected_length)
      return -1;
    *beginp = (char *) "";
    return 0;
  }

  /* find the next tag, where the content ends */
  end = strchr(buffer, '<');
  if (!end)
    return -1;

  length = (size_t) (end-buffer);
  if (length != expected_length)
    return -1;
  nstate->tagbuffer = end;
  *end = '\0'; /* mark as 0-terminated for now */
  *beginp = buffer;
  return 1;
}

static void
hwloc__nolibxml_import_close_content(hwloc__xml_import_state_t state)
{
  /* put back the '<' that we overwrote to 0-terminate the content */
  hwloc__nolibxml_import_state_data_t nstate = (void*) state->data;
  if (!nstate->closed)
    *nstate->tagbuffer = '<';
}

static int
hwloc_nolibxml_look_init(struct hwloc_xml_backend_data_s *bdata,
			 struct hwloc__xml_import_state_s *state)
{
  hwloc__nolibxml_import_state_data_t nstate = (void*) state->data;
  struct hwloc__nolibxml_backend_data_s *nbdata = bdata->data;
  unsigned major, minor;
  char *end;
  char *buffer = nbdata->buffer;
  char *tagname;

  HWLOC_BUILD_ASSERT(sizeof(*nstate) <= sizeof(state->data));

  /* skip headers */
  while (!strncmp(buffer, "<?xml ", 6) || !strncmp(buffer, "<!DOCTYPE ", 10)) {
    buffer = strchr(buffer, '\n');
    if (!buffer)
      goto failed;
    buffer++;
  }

  /* find topology tag */
  if (sscanf(buffer, "<topology version=\"%u.%u\">", &major, &minor) == 2) {
    bdata->version_major = major;
    bdata->version_minor = minor;
    end = strchr(buffer, '>') + 1;
    tagname = "topology";
  } else if (!strncmp(buffer, "<topology>", 10)) {
    bdata->version_major = 1;
    bdata->version_minor = 0;
    end = buffer + 10;
    tagname = "topology";
  } else if (!strncmp(buffer, "<root>", 6)) {
    bdata->version_major = 0;
    bdata->version_minor = 9;
    end = buffer + 6;
    tagname = "root";
  } else
    goto failed;

  state->global->next_attr = hwloc__nolibxml_import_next_attr;
  state->global->find_child = hwloc__nolibxml_import_find_child;
  state->global->close_tag = hwloc__nolibxml_import_close_tag;
  state->global->close_child = hwloc__nolibxml_import_close_child;
  state->global->get_content = hwloc__nolibxml_import_get_content;
  state->global->close_content = hwloc__nolibxml_import_close_content;
  state->parent = NULL;
  nstate->closed = 0;
  nstate->tagbuffer = end;
  nstate->tagname = tagname;
  nstate->attrbuffer = NULL;
  return 0; /* success */

 failed:
  return -1; /* failed */
}

/* can be called at the end of the import (to cleanup things early),
 * or by backend_exit() if load failed for other reasons.
 */
static void
hwloc_nolibxml_free_buffers(struct hwloc_xml_backend_data_s *bdata)
{
  struct hwloc__nolibxml_backend_data_s *nbdata = bdata->data;
  if (nbdata->buffer) {
    free(nbdata->buffer);
    nbdata->buffer = NULL;
  }
}

static void
hwloc_nolibxml_look_done(struct hwloc_xml_backend_data_s *bdata, int result)
{
  hwloc_nolibxml_free_buffers(bdata);

  if (result < 0 && hwloc__xml_verbose())
    fprintf(stderr, "Failed to parse XML input with the minimalistic parser. If it was not\n"
	    "generated by hwloc, try enabling full XML support with libxml2.\n");
}

/********************
 * Backend routines *
 ********************/

static void
hwloc_nolibxml_backend_exit(struct hwloc_xml_backend_data_s *bdata)
{
  struct hwloc__nolibxml_backend_data_s *nbdata = bdata->data;
  hwloc_nolibxml_free_buffers(bdata);
  free(nbdata);
}

static int
hwloc_nolibxml_read_file(const char *xmlpath, char **bufferp, size_t *buflenp)
{
  FILE * file;
  size_t buflen, offset, readlen;
  struct stat statbuf;
  char *buffer, *tmp;
  size_t ret;

  if (!strcmp(xmlpath, "-"))
    xmlpath = "/dev/stdin";

  file = fopen(xmlpath, "r");
  if (!file)
    goto out;

  /* find the required buffer size for regular files, or use 4k when unknown, we'll realloc later if needed */
  buflen = 4096;
  if (!stat(xmlpath, &statbuf))
    if (S_ISREG(statbuf.st_mode))
      buflen = statbuf.st_size+1; /* one additional byte so that the first fread() gets EOF too */

  buffer = malloc(buflen+1); /* one more byte for the ending \0 */
  if (!buffer)
    goto out_with_file;

  offset = 0; readlen = buflen;
  while (1) {
    ret = fread(buffer+offset, 1, readlen, file);

    offset += ret;
    buffer[offset] = 0;

    if (ret != readlen)
      break;

    buflen *= 2;
    tmp = realloc(buffer, buflen+1);
    if (!tmp)
      goto out_with_buffer;
    buffer = tmp;
    readlen = buflen/2;
  }

  fclose(file);
  *bufferp = buffer;
  *buflenp = offset+1;
  return 0;

 out_with_buffer:
  free(buffer);
 out_with_file:
  fclose(file);
 out:
  return -1;
}

static int
hwloc_nolibxml_backend_init(struct hwloc_xml_backend_data_s *bdata,
			    const char *xmlpath, const char *xmlbuffer, int xmlbuflen)
{
  struct hwloc__nolibxml_backend_data_s *nbdata = malloc(sizeof(*nbdata));

  if (!nbdata)
    goto out;
  bdata->data = nbdata;

  if (xmlbuffer) {
    nbdata->buffer = malloc(xmlbuflen+1);
    if (!nbdata->buffer)
      goto out_with_nbdata;
    nbdata->buflen = xmlbuflen+1;
    memcpy(nbdata->buffer, xmlbuffer, xmlbuflen);
    nbdata->buffer[xmlbuflen] = '\0';

  } else {
    int err = hwloc_nolibxml_read_file(xmlpath, &nbdata->buffer, &nbdata->buflen);
    if (err < 0)
      goto out_with_nbdata;
  }

  bdata->look_init = hwloc_nolibxml_look_init;
  bdata->look_done = hwloc_nolibxml_look_done;
  bdata->backend_exit = hwloc_nolibxml_backend_exit;
  return 0;

out_with_nbdata:
  free(nbdata);
out:
  return -1;
}

static int
hwloc_nolibxml_import_diff(struct hwloc__xml_import_state_s *state,
			   const char *xmlpath, const char *xmlbuffer, int xmlbuflen,
			   hwloc_topology_diff_t *firstdiffp, char **refnamep)
{
  hwloc__nolibxml_import_state_data_t nstate = (void*) state->data;
  struct hwloc__xml_import_state_s childstate;
  char *refname = NULL;
  char *buffer, *tmp, *tag;
  size_t buflen;
  int ret;

  HWLOC_BUILD_ASSERT(sizeof(*nstate) <= sizeof(state->data));

  if (xmlbuffer) {
    buffer = malloc(xmlbuflen);
    if (!buffer)
      goto out;
    memcpy(buffer, xmlbuffer, xmlbuflen);
    buflen = xmlbuflen;

  } else {
    ret = hwloc_nolibxml_read_file(xmlpath, &buffer, &buflen);
    if (ret < 0)
      goto out;
  }

  /* skip headers */
  tmp = buffer;
  while (!strncmp(tmp, "<?xml ", 6) || !strncmp(tmp, "<!DOCTYPE ", 10)) {
    tmp = strchr(tmp, '\n');
    if (!tmp)
      goto out_with_buffer;
    tmp++;
  }

  state->global->next_attr = hwloc__nolibxml_import_next_attr;
  state->global->find_child = hwloc__nolibxml_import_find_child;
  state->global->close_tag = hwloc__nolibxml_import_close_tag;
  state->global->close_child = hwloc__nolibxml_import_close_child;
  state->global->get_content = hwloc__nolibxml_import_get_content;
  state->global->close_content = hwloc__nolibxml_import_close_content;
  state->parent = NULL;
  nstate->closed = 0;
  nstate->tagbuffer = tmp;
  nstate->tagname = NULL;
  nstate->attrbuffer = NULL;

  /* find root */
  ret = hwloc__nolibxml_import_find_child(state, &childstate, &tag);
  if (ret < 0)
    goto out_with_buffer;
  if (!tag || strcmp(tag, "topologydiff"))
    goto out_with_buffer;

  while (1) {
    char *attrname, *attrvalue;
    if (hwloc__nolibxml_import_next_attr(&childstate, &attrname, &attrvalue) < 0)
      break;
    if (!strcmp(attrname, "refname")) {
      free(refname);
      refname = strdup(attrvalue);
    } else
      goto out_with_buffer;
  }

  ret = hwloc__xml_import_diff(&childstate, firstdiffp);
  if (refnamep && !ret)
    *refnamep = refname;
  else
    free(refname);

  free(buffer);
  return ret;

out_with_buffer:
  free(buffer);
  free(refname);
out:
  return -1;
}

/*******************
 * Export routines *
 *******************/

typedef struct hwloc__nolibxml_export_state_data_s {
  char *buffer; /* (moving) buffer where to write */
  size_t written; /* how many bytes were written (or would have be written if not truncated) */
  size_t remaining; /* how many bytes are still available in the buffer */
  unsigned indent; /* indentation level for the next line */
  unsigned nr_children;
  unsigned has_content;
} __hwloc_attribute_may_alias * hwloc__nolibxml_export_state_data_t;

static void
hwloc__nolibxml_export_update_buffer(hwloc__nolibxml_export_state_data_t ndata, int res)
{
  if (res >= 0) {
    ndata->written += res;
    if (res >= (int) ndata->remaining)
      res = ndata->remaining>0 ? (int)ndata->remaining-1 : 0;
    ndata->buffer += res;
    ndata->remaining -= res;
  }
}

static char *
hwloc__nolibxml_export_escape_string(const char *src)
{
  size_t fulllen, sublen;
  char *escaped, *dst;

  fulllen = strlen(src);

  sublen = strcspn(src, "\n\r\t\"<>&");
  if (sublen == fulllen)
    return NULL; /* nothing to escape */

  escaped = malloc(fulllen*6+1); /* escaped chars are replaced by at most 6 char */
  dst = escaped;

  memcpy(dst, src, sublen);
  src += sublen;
  dst += sublen;

  while (*src) {
    int replen;
    switch (*src) {
    case '\n': strcpy(dst, "&#10;");  replen=5; break;
    case '\r': strcpy(dst, "&#13;");  replen=5; break;
    case '\t': strcpy(dst, "&#9;");   replen=4; break;
    case '\"': strcpy(dst, "&quot;"); replen=6; break;
    case '<':  strcpy(dst, "&lt;");   replen=4; break;
    case '>':  strcpy(dst, "&gt;");   replen=4; break;
    case '&':  strcpy(dst, "&amp;");  replen=5; break;
    default: replen=0; break;
    }
    dst+=replen; src++;

    sublen = strcspn(src, "\n\r\t\"<>&");
    memcpy(dst, src, sublen);
    src += sublen;
    dst += sublen;
  }

  *dst = 0;
  return escaped;
}

static void
hwloc__nolibxml_export_new_child(hwloc__xml_export_state_t parentstate,
				 hwloc__xml_export_state_t state,
				 const char *name)
{
  hwloc__nolibxml_export_state_data_t npdata = (void *) parentstate->data;
  hwloc__nolibxml_export_state_data_t ndata = (void *) state->data;
  int res;

  assert(!npdata->has_content);
  if (!npdata->nr_children) {
    res = hwloc_snprintf(npdata->buffer, npdata->remaining, ">\n");
    hwloc__nolibxml_export_update_buffer(npdata, res);
  }
  npdata->nr_children++;

  state->parent = parentstate;
  state->new_child = parentstate->new_child;
  state->new_prop = parentstate->new_prop;
  state->add_content = parentstate->add_content;
  state->end_object = parentstate->end_object;
  state->global = parentstate->global;

  ndata->buffer = npdata->buffer;
  ndata->written = npdata->written;
  ndata->remaining = npdata->remaining;
  ndata->indent = npdata->indent + 2;

  ndata->nr_children = 0;
  ndata->has_content = 0;

  res = hwloc_snprintf(ndata->buffer, ndata->remaining, "%*s<%s", (int) npdata->indent, "", name);
  hwloc__nolibxml_export_update_buffer(ndata, res);
}

static void
hwloc__nolibxml_export_new_prop(hwloc__xml_export_state_t state, const char *name, const char *value)
{
  hwloc__nolibxml_export_state_data_t ndata = (void *) state->data;
  char *escaped = hwloc__nolibxml_export_escape_string(value);
  int res = hwloc_snprintf(ndata->buffer, ndata->remaining, " %s=\"%s\"", name, escaped ? (const char *) escaped : value);
  hwloc__nolibxml_export_update_buffer(ndata, res);
  free(escaped);
}

static void
hwloc__nolibxml_export_end_object(hwloc__xml_export_state_t state, const char *name)
{
  hwloc__nolibxml_export_state_data_t ndata = (void *) state->data;
  hwloc__nolibxml_export_state_data_t npdata = (void *) state->parent->data;
  int res;

  assert (!(ndata->has_content && ndata->nr_children));
  if (ndata->has_content) {
    res = hwloc_snprintf(ndata->buffer, ndata->remaining, "</%s>\n", name);
  } else if (ndata->nr_children) {
    res = hwloc_snprintf(ndata->buffer, ndata->remaining, "%*s</%s>\n", (int) npdata->indent, "", name);
  } else {
    res = hwloc_snprintf(ndata->buffer, ndata->remaining, "/>\n");
  }
  hwloc__nolibxml_export_update_buffer(ndata, res);

  npdata->buffer = ndata->buffer;
  npdata->written = ndata->written;
  npdata->remaining = ndata->remaining;
}

static void
hwloc__nolibxml_export_add_content(hwloc__xml_export_state_t state, const char *buffer, size_t length __hwloc_attribute_unused)
{
  hwloc__nolibxml_export_state_data_t ndata = (void *) state->data;
  int res;

  assert(!ndata->nr_children);
  if (!ndata->has_content) {
    res = hwloc_snprintf(ndata->buffer, ndata->remaining, ">");
    hwloc__nolibxml_export_update_buffer(ndata, res);
  }
  ndata->has_content = 1;

  res = hwloc_snprintf(ndata->buffer, ndata->remaining, "%s", buffer);
  hwloc__nolibxml_export_update_buffer(ndata, res);
}

static size_t
hwloc___nolibxml_prepare_export(hwloc_topology_t topology, struct hwloc__xml_export_data_s *edata,
				char *xmlbuffer, int buflen, unsigned long flags)
{
  struct hwloc__xml_export_state_s state, childstate;
  hwloc__nolibxml_export_state_data_t ndata = (void *) &state.data;
  int v1export = flags & HWLOC_TOPOLOGY_EXPORT_XML_FLAG_V1;
  int res;

  HWLOC_BUILD_ASSERT(sizeof(*ndata) <= sizeof(state.data));

  state.new_child = hwloc__nolibxml_export_new_child;
  state.new_prop = hwloc__nolibxml_export_new_prop;
  state.add_content = hwloc__nolibxml_export_add_content;
  state.end_object = hwloc__nolibxml_export_end_object;
  state.global = edata;

  ndata->indent = 0;
  ndata->written = 0;
  ndata->buffer = xmlbuffer;
  ndata->remaining = buflen;

  ndata->nr_children = 1; /* don't close a non-existing previous tag when opening the topology tag */
  ndata->has_content = 0;

  res = hwloc_snprintf(ndata->buffer, ndata->remaining,
		 "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
		 "<!DOCTYPE topology SYSTEM \"%s\">\n", v1export ? "hwloc.dtd" : "hwloc2.dtd");
  hwloc__nolibxml_export_update_buffer(ndata, res);
  hwloc__nolibxml_export_new_child(&state, &childstate, "topology");
  if (!(flags & HWLOC_TOPOLOGY_EXPORT_XML_FLAG_V1))
    hwloc__nolibxml_export_new_prop(&childstate, "version", "2.0");
  hwloc__xml_export_topology (&childstate, topology, flags);
  hwloc__nolibxml_export_end_object(&childstate, "topology");

  return ndata->written+1; /* ending \0 */
}

static int
hwloc_nolibxml_export_buffer(hwloc_topology_t topology, struct hwloc__xml_export_data_s *edata,
			     char **bufferp, int *buflenp, unsigned long flags)
{
  char *buffer;
  size_t bufferlen, res;

  bufferlen = 16384; /* random guess for large enough default */
  buffer = malloc(bufferlen);
  if (!buffer)
    return -1;
  res = hwloc___nolibxml_prepare_export(topology, edata, buffer, (int)bufferlen, flags);

  if (res > bufferlen) {
    char *tmp = realloc(buffer, res);
    if (!tmp) {
      free(buffer);
      return -1;
    }
    buffer = tmp;
    hwloc___nolibxml_prepare_export(topology, edata, buffer, (int)res, flags);
  }

  *bufferp = buffer;
  *buflenp = (int)res;
  return 0;
}

static int
hwloc_nolibxml_export_file(hwloc_topology_t topology, struct hwloc__xml_export_data_s *edata,
			   const char *filename, unsigned long flags)
{
  FILE *file;
  char *buffer;
  int bufferlen;
  int ret;

  ret = hwloc_nolibxml_export_buffer(topology, edata, &buffer, &bufferlen, flags);
  if (ret < 0)
    return -1;

  if (!strcmp(filename, "-")) {
    file = stdout;
  } else {
    file = fopen(filename, "w");
    if (!file) {
      free(buffer);
      return -1;
    }
  }

  ret = (int)fwrite(buffer, 1, bufferlen-1 /* don't write the ending \0 */, file);
  if (ret == bufferlen-1) {
    ret = 0;
  } else {
    errno = ferror(file);
    ret = -1;
  }

  free(buffer);

  if (file != stdout)
    fclose(file);
  return ret;
}

static size_t
hwloc___nolibxml_prepare_export_diff(hwloc_topology_diff_t diff, const char *refname, char *xmlbuffer, int buflen)
{
  struct hwloc__xml_export_state_s state, childstate;
  hwloc__nolibxml_export_state_data_t ndata = (void *) &state.data;
  int res;

  HWLOC_BUILD_ASSERT(sizeof(*ndata) <= sizeof(state.data));

  state.new_child = hwloc__nolibxml_export_new_child;
  state.new_prop = hwloc__nolibxml_export_new_prop;
  state.add_content = hwloc__nolibxml_export_add_content;
  state.end_object = hwloc__nolibxml_export_end_object;
  state.global = NULL;

  ndata->indent = 0;
  ndata->written = 0;
  ndata->buffer = xmlbuffer;
  ndata->remaining = buflen;

  ndata->nr_children = 1; /* don't close a non-existing previous tag when opening the topology tag */
  ndata->has_content = 0;

  res = hwloc_snprintf(ndata->buffer, ndata->remaining,
		 "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
		 "<!DOCTYPE topologydiff SYSTEM \"hwloc2-diff.dtd\">\n");
  hwloc__nolibxml_export_update_buffer(ndata, res);
  hwloc__nolibxml_export_new_child(&state, &childstate, "topologydiff");
  if (refname)
    hwloc__nolibxml_export_new_prop(&childstate, "refname", refname);
  hwloc__xml_export_diff (&childstate, diff);
  hwloc__nolibxml_export_end_object(&childstate, "topologydiff");

  return ndata->written+1;
}

static int
hwloc_nolibxml_export_diff_buffer(hwloc_topology_diff_t diff, const char *refname, char **bufferp, int *buflenp)
{
  char *buffer;
  size_t bufferlen, res;

  bufferlen = 16384; /* random guess for large enough default */
  buffer = malloc(bufferlen);
  if (!buffer)
    return -1;
  res = hwloc___nolibxml_prepare_export_diff(diff, refname, buffer, (int)bufferlen);

  if (res > bufferlen) {
    char *tmp = realloc(buffer, res);
    if (!tmp) {
      free(buffer);
      return -1;
    }
    buffer = tmp;
    hwloc___nolibxml_prepare_export_diff(diff, refname, buffer, (int)res);
  }

  *bufferp = buffer;
  *buflenp = (int)res;
  return 0;
}

static int
hwloc_nolibxml_export_diff_file(hwloc_topology_diff_t diff, const char *refname, const char *filename)
{
  FILE *file;
  char *buffer;
  int bufferlen;
  int ret;

  ret = hwloc_nolibxml_export_diff_buffer(diff, refname, &buffer, &bufferlen);
  if (ret < 0)
    return -1;

  if (!strcmp(filename, "-")) {
    file = stdout;
  } else {
    file = fopen(filename, "w");
    if (!file) {
      free(buffer);
      return -1;
    }
  }

  ret = (int)fwrite(buffer, 1, bufferlen-1 /* don't write the ending \0 */, file);
  if (ret == bufferlen-1) {
    ret = 0;
  } else {
    errno = ferror(file);
    ret = -1;
  }

  free(buffer);

  if (file != stdout)
    fclose(file);
  return ret;
}

static void
hwloc_nolibxml_free_buffer(void *xmlbuffer)
{
  free(xmlbuffer);
}

/*************
 * Callbacks *
 *************/

static struct hwloc_xml_callbacks hwloc_xml_nolibxml_callbacks = {
  hwloc_nolibxml_backend_init,
  hwloc_nolibxml_export_file,
  hwloc_nolibxml_export_buffer,
  hwloc_nolibxml_free_buffer,
  hwloc_nolibxml_import_diff,
  hwloc_nolibxml_export_diff_file,
  hwloc_nolibxml_export_diff_buffer
};

static struct hwloc_xml_component hwloc_nolibxml_xml_component = {
  &hwloc_xml_nolibxml_callbacks,
  NULL
};

const struct hwloc_component hwloc_xml_nolibxml_component = {
  HWLOC_COMPONENT_ABI,
  NULL, NULL,
  HWLOC_COMPONENT_TYPE_XML,
  0,
  &hwloc_nolibxml_xml_component
};
