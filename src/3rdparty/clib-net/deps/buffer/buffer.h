
//
// buffer.h
//
// Copyright (c) 2012 TJ Holowaychuk <tj@vision-media.ca>
//

#ifndef BUFFER_H
#define BUFFER_H 1

#include <sys/types.h>

/*
 * Default buffer size.
 */

#ifndef BUFFER_DEFAULT_SIZE
#define BUFFER_DEFAULT_SIZE 64
#endif

/*
 * Buffer struct.
 */

typedef struct {
  size_t len;
  char *alloc;
  char *data;
} buffer_t;

// prototypes

buffer_t *
buffer_new();

buffer_t *
buffer_new_with_size(size_t n);

buffer_t *
buffer_new_with_string(char *str);

buffer_t *
buffer_new_with_string_length(char *str, size_t len);

buffer_t *
buffer_new_with_copy(char *str);

size_t
buffer_size(buffer_t *self);

size_t
buffer_length(buffer_t *self);

void
buffer_free(buffer_t *self);

int
buffer_prepend(buffer_t *self, char *str);

int
buffer_append(buffer_t *self, const char *str);

int
buffer_appendf(buffer_t *self, const char *format, ...);

int
buffer_append_n(buffer_t *self, const char *str, size_t len);

int
buffer_equals(buffer_t *self, buffer_t *other);

ssize_t
buffer_indexof(buffer_t *self, char *str);

buffer_t *
buffer_slice(buffer_t *self, size_t from, ssize_t to);

ssize_t
buffer_compact(buffer_t *self);

void
buffer_fill(buffer_t *self, int c);

void
buffer_clear(buffer_t *self);

void
buffer_trim_left(buffer_t *self);

void
buffer_trim_right(buffer_t *self);

void
buffer_trim(buffer_t *self);

void
buffer_print(buffer_t *self);

#define buffer_string(self) (self->data)

#endif
