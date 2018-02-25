
# buffer

  Tiny C string manipulation library.

## Installation

  Install with [clib](https://github.com/clibs/clib):

```
$ clib install clibs/buffer
```

## API

```c
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
buffer_append(buffer_t *self, char *str);

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
```

## License

  MIT
