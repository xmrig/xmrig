
0.4.0 / 2015-01-05
==================

  * refactor
  * buffer: Remove printf() statement
  * Add `buffer_appendf()` (#10, marcomorain)

0.3.0 / 2014-12-24
==================

  * travis: Fail the build if any memory is leaked
  * test: Fix memory leaks
  * travis: setup
  * Add `buffer_append_n(buffer_t *, const char *, size_t)`

0.2.1 / 2014-12-23
==================

  * fix header guard
  * fix compilation on linux
  * Add missing null terminator after realloc (buffer resize)
  * Make it safe to always use `data` as a character string

0.2.0 / 2013-01-05 
==================

  * add print_buffer()
  * add buffer_compact() 

0.1.0 / 2012-12-26 
==================

  * add trim functions
  * add buffer_clear(buffer_t *self)
  * add buffer_fill(buffer_t *self, int c)
  * add buffer_new_with_string_length(char *str, size_t len)
  * add buffer_new_with_copy(char *str)
  * add buffer_indexof()

