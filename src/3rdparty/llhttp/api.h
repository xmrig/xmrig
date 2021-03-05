#ifndef INCLUDE_LLHTTP_API_H_
#define INCLUDE_LLHTTP_API_H_
#ifdef __cplusplus
extern "C" {
#endif
#include <stddef.h>

typedef llhttp__internal_t llhttp_t;
typedef struct llhttp_settings_s llhttp_settings_t;

typedef int (*llhttp_data_cb)(llhttp_t*, const char *at, size_t length);
typedef int (*llhttp_cb)(llhttp_t*);

struct llhttp_settings_s {
  /* Possible return values 0, -1, `HPE_PAUSED` */
  llhttp_cb      on_message_begin;

  llhttp_data_cb on_url;
  llhttp_data_cb on_status;
  llhttp_data_cb on_header_field;
  llhttp_data_cb on_header_value;

  /* Possible return values:
   * 0  - Proceed normally
   * 1  - Assume that request/response has no body, and proceed to parsing the
   *      next message
   * 2  - Assume absence of body (as above) and make `llhttp_execute()` return
   *      `HPE_PAUSED_UPGRADE`
   * -1 - Error
   * `HPE_PAUSED`
   */
  llhttp_cb      on_headers_complete;

  llhttp_data_cb on_body;

  /* Possible return values 0, -1, `HPE_PAUSED` */
  llhttp_cb      on_message_complete;

  /* When on_chunk_header is called, the current chunk length is stored
   * in parser->content_length.
   * Possible return values 0, -1, `HPE_PAUSED`
   */
  llhttp_cb      on_chunk_header;
  llhttp_cb      on_chunk_complete;

  llhttp_cb      on_url_complete;
  llhttp_cb      on_status_complete;
  llhttp_cb      on_header_field_complete;
  llhttp_cb      on_header_value_complete;
};

/* Initialize the parser with specific type and user settings.
 *
 * NOTE: lifetime of `settings` has to be at least the same as the lifetime of
 * the `parser` here. In practice, `settings` has to be either a static
 * variable or be allocated with `malloc`, `new`, etc.
 */
void llhttp_init(llhttp_t* parser, llhttp_type_t type,
                 const llhttp_settings_t* settings);

/* Reset an already initialized parser back to the start state, preserving the
 * existing parser type, callback settings, user data, and lenient flags.
 */
void llhttp_reset(llhttp_t* parser);

/* Initialize the settings object */
void llhttp_settings_init(llhttp_settings_t* settings);

/* Parse full or partial request/response, invoking user callbacks along the
 * way.
 *
 * If any of `llhttp_data_cb` returns errno not equal to `HPE_OK` - the parsing
 * interrupts, and such errno is returned from `llhttp_execute()`. If
 * `HPE_PAUSED` was used as a errno, the execution can be resumed with
 * `llhttp_resume()` call.
 *
 * In a special case of CONNECT/Upgrade request/response `HPE_PAUSED_UPGRADE`
 * is returned after fully parsing the request/response. If the user wishes to
 * continue parsing, they need to invoke `llhttp_resume_after_upgrade()`.
 *
 * NOTE: if this function ever returns a non-pause type error, it will continue
 * to return the same error upon each successive call up until `llhttp_init()`
 * is called.
 */
llhttp_errno_t llhttp_execute(llhttp_t* parser, const char* data, size_t len);

/* This method should be called when the other side has no further bytes to
 * send (e.g. shutdown of readable side of the TCP connection.)
 *
 * Requests without `Content-Length` and other messages might require treating
 * all incoming bytes as the part of the body, up to the last byte of the
 * connection. This method will invoke `on_message_complete()` callback if the
 * request was terminated safely. Otherwise a error code would be returned.
 */
llhttp_errno_t llhttp_finish(llhttp_t* parser);

/* Returns `1` if the incoming message is parsed until the last byte, and has
 * to be completed by calling `llhttp_finish()` on EOF
 */
int llhttp_message_needs_eof(const llhttp_t* parser);

/* Returns `1` if there might be any other messages following the last that was
 * successfully parsed.
 */
int llhttp_should_keep_alive(const llhttp_t* parser);

/* Make further calls of `llhttp_execute()` return `HPE_PAUSED` and set
 * appropriate error reason.
 *
 * Important: do not call this from user callbacks! User callbacks must return
 * `HPE_PAUSED` if pausing is required.
 */
void llhttp_pause(llhttp_t* parser);

/* Might be called to resume the execution after the pause in user's callback.
 * See `llhttp_execute()` above for details.
 *
 * Call this only if `llhttp_execute()` returns `HPE_PAUSED`.
 */
void llhttp_resume(llhttp_t* parser);

/* Might be called to resume the execution after the pause in user's callback.
 * See `llhttp_execute()` above for details.
 *
 * Call this only if `llhttp_execute()` returns `HPE_PAUSED_UPGRADE`
 */
void llhttp_resume_after_upgrade(llhttp_t* parser);

/* Returns the latest return error */
llhttp_errno_t llhttp_get_errno(const llhttp_t* parser);

/* Returns the verbal explanation of the latest returned error.
 *
 * Note: User callback should set error reason when returning the error. See
 * `llhttp_set_error_reason()` for details.
 */
const char* llhttp_get_error_reason(const llhttp_t* parser);

/* Assign verbal description to the returned error. Must be called in user
 * callbacks right before returning the errno.
 *
 * Note: `HPE_USER` error code might be useful in user callbacks.
 */
void llhttp_set_error_reason(llhttp_t* parser, const char* reason);

/* Returns the pointer to the last parsed byte before the returned error. The
 * pointer is relative to the `data` argument of `llhttp_execute()`.
 *
 * Note: this method might be useful for counting the number of parsed bytes.
 */
const char* llhttp_get_error_pos(const llhttp_t* parser);

/* Returns textual name of error code */
const char* llhttp_errno_name(llhttp_errno_t err);

/* Returns textual name of HTTP method */
const char* llhttp_method_name(llhttp_method_t method);


/* Enables/disables lenient header value parsing (disabled by default).
 *
 * Lenient parsing disables header value token checks, extending llhttp's
 * protocol support to highly non-compliant clients/server. No
 * `HPE_INVALID_HEADER_TOKEN` will be raised for incorrect header values when
 * lenient parsing is "on".
 *
 * **(USE AT YOUR OWN RISK)**
 */
void llhttp_set_lenient_headers(llhttp_t* parser, int enabled);


/* Enables/disables lenient handling of conflicting `Transfer-Encoding` and
 * `Content-Length` headers (disabled by default).
 *
 * Normally `llhttp` would error when `Transfer-Encoding` is present in
 * conjunction with `Content-Length`. This error is important to prevent HTTP
 * request smuggling, but may be less desirable for small number of cases
 * involving legacy servers.
 *
 * **(USE AT YOUR OWN RISK)**
 */
void llhttp_set_lenient_chunked_length(llhttp_t* parser, int enabled);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif  /* INCLUDE_LLHTTP_API_H_ */
