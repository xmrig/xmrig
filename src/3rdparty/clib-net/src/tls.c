
/*
 * Copyright 2014 <yorkiefixer@gmail.com>
 */

#include <assert.h>
#include "tls.h"

void
ssl_init() {
  SSL_library_init();
  OpenSSL_add_all_algorithms();
  SSL_load_error_strings();
  ERR_load_crypto_strings();
}

void
ssl_destroy() {
  EVP_cleanup();
  ERR_free_strings();
}

tls_ctx *
tls_ctx_new() {
  tls_ctx *ctx = SSL_CTX_new(SSLv23_method());
#ifdef SSL_OP_NO_COMPRESSION
  SSL_CTX_set_options(ctx, SSL_OP_NO_COMPRESSION);
#endif
  
  return ctx;
}

tls_t *
tls_create(tls_ctx *ctx) {
  tls_t * tls = (tls_t *) malloc(sizeof(tls_t)+1);
  if (tls == NULL)
    fprintf(stderr, "tls> %s", "Out of Memory");

  tls->ctx = ctx;
  tls->ssl = SSL_new(tls->ctx);
  tls->bio_in = BIO_new(BIO_s_mem());
  tls->bio_out = BIO_new(BIO_s_mem());
  tls->connected = -1;
  tls->buffer = buffer_new();

#ifdef SSL_MODE_RELEASE_BUFFERS
  long mode = SSL_get_mode(tls->ssl);
  SSL_set_mode(tls->ssl, mode | SSL_MODE_RELEASE_BUFFERS);
#endif

  if (tls->ssl == NULL)
    printf("tls> %s", "Out of Memory");

  SSL_set_connect_state(tls->ssl);
  SSL_set_bio(tls->ssl, tls->bio_in, tls->bio_out);
  return tls;
}

int
tls_shutdown(tls_t * tls) {
  assert(tls != NULL);
  if (SSL_shutdown(tls->ssl) == 0) {
    SSL_shutdown(tls->ssl);
  }
  return 0;
}

int
tls_free(tls_t * tls) {
  if (tls->ssl) {
    SSL_free(tls->ssl);
    SSL_CTX_free(tls->ctx);
    tls->ssl = NULL;
  }

  buffer_free(tls->buffer);
  free(tls);
  return 0;
}

int
tls_get_peer_cert(tls_t *tls) {
  X509* peer_cert = SSL_get_peer_certificate(tls->ssl);
  if (peer_cert != NULL) {
    /*
     * TODO(Yorkie): This function is used just for debug
     */
    X509_free(peer_cert);
  }
  return 0;
}

int
tls_connect(tls_t *tls) {
  int rv;
  int er;

  rv = SSL_do_handshake(tls->ssl);
  if (rv == 1) {
    /* 
     * `SSL_do_handshake()` could not return 1,
     *  that caused no message could be returned in `SSL_get_error()`.
     *  TODO(Yorkie): handle error, exit?
     */
    return -1;
  }

  if (!SSL_is_init_finished(tls->ssl))
    er = SSL_connect(tls->ssl);
  else
    return -1;

  if (er < 0 && SSL_get_error(tls->ssl, er) == SSL_ERROR_WANT_READ)
    return 0;
  else
    return -1;
}

int
tls_handle_bio_error(tls_t *tls, int err) {
  int rv;
  int retry = BIO_should_retry(tls->bio_out);
  if (BIO_should_write(tls->bio_out))
    rv = -retry;
  else if (BIO_should_read(tls->bio_out))
    rv = -retry;
  else {
    char ssl_error_buf[512];
    ERR_error_string_n(err, ssl_error_buf, sizeof(ssl_error_buf));
    fprintf(stderr, "[%p] BIO: read failed: (%d) %s\n", tls->ssl, err, ssl_error_buf);
    return err;
  }
  return err;
}

int
tls_handle_ssl_error(tls_t *tls, int err) {
  int ret;
  int rv = SSL_get_error(tls->ssl, err);
  switch (rv) {
    case SSL_ERROR_WANT_READ:
      ret = 1;
      break;
    default:
      ret = -2;
      break;
  }
  return ret;
}

int
tls_bio_read(tls_t *tls, int buf_len) {
  if (buf_len == 0) {
    buf_len = sizeof(tls->buf);
  }
  memset(tls->buf, 0, buf_len);

  int ret = BIO_read(tls->bio_out, tls->buf, buf_len);
  if (ret >= 0) {
    tls->buf[ret] = 0;
    return ret;
  } else {
    return tls_handle_bio_error(tls, ret);
  }
}

int
tls_bio_write(tls_t *tls, char *buf, int len) {
  int ret = BIO_write(tls->bio_in, buf, len);
  if (ret >= 0)
    return ret;
  else
    return tls_handle_bio_error(tls, ret);
}

int
tls_read(tls_t *tls) {
  int err;
  int ret;
  int read;

  int done = SSL_is_init_finished(tls->ssl);
  if (!done) {
    err = SSL_connect(tls->ssl);
    if (err <= 0) {
      return tls_handle_ssl_error(tls, err);
    }

    /*
     * TODO(Yorkie): returns not 1 nor < 0
     */
    assert(err == 1);
  }

  /* finished */
  buffer_clear(tls->buffer);

  ret = -1;
  do {
    read = SSL_read(tls->ssl, tls->buf, SSL_CHUNK_SIZE);
    if (read > 0) {
      ret = 0;
      tls->buf[read] = 0;
      buffer_append(tls->buffer, tls->buf);
    } else {
      tls_handle_ssl_error(tls, read);
    }
  } while (read > 0);

  if (tls->connected == -1) {
    tls->connected = 1;
  } else {
    ret = 0;
  }
  return ret;
}

int
tls_write(tls_t *tls, char *buf_w, int len) {
  return SSL_write(tls->ssl, buf_w, len);
}