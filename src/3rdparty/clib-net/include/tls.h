
/*
 * Copyright 2014 <yorkiefixer@gmail.com>
 */

#ifndef __TLS_H__
#define __TLS_H__

#include <openssl/ssl.h>
#include <openssl/err.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>
#include <buffer/buffer.h>

#define SSL_CHUNK_SIZE 512

typedef SSL_CTX tls_ctx;
typedef struct tls_s {
  SSL_CTX * ctx;
  SSL * ssl;
  BIO * bio_in;
  BIO * bio_out;
  buffer_t * buffer;
  int connected;
  char * data;
  char buf[SSL_CHUNK_SIZE]; /* internal usage */
} tls_t;

static const int X509_NAME_FLAGS = ASN1_STRFLGS_ESC_CTRL
                                 | ASN1_STRFLGS_ESC_MSB
                                 | XN_FLAG_SEP_MULTILINE
                                 | XN_FLAG_FN_SN;

/*
 * initialize the ssl
 */
void
ssl_init();

/*
 * destroy the ssl settings and internal tables
 */
void
ssl_destroy();

/*
 * create a context for ssl
 */
tls_ctx *
tls_ctx_new();

/*
 * create a tls instance
 */
tls_t *
tls_create(tls_ctx * ctx);

/*
 * shutdown tls
 */
int
tls_shutdown(tls_t * tls);

/*
 * destroy a tls instance
 */
int
tls_free(tls_t * tls);

/*
 * get peer certification info
 */
int
tls_get_peer_cert(tls_t * tls);

/*
 * do connect to tls
 */
int
tls_connect(tls_t * tls);

/*
 * handle error in bio
 */
int
tls_handle_bio_error(tls_t * tls, int err);

/*
 * handle error in ssl
 */
int
tls_handle_ssl_error(tls_t *tls, int err);

/*
 * a port in tls for `bio_read`
 */
int
tls_bio_read(tls_t * tls, int len);

/*
 * a port in tls for `bio_write`
 */
int
tls_bio_write(tls_t * tls, char * written, int len);

/*
 * read
 */
int
tls_read(tls_t * tls);

/*
 * write
 */
int
tls_write(tls_t * tls, char * written, int len);

/*
 * write a tls packet
 */
#define REQUEST_TLS_WRITE(name, cmd, read, req) do {                      \
  tls_write(req->tls, cmd);                                               \
  do {                                                                    \
    read = tls_bio_read(req->tls, 0);                                     \
    if (read > 0) {                                                       \
      REQUEST_WRITE(req, req->tls->buf, read, name);                      \
    }                                                                     \
  } while (read > 0);                                                     \
}                                                                         \
while (0)


#endif /* __TLS_H__ */