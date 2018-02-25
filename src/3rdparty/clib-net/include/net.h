
/*
 * Copyright 2014 <yorkiefixer@gmail.com>
 */

#ifndef __NET_H__
#define __NET_H__

#include <uv.h>
#include <buffer/buffer.h>

#ifndef XMRIG_NO_TLS
#include "tls.h"
#endif

#define NOT_SSL 0x00
#define USE_SSL 0x01

#define NET_OK 0
#define NET_VERSION_MAJOR 0
#define NET_VERSION_MINOR 1
#define NET_VERSION_PATCH 2
#define NET_VERSION_IS_RELEASE 0

typedef struct net_s net_t;
typedef struct addrinfo net_ai;
typedef struct sockaddr_in socketPair_t;

#define NET_FIELDS                    \
  NET_CONNECTION_FIELDS               \
  NET_UV_FIELDS                       \
  NET_TLS_FIELDS                      \

#define NET_CONNECTION_FIELDS         \
  char *hostname;                     \
  int   port;                         \
  int   connected;                    \


#define NET_UV_FIELDS                 \
  uv_getaddrinfo_t *resolver;         \
  uv_loop_t        *loop;             \
  uv_tcp_t         *handle;           \
  uv_connect_t     *conn;             \

#ifndef XMRIG_NO_TLS
    #define NET_TLS_FIELDS              \
        int     use_ssl;                \
        int     tls_established;        \
        tls_t   *tls;
#else
    #define NET_TLS_FIELDS              \
        int     use_ssl;                \
        int     tls_established;
#endif


struct net_s {
  NET_FIELDS;
  void  *data;
  void (*conn_cb)(net_t*);
  void (*read_cb)(net_t*, size_t, char*);
  void (*error_cb)(net_t*, int, char*);
  void (*close_cb)(uv_handle_t*);
};

/*
 * Create an new network.
 */
net_t *
net_new(char * hostname, int port);


#ifndef XMRIG_NO_TLS
/*
 * Set SSL's Context
 */
int
net_set_tls(net_t * net, tls_ctx * ctx);
#endif
/*
 * Do connect to new
 */
int
net_connect(net_t * net);

/*
 * Just close the holding connection
 */
int
net_close(net_t * net, void (*cb)(uv_handle_t*));

/*
 * free connection
 */
int
net_free(net_t * net);

/*
 * real free function
 */
void
net_free_cb(uv_handle_t * handle);

/*
 * DNS resolve
 */
int
net_resolve(net_t * net);

/*
 * DNS -> IP done, and call `net_resolve_cb`
 */
void
net_resolve_cb(uv_getaddrinfo_t *rv, int stat, net_ai * ai);

/*
 * connect created, and call `net_connect_cb`
 */
void 
net_connect_cb(uv_connect_t *conn, int stat);

/*
 * realloc buffer before you read
 */
void
net_alloc(uv_handle_t* handle, size_t size, uv_buf_t* buf);

/*
 * read buffer
 */
void
net_read(uv_stream_t* handle, ssize_t nread, const uv_buf_t* buf);

/*
 * write buffer
 */
int
net_write(net_t * net, char * buf);

/*
 * write buffer with length specified
 */
int
net_write2(net_t * net, char * buf, unsigned int len);

/*
 * return use_ssl
 */
int
net_use_ssl(net_t * net);

/*
 * continue to read after on data
 */
int
net_resume(net_t * net);

/*
 * pause read
 */
int
net_pause(net_t * net);

/*
 * set error_cb
 */
int
net_set_error_cb(net_t * net, void * cb);

/*
 * write buffer, and call `net_write_cb`.
 */
void
net_write_cb(uv_write_t *writer, int stat);

#endif /* __NET_H__ */