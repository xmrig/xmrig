
#include <assert.h>
#include <uv.h>
#include "net.h"

net_t *
net_new(char * hostname, int port) {
  net_t * net = (net_t*) malloc(sizeof(net_t));
  net->loop = uv_default_loop();
  net->hostname = hostname;
  net->port = port;
  net->connected = 0;
  net->tls_established = 0;
  net->use_ssl = 0;
  net->conn_cb = NULL;
  net->read_cb = NULL;
  net->error_cb = NULL;
  net->close_cb = NULL;
  net->handle = (uv_tcp_t *) malloc(sizeof(uv_tcp_t));
  net->conn   = (uv_connect_t *) malloc(sizeof(uv_connect_t));
  net->handle->data
    = net->conn->data
    = (void *) net;

  return net;
}

#ifndef XMRIG_NO_TLS
int
net_set_tls(net_t * net, tls_ctx * ctx) {
  net->use_ssl = USE_SSL;
  net->tls = tls_create(ctx);
  return NET_OK;
}
#endif

int
net_connect(net_t * net) {
  net_resolve(net);
  return NET_OK;
}

void
net_close_cb(uv_handle_t *handle) {
  net_t * net = (net_t*) handle->data;

  if (net) {
    if (net->close_cb) {
      net->close_cb(net);
    }
  }
}

int
net_close(net_t * net, void (*cb)(uv_handle_t*)) {
  int r = net->connected;
  if (r == 1) {
    net->connected = 0;
    net->tls_established = 0;

#ifndef XMRIG_NO_TLS
    if (net->use_ssl) {
      tls_shutdown(net->tls);
    }
#endif

    if (uv_is_readable((uv_stream_t*)net->handle) == 1) {
      uv_read_stop((uv_stream_t*)net->handle);
    }

    if (uv_is_closing((const uv_handle_t *) net->handle) == 0) {
      uv_close((uv_handle_t *)net->handle, net_close_cb);
    }

#ifndef XMRIG_NO_TLS
    if (net->use_ssl) {
      tls_free(net->tls);
    }
#endif
  } else{
      if (net->close_cb) {
          net->close_cb(net);
      }
  }

  return r;
}

int
net_free(net_t * net) {
  if (net != NULL) {
    free(net);
    net = NULL;
  }

  return NET_OK;
}

void
net_free_cb(uv_handle_t * handle) {
  net_t * net = (net_t *) handle->data;
  if (net->handle != NULL) {
    free(net->handle);
    net->handle = NULL;
  }
  if (net->conn != NULL) {
    free(net->conn);
    net->conn = NULL;
  }
  if (net->resolver != NULL) {
    free(net->resolver);
    net->resolver = NULL;
  }
  if (net != NULL) {
    free(net);
    net = NULL;
  }
}

int
net_resolve(net_t * net) {
  net_ai hints;
  int ret;
  char buf[6];

  snprintf(buf, sizeof(buf), "%d", net->port);
  memset(&hints, 0, sizeof(struct addrinfo));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;

  net->resolver = malloc(sizeof(uv_getaddrinfo_t));
  if (!net->resolver) {
    /*
     * TODO(Yorkie): depent parital handles
     */
    return -1;
  }

  net->resolver->data = (void *) net;
  ret = uv_getaddrinfo(net->loop, net->resolver,
    net_resolve_cb, net->hostname, NULL, &hints);

  return ret;
}

void
net_resolve_cb(uv_getaddrinfo_t *rv, int err, net_ai * ai) {
  net_t * net = (net_t*) rv->data;
  char addr[INET6_ADDRSTRLEN];
  int ret;
  struct sockaddr_in dest;

  if (err != 0) {
    if (net->error_cb) {
      net->error_cb(net, err, (char *) uv_strerror(err));
    } else {
      printf("error(%s:%d) %s", net->hostname, net->port, (char *) uv_strerror(err));
      net_free(net);
    }
    return;
  }

  uv_ip4_name((socketPair_t *) ai->ai_addr, addr, INET6_ADDRSTRLEN);
  ret = uv_ip4_addr(addr, net->port, &dest);

  if (ret != 0) {
    if (net->error_cb) {
      net->error_cb(net, ret, (char *) uv_strerror(err));
    } else {
      printf("error(%s:%d) %s", net->hostname, net->port, (char *) uv_strerror(err));
      net_free(net);
    }
    return;
  }

  /*
   * create tcp instance.
   */
  uv_tcp_init(net->loop, net->handle);
  uv_tcp_nodelay(net->handle, 1);

#   ifndef WIN32
  uv_tcp_keepalive(net->handle, 1, 60);
#   endif

  ret = uv_tcp_connect(net->conn, net->handle, (const struct sockaddr*) &dest, net_connect_cb);
  if (ret != NET_OK) {
    if (net->error_cb) {
      net->error_cb(net, ret, (char *) uv_strerror(err));
    } else {
      printf("error(%s:%d) %s", net->hostname, net->port, (char *) uv_strerror(err));
      net_free(net);
    }
    return;
  }

  /*
   * free
   */
  uv_freeaddrinfo(ai);
}

void
net_connect_cb(uv_connect_t *conn, int err) {
  net_t * net = (net_t *) conn->data;
  int read;

  if (err < 0) {
    if (net->error_cb) {
      net->error_cb(net, err, (char *) uv_strerror(err));
    } else {
      printf("error(%s:%d) %s", net->hostname, net->port, (char *) uv_strerror(err));
      net_free(net);
    }
    return;
  }

  /*
   * change the `connected` state
   */
  net->connected = 1;

  /*
   * read buffers via uv
   */
  uv_read_start((uv_stream_t *) net->handle, net_alloc, net_read);

  /*
   * call `onConnect`, the tcp connection has been
   *  established in user-land.
   */
  if (net->use_ssl == NOT_SSL && net->conn_cb != NULL) {
    net->conn_cb(net);
  }

#ifndef XMRIG_NO_TLS
  /*
   * Handle TLS Partial
   */
  if (net->use_ssl == USE_SSL && tls_connect(net->tls) == NET_OK) {
    do {
      read = tls_bio_read(net->tls, 0);
      if (read > 0) {
        char* buf = (char *) calloc(read, 1);
        memset(buf, 0, read);
        memcpy(buf, net->tls->buf, read);
        uv_buf_t uvbuf = uv_buf_init(buf, read);
        uv_try_write((uv_stream_t*)net->handle, &uvbuf, 1);
        free(buf);
      }
    } while (read > 0);
  }
#endif
}

void
net_alloc(uv_handle_t* handle, size_t size, uv_buf_t* buf) {
  buf->base = (char *) calloc(size, 1);
  buf->len = size;
}

void
net_read(uv_stream_t* handle, ssize_t nread, const uv_buf_t* buf) {
  net_t * net = (net_t *) handle->data;

  if (nread < 0) {
    if (net->error_cb) {
      net->error_cb(net, nread, (char *) uv_strerror(nread));
    } else {
      printf("error(%s:%d) %s", net->hostname, net->port, (char *) uv_strerror(nread));
      net_free(net);
    }
    return;
  }

#ifndef XMRIG_NO_TLS
  /* 
   * BIO Return rule:
   * All these functions return either the amount of data successfully
   * read or written (if the return value is positive) or that no data 
   * was successfully read or written if the result is 0 or -1. If the 
   * return value is -2 then the operation is not implemented in the specific BIO type.
   */
  if (net->use_ssl) {
    tls_bio_write(net->tls, buf->base, nread);
    free(buf->base);

    int read = 0;
    int stat = tls_read(net->tls);
    if (stat == 1) {
      /* 
       * continue: Say hello
       */
      do {
        read = tls_bio_read(net->tls, 0);
        if (read > 0) {
          char* buf2 = (char *) calloc(read, 1);
          memset(buf2, 0, read);
          memcpy(buf2, net->tls->buf, read);
          uv_buf_t uvbuf = uv_buf_init(buf2, read);
          uv_try_write((uv_stream_t*)net->handle, &uvbuf, 1);
          free(buf2);
        }
      } while (read > 0);

    } else {
      /*
       * SSL Connection is created
       * Here need to call user-land callback
       */
      if (!net->tls_established) {
        net->tls_established = 1;
        if (net->conn_cb != NULL) {
          net->conn_cb(net);
        }
      }

      /*
       * read buffer
       */
      if (stat == 0) {
        if (buffer_string(net->tls->buffer) > 0)
          // uv_read_stop((uv_stream_t*)net->handle);

        if (net->read_cb != NULL && net->connected && net->tls_established) {
          net->read_cb(net, buffer_length(net->tls->buffer),
                            buffer_string(net->tls->buffer));
        }
      }
    }
    return;
  }
#endif

  /*
   * TCP Part, no SSL, just proxy of uv.
   */
  //uv_read_stop(handle);
  buf->base[nread] = 0;
  if (net->read_cb != NULL) {
    net->read_cb(net, nread, buf->base);
    free(buf->base);
  }
}

int
net_write(net_t * net, char * buf) {
  return net_write2(net, buf, strlen(buf));
}

int
net_write2(net_t * net, char * buf, unsigned int len) {
  uv_buf_t uvbuf;
  int read = 0;
  int res = NET_OK;

  switch (net->use_ssl) {
  case USE_SSL:
#ifndef XMRIG_NO_TLS
    tls_write(net->tls, buf, (int)len);
    do {
      read = tls_bio_read(net->tls, 0);
      if (read > 0) {
        uvbuf = uv_buf_init(net->tls->buf, read);
        res = uv_try_write((uv_stream_t*)net->handle, &uvbuf,1);
      }
    } while (read > 0);
    break;
#endif
  case NOT_SSL:
    uvbuf = uv_buf_init(buf, len);
    res = uv_try_write((uv_stream_t*)net->handle, &uvbuf, 1);
    break;
  }

  return res;
}

int
net_use_ssl(net_t * net) {
  return net->use_ssl == USE_SSL;
}

int
net_resume(net_t * net) {
  uv_read_start((uv_stream_t *)net->handle, net_alloc, net_read);
  return NET_OK;
}

int
net_pause(net_t * net) {
  uv_read_stop((uv_stream_t *)net->handle);
  return NET_OK;
}

int
net_set_error_cb(net_t * net, void * cb) {
  net->error_cb = cb;
  return NET_OK;
}
