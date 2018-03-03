
# net-client

Simple network client

### Requirement

* libuv 0.10.x

* buffer 0.2.0

### Installation

```sh
$ clib install clibs/net
$ git clone https://github.com/joyent/libuv.git deps/libuv
$ checkout v0.10.25
```

### Run tests

```sh
make test
./test
```

### Example

```c
static void 
imap_parser(net_t * net, size_t read, char * buf) {
  printf("%s\n", buf);
  printf("%zu\n", read);
}

int 
main(int argc, char *argv[]) {
  ssl_init();
  tls_ctx * ctx = tls_ctx_new();
  net_t * net = net_new("imap.gmail.com", 993); 
  net->onRead = imap_parser;

  // convert this socket to ssl
  net_set_tls(net, ctx);
  net_connect(net);

  uv_run(net->loop, UV_RUN_DEFAULT);
}
```

### License

MIT
