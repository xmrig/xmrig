
CC     ?= gcc
SRC     = $(wildcard src/*.c) tests/simple.c
DEPS    = $(wildcard deps/*/*.c)
CFLAGS  = -std=c++11\
          -Iinclude\
          -Isrc\
          -Ideps\
          -Ideps/libuv/include\
          -Ldeps/libuv/build/Release\
          -Wall -Wno-unused-function
LDFLAGS = -lcrypto -lssl -luv

test: $(SRC)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(DEPS) $(LDFLAGS)

.PHONY: test
