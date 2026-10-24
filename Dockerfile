# Build in disposable container so run-time container is small
FROM alpine:latest as build

# Build from master by default but allow build time specification
ARG ref=master
ENV my_ref=$ref

# Developers may wish to specify an alternate repository for source
ARG repo=https://github.com/xmrig/xmrig.git
ENV my_repo=$repo

RUN set -ex && \
        # testing required for hwloc
	echo @testing http://nl.alpinelinux.org/alpine/edge/testing >> /etc/apk/repositories

RUN set -ex && \
	apk --no-cache --update add \
	    coreutils file grep openssl tar binutils \
	    cmake g++ git linux-headers libpthread-stubs make hwloc-dev@testing \
	    libuv-dev openssl-dev

WORKDIR /usr/local/src

RUN set -ex && \
    	git clone $my_repo xmrig && \
	cd xmrig && git checkout $my_ref && \
	cmake -B build && \
	cd build && \
	make 

# runtime container
FROM alpine:latest

RUN set -ex && \
        # testing required for hwloc
	echo @testing http://nl.alpinelinux.org/alpine/edge/testing >> /etc/apk/repositories

RUN set -ex && \
	apk --no-cache --update add \
	# required libraries packages
		openssl libuv hwloc@testing

COPY --from=build /usr/local/src/xmrig/build/xmrig /bin/

ENTRYPOINT ["/bin/xmrig"]
