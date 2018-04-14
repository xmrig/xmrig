FROM ubuntu:latest

RUN add-apt-repository ppa:jonathonf/gcc-7.1 && \
	apt-get update && \
	apt-get install gcc-7 g++-7 software-properties-common git build-essential libbz2-dev cmake libuv1-dev libssl-dev wget -y && \
	apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://dl.bintray.com/boostorg/release/1.66.0/source/boost_1_66_0.tar.gz \
    && tar xfz oost_1_66_0.tar.gz \
    && rm boost_1_66_0.tar.gz \
    && cd boost_1_66_0 \
    && ./bootstrap.sh \
    && ./b2 --with-libraries=system -j 4 link=shared runtime-link=shared install \
    && cd .. && rm -rf boost_1_66_0 && ldconfig

RUN  git clone https://github.com/Bendr0id/xmrigCC.git && \
	cd xmrigCC && \
	cmake . -DCMAKE_C_COMPILER=gcc-7 -DCMAKE_CXX_COMPILER=g++-7 -DWITH_CC_SERVER=OFF -DWITH_HTTPD=OFF && \
	make 
	
COPY Dockerfile /Dockerfile

ENTRYPOINT  ["/xmrigCC/xmrigDaemon"]
