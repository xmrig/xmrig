FROM ubuntu:latest

RUN apt-get update && \
	apt install software-properties-common git build-essential libbz2-dev cmake libuv1-dev libssl-dev wget gcc g++ -y && \
	apt clean && \
   	rm -rf /var/lib/apt/lists/*

RUN wget https://dl.bintray.com/boostorg/release/1.66.0/source/boost_1_66_0.tar.gz \
    && tar xfz boost_1_66_0.tar.gz \
    && cd boost_1_66_0 \
    && ./bootstrap.sh --with-libraries=system \
    && ./b2 link=static runtime-link=static install \
    && cd .. && rm -rf boost_1_66_0 && rm boost_1_66_0.tar.gz && ldconfig

RUN  git clone https://github.com/Bendr0id/xmrigCC.git && \
	cd xmrigCC && \
	cmake . -DWITH_CC_SERVER=OFF -DWITH_HTTPD=OFF && \
	make 
	
COPY Dockerfile /Dockerfile

ENTRYPOINT  ["/xmrigCC/xmrigDaemon"]
