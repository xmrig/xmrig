FROM ubuntu:latest

RUN apt-get update && \
	apt-get install software-properties-common git build-essential cmake libuv1-dev libssl-dev libboost-system-dev -y

RUN add-apt-repository ppa:jonathonf/gcc-7.1 && \
	apt-get update && \
	apt-get install gcc-7 g++-7 -y

RUN  git clone https://github.com/Bendr0id/xmrigCC.git && \
	cd xmrigCC && \
	cmake . -DCMAKE_C_COMPILER=gcc-7 -DCMAKE_CXX_COMPILER=g++-7 -DWITH_CC_SERVER=OFF -DWITH_HTTPD=OFF && \
	make 
	
COPY Dockerfile /Dockerfile

ENTRYPOINT  ["/xmrigCC/xmrigDaemon"]
