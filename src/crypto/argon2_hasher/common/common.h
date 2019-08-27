//
// Created by Haifa Bogdan Adnan on 04/08/2018.
//

#ifndef ARGON2_COMMON_H
#define ARGON2_COMMON_H

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <string>
#include <vector>
#include <queue>
#include <list>
#include <map>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <regex>
#include <random>
#include <algorithm>
#include <thread>
#include <mutex>
#include <chrono>

#include <cmath>
#include <signal.h>

#include <dlfcn.h>
#include "DLLImport.h"

#ifndef _WIN64
#include <unistd.h>
#include <sys/time.h>

#include<sys/socket.h>
#include<netdb.h>
#include<arpa/inet.h>
#include <fcntl.h>
#else
#include <win64.h>
#endif

#ifdef __APPLE__
#include "../macosx/cpu_affinity.h"
#endif

using namespace std;

#define LOG(msg) cout<<msg<<endl<<flush

vector<string> getFiles(const string &folder);

#endif //ARGON2_COMMON_H
