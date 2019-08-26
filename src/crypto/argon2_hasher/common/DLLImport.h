//
// Created by Haifa Bogdan Adnan on 04.11.2018.
//

#ifndef ARGON2_DLLIMPORT_H
#define ARGON2_DLLIMPORT_H

#ifndef DLLEXPORT
    #ifndef _WIN64
        #define DLLEXPORT
    #else
        #define DLLEXPORT __declspec(dllimport)
    #endif
#endif

#endif //ARGON2_DLLIMPORT_H
