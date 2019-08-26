//
// Created by Haifa Bogdan Adnan on 04.11.2018.
//

#ifndef ARGON2_DLLEXPORT_H
#define ARGON2_DLLEXPORT_H

#undef DLLEXPORT

#ifndef _WIN64
	#define DLLEXPORT
#else
	#define DLLEXPORT __declspec(dllexport)
#endif

#endif //ARGON2_DLLEXPORT_H
