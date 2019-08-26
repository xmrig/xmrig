//
// Created by Haifa Bogdan Adnan on 17/08/2018.
//

#ifndef ARGON2_RANDOM_GENERATOR_H
#define ARGON2_RANDOM_GENERATOR_H

class DLLEXPORT random_generator {
public:
    random_generator();
    static random_generator &instance();

    void get_random_data(unsigned char *buffer, int length);

private:
    random_device __randomDevice;
    mt19937 __mt19937Gen;
    uniform_int_distribution<> __mt19937Distr;
    mutex __thread_lock;

    static random_generator __instance;
};

#endif //ARGON2_RANDOM_GENERATOR_H
