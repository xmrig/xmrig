//
// Created by Haifa Bogdan Adnan on 30/05/2019.
//

#ifndef ARGON2_HEX_H
#define ARGON2_HEX_H

class DLLEXPORT hex {
public:
    static void encode(const unsigned char *input, int input_size, char *output);
    static int decode(const char *input, unsigned char *output, int output_size);
};

#endif //ARGON2_HEX_H
