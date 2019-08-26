//
// Created by Haifa Bogdan Adnan on 30/05/2019.
//

#include "crypto/argon2_hasher/common/DLLExport.h"
#include "../common/common.h"
#include "hex.h"

void hex::encode(const unsigned char *input, int input_size, char *output) {
    for ( int i=0; i<input_size; i++ ) {
        char b1= *input >> 4;   // hi nybble
        char b2= *input & 0x0f; // lo nybble
        b1+='0'; if (b1>'9') b1 += 7;  // gap between '9' and 'A'
        b2+='0'; if (b2>'9') b2 += 7;
        *(output++)= b1;
        *(output++) = b2;
        input++;
    }
    *output = 0;
}

int hex::decode(const char *input, unsigned char *output, int output_size) {
    size_t in_len = strlen(input);
    for ( int i=0; i<in_len; i+=2 ) {
        unsigned char b1= input[i] -'0'; if (b1>9) b1 -= 7;
        unsigned char b2= input[i+1] -'0'; if (b2>9) b2 -= 7;
        *(output++) = (b1<<4) + b2;  // <<4 multiplies by 16
    }
    return in_len / 2;
}
