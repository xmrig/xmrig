//
// Created by Haifa Bogdan Adnan on 17/08/2018.
//

#include "crypto/argon2_hasher/common/DLLExport.h"
#include "../common/common.h"
#include "base64.h"

static const string base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";

static inline bool is_base64(unsigned char c) {
        return (isalnum(c) || (c == '+') || (c == '/'));
}

void base64::encode(const char *input, int input_size, char *output) {
        char *ret = output;
        int i = 0;
        int j = 0;
        unsigned char char_array_3[3];
        unsigned char char_array_4[4];

        while (input_size--) {
                char_array_3[i++] = *(input++);
                if (i == 3) {
                        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
                        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
                        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
                        char_array_4[3] = char_array_3[2] & 0x3f;

                        for(i = 0; (i <4) ; i++)
                                *(ret++) = base64_chars[char_array_4[i]];
                        i = 0;
                }
        }

        if (i)
        {
                for(j = i; j < 3; j++)
                        char_array_3[j] = '\0';

                char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
                char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
                char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
                char_array_4[3] = char_array_3[2] & 0x3f;

                for (j = 0; (j < i + 1); j++)
                        *(ret++) = base64_chars[char_array_4[j]];

                while((i++ < 3))
                        *(ret++) = '=';

        }
}

int base64::decode(const char *input, char *output, int output_size) {
        size_t in_len = strlen(input);
        int i = 0;
        int j = 0;
        int in_ = 0;
        unsigned char char_array_4[4], char_array_3[3];
        char *ret = output;
        int out_size = 0;

        while (in_len-- && ( input[in_] != '=') && is_base64(input[in_])) {
                char_array_4[i++] = input[in_]; in_++;
                if (i ==4) {
                        for (i = 0; i <4; i++)
                                char_array_4[i] = base64_chars.find(char_array_4[i]);

                        char_array_3[0] = ( char_array_4[0] << 2       ) + ((char_array_4[1] & 0x30) >> 4);
                        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
                        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];

                        for (i = 0; (i < 3); i++) {
                                out_size ++;
                                if(output_size < out_size)
                                        return -1;
                                *(ret++) = char_array_3[i];
                        }
                        i = 0;
                }
        }

        if (i) {
                for (j = 0; j < i; j++)
                        char_array_4[j] = base64_chars.find(char_array_4[j]);

                char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
                char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);

                for (j = 0; (j < i - 1); j++) {
                        out_size ++;
                        if(output_size < out_size)
                                return -1;
                        *(ret++) = char_array_3[j];
                }
        }
        return out_size;
}

