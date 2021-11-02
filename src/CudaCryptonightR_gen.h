#ifndef XMRIG_CUDACRYPTONIGHTR_GEN_H
#define XMRIG_CUDACRYPTONIGHTR_GEN_H

#include <cstdint>
#include <vector>
#include <string>

void CryptonightR_get_program(std::vector<char>& ptx, std::string& lowered_name, uint64_t height, int arch_major, int arch_minor, bool background = false);

#endif // XMRIG_CUDACRYPTONIGHTR_GEN_H
