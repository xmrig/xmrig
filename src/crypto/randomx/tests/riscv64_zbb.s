/* RISC-V - test if the Zbb extension is present */

.text
.global main

main:
    ror x6, x6, x7
    li x10, 0
    ret
