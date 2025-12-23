/* RISC-V - test if the Zba extension is present */

.text
.global main

main:
    sh1add x6, x6, x7
    li x10, 0
    ret
