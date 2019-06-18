# RandomX
RandomX is a proof-of-work (PoW) algorithm that is optimized for general-purpose CPUs. RandomX uses random code execution (hence the name) together with several memory-hard techniques to minimize the efficiency advantage of specialized hardware.

## Overview

RandomX utilizes a virtual machine that executes programs in a special instruction set that consists of integer math, floating point math and branches. These programs can be translated into the CPU's native machine code on the fly (example: [program.asm](doc/program.asm)). At the end, the outputs of the executed programs are consolidated into a 256-bit result using a cryptographic hashing function ([Blake2b](https://blake2.net/)).

RandomX can operate in two main modes with different memory requirements:

* **Fast mode** - requires 2080 MiB of shared memory.
* **Light mode** - requires only 256 MiB of shared memory, but runs significantly slower

Both modes are interchangeable as they give the same results. The fast mode is suitable for "mining", while the light mode is expected to be used only for proof verification.

## Documentation

Full specification is available in [specs.md](doc/specs.md).

Design description and analysis is available in [design.md](doc/design.md).

## Build

RandomX is written in C++11 and builds a static library with a C API provided by header file [randomx.h](src/randomx.h). Minimal API usage example is provided in [api-example1.c](src/tests/api-example1.c). The reference code includes a `benchmark` executable for testing.

### Linux

Build dependencies: `make` and `gcc` (minimum version 4.8, but version 7+ is recommended).

Build using the provided makefile.

### Windows

Build dependencies: Visual Studio 2017.

A solution file is provided.

### Precompiled binaries

Precompiled `benchmark` binaries are available on the [Releases page](https://github.com/tevador/RandomX/releases).

## Proof of work

RandomX was primarily designed as a PoW algorithm for [Monero](https://www.getmonero.org/). The recommended usage is following:

* The key `K` is selected to be the hash of a block in the blockchain - this block is called the 'key block'. For optimal mining and verification performance, the key should change every 2048 blocks (~2.8 days) and there should be a delay of 64 blocks (~2 hours) between the key block and the change of the key `K`. This can be achieved by changing the key when `blockHeight % 2048 == 64` and selecting key block such that `keyBlockHeight % 2048 == 0`.
* The input `H` is the standard hashing blob with a selected nonce value.

If you wish to use RandomX as a PoW algorithm for your cryptocurrency, please follow the [configuration guidelines](doc/configuration.md).

### CPU performance
Preliminary performance of selected CPUs using the optimal number of threads (T) and large pages (if possible), in hashes per second (H/s):

|CPU|RAM|OS|AES|Fast mode|Light mode|
|---|---|--|---|---------|--------------|
AMD Ryzen 7 1700|16 GB DDR4|Ubuntu 16.04|hardware|4100 H/s (8T)|620 H/s (16T)|
Intel Core i7-8550U|16 GB DDR4|Windows 10|hardware|1700 H/s (4T)|350 H/s (8T)|
Intel Core i3-3220|4 GB DDR3|Ubuntu 16.04|software|510 H/s (4T)|150 H/s (4T)|
Raspberry Pi 3|1 GB DDR2|Ubuntu 16.04|software|-|2.0 H/s (4T) †|

† Using the interpreter mode. Compiled mode is expected to increase performance by a factor of 10.

### GPU performance

SChernykh is developing GPU mining code for RandomX. Benchmarks are included in the following repositories:

* [CUDA miner](https://github.com/SChernykh/RandomX_CUDA) - NVIDIA GPUs.
* [OpenCL miner](https://github.com/SChernykh/RandomX_OpenCL) - currently only for AMD Vega (uses GCN5 machine code).

Note that GPUs are at a disadvantage when running RandomX since the algorithm was designed to be efficient on CPUs.

# FAQ

### Which CPU is best for mining RandomX?

Most Intel and AMD CPUs made since 2011 should be fairly efficient at RandomX. More specifically, efficient mining requires:

* 64-bit architecture
* IEEE 754 compliant floating point unit
* Hardware AES support ([AES-NI](https://en.wikipedia.org/wiki/AES_instruction_set) extension for x86, Cryptography extensions for ARMv8)
* 16 KiB of L1 cache, 256 KiB of L2 cache and 2 MiB of L3 cache per mining thread
* Support for large memory pages
* At least 2.5 GiB of free RAM per NUMA node
* Multiple memory channels may be required:
    * DDR3 memory is limited to about 1500 H/s per channel
    * DDR4 memory is limited to about 4000 H/s per channel



### Does RandomX facilitate botnets/malware mining or web mining?
Efficient mining requires more than 2 GiB of memory, which is difficult to hide in an infected computer and disqualifies many low-end machines such as IoT devices. Web mining is infeasible due to the large memory requirement and the lack of directed rounding support for floating point operations in both Javascript and WebAssembly.

### Since RandomX uses floating point math, does it give reproducible results on different platforms?

RandomX uses only operations that are guaranteed to give correctly rounded results by the [IEEE 754](https://en.wikipedia.org/wiki/IEEE_754) standard: addition, subtraction, multiplication, division and square root. Special care is taken to avoid corner cases such as NaN values or denormals.

The reference implementation has been validated on the following platforms:
* x86 (32-bit, little-endian)
* x86-64 (64-bit, little-endian)
* ARMv7+VFPv3 (32-bit, little-endian)
* ARMv8 (64-bit, little-endian)
* PPC64 (64-bit, big-endian)

## Acknowledgements
* [SChernykh](https://github.com/SChernykh) - contributed significantly to the design of RandomX
* [hyc](https://github.com/hyc) - original idea of using random code execution for PoW
* [nioroso-x3](https://github.com/nioroso-x3) - provided access to PowerPC for testing purposes

RandomX uses some source code from the following 3rd party repositories:
* Argon2d, Blake2b hashing functions: https://github.com/P-H-C/phc-winner-argon2

The author of RandomX declares no competing financial interest in RandomX adoption, other than being a holder of Monero. The development of RandomX was funded from the author's own pocket with only the help listed above.

## Donations

If you'd like to use RandomX, please consider donating to help cover the development cost of the algorithm.

Author's XMR address:
```
845xHUh5GvfHwc2R8DVJCE7BT2sd4YEcmjG8GNSdmeNsP5DTEjXd1CNgxTcjHjiFuthRHAoVEJjM7GyKzQKLJtbd56xbh7V
```
