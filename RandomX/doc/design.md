# RandomX design
To minimize the performance advantage of specialized hardware, a proof of work (PoW) algorithm must achieve *device binding* by targeting specific features of existing general-purpose hardware. This is a complex task because we have to target a large class of devices with different architectures from different manufacturers.

There are two distinct classes of general processing devices: central processing units (CPUs) and graphics processing units (GPUs). RandomX targets CPUs for the following reasons:

* CPUs, being less specialized devices, are more prevalent and widely accessible. A CPU-bound algorithm is more egalitarian and allows more participants to join the network. This is one of the goals stated in the original CryptoNote whitepaper [[1](https://cryptonote.org/whitepaper.pdf)]. 
* A large common subset of native hardware instructions exists among different CPU architectures. The same cannot be said about GPUs. For example, there is no common integer multiplication instruction for NVIDIA and AMD GPUs [[2](https://github.com/ifdefelse/ProgPOW/issues/16)].
* All major CPU instruction sets are well documented with multiple open source compilers available. In comparison, GPU instruction sets are usually proprietary and may require vendor specific closed-source drivers for maximum performance.

## 1. Design considerations

The most basic idea of a CPU-bound proof of work is that the "work" must be dynamic. This takes advantage of the fact that CPUs accept two kinds of inputs: *data* (the main input) and *code* (which specifies what to perform with the data).

Conversely, typical cryptographic hashing functions [[3](https://en.wikipedia.org/wiki/Cryptographic_hash_function)] do not represent suitable work for the CPU because their only input is *data*, while the sequence of operations is fixed and can be performed more efficiently by a specialized integrated circuit.

### 1.1 Dynamic proof of work

A dynamic proof of work algorithm can generally consist of the following 4 steps:

1) Generate a random program.
2) Translate it into the native machine code of the CPU.
3) Execute the program.
4) Transform the output of the program into a cryptographically secure value.

The actual 'useful' CPU-bound work is performed in step 3, so the algorithm must be tuned to minimize the overhead of the remaining steps.

#### 1.1.1 Generating a random program

Early attempts at a dynamic proof of work design were based on generating a program in a high-level language, such as C or Javascript [[4](https://github.com/hyc/randprog), [5](https://github.com/tevador/RandomJS)]. However, this is very inefficient for two main reasons:

* High level languages have a complex syntax, so generating a valid program is relatively slow since it requires the creation of an abstract syntax tree (ASL).
* Once the source code of the program is generated, the compiler will generally parse the textual representation back into the ASL, which makes the whole process of generating source code redundant.

The fastest way to generate a random program is to use a *logic-less* generator - simply filling a buffer with random data. This of course requires designing a syntaxless programming language (or instruction set) in which all random bit strings represent valid programs.

#### 1.1.2 Translating the program into machine code

This step is inevitable because we don't want to limit the algorithm to a specific CPU architecture. In order to generate machine code as fast as possible, we need our instruction set to be as close to native hardware as possible, while still generic enough to support different architectures. There is not enough time for expensive optimizations during code compilation.

#### 1.1.3 Executing the program

The actual program execution should utilize as many CPU components as possible. Some of the features that should be utilized in the program are:

* multi-level caches (L1, L2, L3)
* μop cache [[6](https://en.wikipedia.org/wiki/CPU_cache#Micro-operation_(%CE%BCop_or_uop)_cache)]
* arithmetic logic unit (ALU)
* floating point unit (FPU)
* memory controller
* instruction level parallelism [[7](https://en.wikipedia.org/wiki/Instruction-level_parallelism)]
    * superscalar execution [[8](https://en.wikipedia.org/wiki/Superscalar_processor)]
    * out-of-order execution [[9](https://en.wikipedia.org/wiki/Out-of-order_execution)]
    * speculative execution [[10](https://en.wikipedia.org/wiki/Speculative_execution)]
    * register renaming [[11](https://en.wikipedia.org/wiki/Register_renaming)]

Chapter 2 describes how the RandomX VM takes advantages of these features.

#### 1.1.4 Calculating the final result

Blake2b [[12](https://blake2.net/)] is a cryptographically secure hashing function that was specifically designed to be fast in software, especially on modern 64-bit processors, where it's around three times faster than SHA-3 and can run at a speed of around 3 clock cycles per byte of input. This function is an ideal candidate to be used in a CPU-friendly proof of work.

For processing larger amounts of data in a cryptographically secure way, the Advanced Encryption Standard (AES) [[13](https://en.wikipedia.org/wiki/Advanced_Encryption_Standard)] can provide the fastest processing speed because many modern CPUs support hardware acceleration of these operations. See chapter 3 for more details about the use of AES in RandomX.

### 1.2 The "Easy program problem"

When a random program is generated, one may choose to execute it only when it's favorable. This strategy is viable for two main reasons:

1. The runtime of randomly generated programs typically follows a log-normal distribution [[14](https://en.wikipedia.org/wiki/Log-normal_distribution)] (also see Appendix C). A generated program may be quickly analyzed and if it's likely to have above-average runtime, program execution may be skipped and a new program may be generated instead. This can significantly boost performance especially in case the runtime distribution has a heavy tail (many long-running outliers) and if program generation is cheap.
2. An implementation may choose to optimize for a subset of the features required for program execution. For example, the support for some operations (such as division) may be dropped or some instruction sequences may be implemented more efficiently. Generated programs would then be analyzed and be executed only if they match the specific requirements of the optimized implementation.

These strategies of searching for programs of particular properties deviate from the objectives of this proof of work, so they must be eliminated. This can be achieved by requiring a sequence of *N* random programs to be executed such that each program is generated from the output of the previous one. The output of the final program is then used as the result.

```
          +---------------+     +---------------+               +---------------+     +---------------+
          |               |     |               |               |               |     |               |
input --> |   program 1   | --> |   program 2   | -->  ...  --> | program (N-1) | --> |   program N   | --> result
          |               |     |               |               |               |     |               |
          +---------------+     +---------------+               +---------------+     +---------------+
```

The principle is that after the first program is executed, a miner has to either commit to finishing the whole chain (which may include unfavorable programs) or start over and waste the effort expended on the unfinished chain. Examples of how this affects the hashrate of different mining strategies are given in Appendix A.

Additionally, this chained program execution has the benefit of equalizing the runtime for the whole chain since the relative deviation of a sum of identically distributed runtimes is decreased.

### 1.3 Verification time

Since the purpose of the proof of work is to be used in a trustless peer-to-peer network, network participants must be able to quickly verify if a proof is valid or not. This puts an upper bound on the complexity of the proof of work algorithm. In particular, we set a goal for RandomX to be at least as fast to verify as the CryptoNight hash function [[15](https://cryptonote.org/cns/cns008.txt)], which it aims to replace.

### 1.4 Memory-hardness

Besides pure computational resources, such as ALUs and FPUs, CPUs usually have access to a large amount of memory in the form of DRAM [[16](https://en.wikipedia.org/wiki/Dynamic_random-access_memory)]. The performance of the memory subsystem is typically tuned to match the compute capabilities, for example [[17](https://en.wikipedia.org/wiki/Multi-channel_memory_architecture)]:

* single channel memory for embedded and low power CPUs
* dual channel memory for desktop CPUs 
* triple or quad channel memory for workstation CPUs
* six or eight channel memory for high-end server CPUs

In order to utilize the external memory as well as the on-chip memory controllers, the proof of work algorithm should access a large memory buffer (called the "Dataset"). The Dataset must be:

1. larger than what can be stored on-chip (to require external memory)
2. dynamic (to require writable memory)

The maximum amount of SRAM that can be put on a single chip is more than 512 MiB for a 16 nm process and more than 2 GiB for a 7 nm process [[18](https://www.grin-forum.org/t/obelisk-grn1-chip-details/4571)]. Ideally, the size of the Dataset should be at least 4 GiB. However, due to constraints on the verification time (see below), the size used by RandomX was selected to be 2080 MiB. While a single chip can theoretically be made with this amount of SRAM using current technology (7 nm in 2019), the feasibility of such solution is questionable, at least in the near future.

#### 1.4.1 Light-client verification

While it's reasonable to require >2 GiB for dedicated mining systems that solve the proof of work, an option must be provided for light clients to verify the proof using a much lower amount of memory.

The ratio of memory required for the 'fast' and 'light' modes must be chosen carefully not to make the light mode viable for mining. In particular, the area-time (AT) product of the light mode should not be smaller than the AT product of the fast mode. Reduction of the AT product is a common way of measuring tradeoff attacks [[19](https://eprint.iacr.org/2015/227.pdf)].

Given the constraints described in the previous chapters, the maximum possible performance ratio between the fast and the light verification modes was empirically determined to be 8. This is because:

1. Further increase of the light verification time would violate the constraints set out in chapter 1.3.
2. Further decrease of the fast mode runtime would violate the constraints set out in chapter 1.1, in particular the overhead time of program generation and result calculation would become too high.

Additionally, 256 MiB was selected as the maximum amount of memory that can be required in the light-client mode. This amount is acceptable even for small single-board computers such as the Raspberry Pi.

To keep a constant memory-time product, the maximum fast-mode memory requirement is:
```
8 * 256 MiB = 2048 MiB
```
This can be further increased since the light mode requires additional chip area for the SuperscalarHash function (see chapter 3.4 and chapter 6 of the Specification). Assuming a conservative estimate of 0.2 mm<sup>2</sup> per SuperscalarHash core and DRAM density of 0.149 Gb/mm<sup>2</sup> [[20](http://en.thelec.kr/news/articleView.html?idxno=20)], the additional memory is:

```
8 * 0.2 * 0.149 * 1024 / 8 = 30.5 MiB
```
or 32 MiB when rounded to the nearest power of 2. The total memory requirement of the fast mode can be 2080 MiB with a roughly constant AT product.

## 2. Virtual machine architecture

This section describes the design of the RandomX virtual machine (VM).

### 2.1 Instruction set

RandomX uses a fixed-length instruction encoding with 8 bytes per instruction. This allows a 32-bit immediate value to be included in the instruction word. The interpretation of the instruction word bits was chosen so that any 8-byte word is a valid instruction. This allows for very efficient random program generation (see chapter 1.1.1).

### 2.2 Program

The program executed by the VM has the form of a loop consisting of 256 random instructions.

* 256 instructions is long enough to provide a large number of possible programs and enough space for branches. The number of different programs that can be generated is limited to 2<sup>512</sup> = 1.3e+154, which is the number of possible seed values of the random generator.
* 256 instructions is short enough so that high-performance CPUs can execute one iteration in similar time it takes to fetch data from DRAM. This is advantageous because it allows Dataset accesses to be synchronized and fully prefetchable (see chapter 2.9).
* Since the program is a loop, it can take advantage of the μop cache [[6](https://en.wikipedia.org/wiki/CPU_cache#Micro-operation_(%CE%BCop_or_uop)_cache)] that is present in some x86 CPUs. Running a loop from the μop cache allows the CPU to power down the x86 instruction decoders, which should help to equalize the power efficiency between x86 and architectures with simple instruction decoding.

### 2.3 Registers

The VM uses 8 integer registers and 12 floating point registers. This is the maximum that can be allocated as physical registers in x86-64, which has the fewest architectural registers among common 64-bit CPU architectures. Using more registers would put x86 CPUs at a disadvantage since they would have to use memory to store VM register contents.

### 2.4 Integer operations

RandomX uses all primitive integer operations that have high output entropy: addition (IADD_RS, IADD_M), subtraction (ISUB_R, ISUB_M, INEG_R), multiplication (IMUL_R, IMUL_M, IMULH_R, IMULH_M, ISMULH_R, ISMULH_M, IMUL_RCP), exclusive or (IXOR_R, IXOR_M) and rotation (IROR_R, IROL_R).

#### 2.4.1 IADD_RS

The IADD_RS instruction utilizes the address calculation logic of CPUs and can be performed in a single hardware instruction by most CPUs (x86 `lea`, arm `add`).

#### 2.4.2 IMUL_RCP

Because integer division is not fully pipelined in CPUs and can be made faster in ASICs, the IMUL_RCP instruction requires only one division per program to calculate the reciprocal. This forces an ASIC to include a hardware divider without giving them a performance advantage during program execution.

#### 2.4.3 ISWAP_R

This instruction can be executed efficiently by CPUs that support register renaming/move elimination.

### 2.5 Floating point operations

RandomX uses double precision floating point operations, which are supported by the majority of CPUs and require more complex hardware than single precision. All operations are performed as 128-bit vector operations, which is also supported by all major CPU architectures.

RandomX uses five operations that are guaranteed by the IEEE 754 standard to give correctly rounded results: addition, subtraction, multiplication, division and square root. All 4 rounding modes defined by the standard are used.

#### 2.5.1 Floating point register groups

The domains of floating point operations are separated into "additive" operations, which use register group F and "multiplicative" operations, which use register group E. This is done to prevent addition/subtraction from becoming no-op when a small number is added to a large number. Since the range of the F group registers is limited to around `±3.0e+14`, adding or subtracting a floating point number with absolute value larger than 1 always changes at least 5 fraction bits.

Because the limited range of group F registers would allow the use of a more efficient fixed-point representation (with 80-bit numbers), the FSCAL instruction manipulates the binary representation of the floating point format to make this optimization more difficult.

Group E registers are restricted to positive values, which avoids `NaN` results (such as square root of a negative number or `0 * ∞`). Division uses only memory source operand to avoid being optimized into multiplication by constant reciprocal. The exponent of group E memory operands is set to a value between -255 and 0 to avoid division and multiplication by 0 and to increase the range of numbers that can be obtained. The approximate range of possible group E register values is `1.7E-77` to `infinity`.

Approximate distribution of floating point register values at the end of each program loop is shown in these figures (left - group F, right - group E):

![Imgur](https://i.imgur.com/64G4qE8.png)

*(Note: bins are marked by the left-side value of the interval, e.g. bin marked `1e-40` contains values from `1e-40` to `1e-20`.)*

The small number of F register values at `1e+14` is caused by the FSCAL instruction, which significantly increases the range of the register values.

Group E registers cover a very large range of values. About 2% of programs produce at least one `infinity` value.

To maximize entropy and also to fit into one 64-byte cache line, floating point registers are combined using the XOR operation at the end of each iteration before being stored into the Scratchpad.

### 2.6 Branches

Modern CPUs invest a lot of die area and energy to handle branches. This includes:

* Branch predictor unit [[21](https://en.wikipedia.org/wiki/Branch_predictor)]
* Checkpoint/rollback states that allow the CPU to recover in case of a branch misprediction.

To take advantage of speculative designs, the random programs should contain branches. However, if branch prediction fails, the speculatively executed instructions are thrown away, which results in a certain amount of wasted energy with each misprediction. Therefore we should aim to minimize the number of mispredictions.

Additionally, branches in the code are essential because they significantly reduce the amount of static optimizations that can be made. For example, consider the following x86 instruction sequence:
```asm
    ...
branch_target_00:
    ...
    xor r8, r9
    test r10, 2088960
    je branch_target_00
    xor r8, r9
    ...
```
The XOR operations would normally cancel out, but cannot be optimized away due to the branch because the result will be different if the branch is taken. Similarly, the ISWAP_R instruction could be always statically optimized out if it wasn't for branches.

In general, random branches must be designed in such way that:

1. Infinite loops are not possible.
1. The number of mispredicted branches is small.
1. Branch condition depends on a runtime value to disable static branch optimizations.

#### 2.6.1 Branch prediction

Unfortunately, we haven't found a way how to utilize branch prediction in RandomX. Because RandomX is a consensus protocol, all the rules must be set out in advance, which includes the rules for branches. Fully predictable branches cannot depend on the runtime value of any VM register (since register values are pseudorandom and unpredictable), so they would have to be static and therefore easily optimizable by specialized hardware.

#### 2.6.2 CBRANCH instruction

RandomX therefore uses random branches with a jump probability of 1/256 and branch condition that depends on an integer register value. These branches will be predicted as "not taken" by the CPU. Such branches are "free" in most CPU designs unless they are taken. While this doesn't take advantage of the branch predictors, speculative designs will see a significant performance boost compared to non-speculative branch handling - see Appendix B for more information.

The branching conditions and jump targets are chosen in such way that infinite loops in RandomX code are impossible because the register controlling the branch will never be modified in the repeated code block. Each CBRANCH instruction can jump up to twice in a row. Handling CBRANCH using predicated execution [[22](https://en.wikipedia.org/wiki/Predication_(computer_architecture))] is impractical because the branch is not taken most of the time.

### 2.7 Instruction-level parallelism

CPUs improve their performance using several techniques that utilize instruction-level parallelism of the executed code. These techniques include:

* Having multiple execution units that can execute operations in parallel (*superscalar execution*).
* Executing instruction not in program order, but in the order of operand availability (*out-of-order execution*).
* Predicting which way branches will go to enhance the benefits of both superscalar and out-of-order execution.

RandomX benefits from all these optimizations. See Appendix B for a detailed analysis.

### 2.8 Scratchpad

The Scratchpad is used as read-write memory. Its size was selected to fit entirely into CPU cache.

#### 2.8.1 Scratchpad levels

The Scratchpad is split into 3 levels to mimic the typical CPU cache hierarchy [[23](https://en.wikipedia.org/wiki/CPU_cache)]. Most VM instructions access "L1" and "L2" Scratchpad because L1 and L2 CPU caches are located close to the CPU execution units and provide the best random access latency. The ratio of reads from L1 and L2 is 3:1, which matches the inverse ratio of typical latencies (see table below).

|CPU μ-architecture|L1 latency|L2 latency|L3 latency|source|
|----------------|----------|----------|----------|------|
ARM Cortex A55|2|6|-|[[24](https://www.anandtech.com/show/11441/dynamiq-and-arms-new-cpus-cortex-a75-a55/4)]
|AMD Zen+|4|12|40|[[25](https://en.wikichip.org/wiki/amd/microarchitectures/zen%2B#Memory_Hierarchy)]|
|Intel Skylake|4|12|42|[[26](https://en.wikichip.org/wiki/amd/microarchitectures/zen%2B#Memory_Hierarchy)]

The L3 cache is much larger and located further from the CPU core. As a result, its access latencies are much higher and can cause stalls in program execution.

RandomX therefore performs only 2 random accesses into "L3" Scratchpad per program iteration (steps 2 and 3 in chapter 4.6.2 of the Specification). Register values from a given iteration are written into the same locations they were loaded from, which guarantees that the required cache lines have been moved into the faster L1 or L2 caches.

Additionally, integer instructions that read from a fixed address also use the whole "L3" Scratchpad (Table 5.1.4 of the Specification) because repetitive accesses will ensure that the cache line will be placed in the L1 cache of the CPU. This shows that the Scratchpad level doesn't always directly correspond to the same CPU cache level.

#### 2.8.2 Scratchpad writes

There are two ways the Scratchpad is modified during VM execution:

1. At the end of each program iteration, all register values are written into "L3" Scratchpad (see Specification chapter 4.6.2, steps 9 and 11). This writes a total of 128 bytes per iteration in two 64-byte blocks.
2. The ISTORE instruction does explicit stores. On average, there are 16 stores per program, out of which 2 stores are into the "L3" level. Each ISTORE instruction writes 8 bytes.

The image below shows an example of the distribution of writes to the Scratchpad. Each pixel in the image represents 8 bytes of the Scratchpad. Red pixels represent portions of the Scratchpad that have been overwritten at least once during hash calculation. The "L1" and "L2" levels are on the left side (almost completely overwritten). The right side of the scratchpad represents the bottom 1792 KiB. Only about 66% of it are overwritten, but the writes are spread uniformly and randomly.

![Imgur](https://i.imgur.com/pRz6aBG.png)

See Appendix D for the analysis of Scratchpad entropy.

#### 2.8.3 Read-write ratio

Programs make, on average, 39 reads (instructions IADD_M, ISUB_M, IMUL_M, IMULH_M, ISMULH_M, IXOR_M, FADD_M, FSUB_M, FDIV_M) and 16 writes (instruction ISTORE) to the Scratchpad per program iteration. Additional 128 bytes are read and written implicitly to initialize and store register values. 64 bytes of data is read from the Dataset per iteration. In total:

* The average amount of data read from memory per program iteration is: 39 * 8 + 128 + 64 = **504 bytes**.
* The average mount of data written to memory per program iteration is: 16 * 8 + 128 = **256 bytes**.

This is close to a 2:1 read/write ratio, which CPUs are optimized for.

### 2.9 Dataset

Since the Scratchpad is usually stored in the CPU cache, only Dataset accesses utilize the memory controllers.

RandomX randomly reads from the Dataset once per program iteration (16384 times per hash result). Since the Dataset must be stored in DRAM, it provides a natural parallelization limit, because DRAM cannot do more than about 25 million random accesses per second per bank group. Each separately addressable bank group allows a throughput of around 1500 H/s.

All Dataset accesses read one CPU cache line (64 bytes) and are fully prefetched. The time to execute one program iteration described in chapter 4.6.2 of the Specification is about the same as typical DRAM access latency (50-100 ns).

#### 2.9.1 Cache

The Cache, which is used for light verification and Dataset construction, is about 8 times smaller than the Dataset. To keep a constant area-time product, each Dataset item is constructed from 8 random Cache accesses.

Because 256 MiB is small enough to be included on-chip, RandomX uses a custom high-latency, high-power mixing function ("SuperscalarHash") which defeats the benefits of using low-latency memory and the energy required to calculate SuperscalarHash makes light mode very inefficient for mining (see chapter 3.4).

Using less than 256 MiB of memory is not possible due to the use of tradeoff-resistant Argon2d with 3 iterations. When using 3 iterations (passes), halving the memory usage increases computational cost 3423 times for the best tradeoff attack [[27](https://eprint.iacr.org/2015/430.pdf)].

## 3. Custom functions

### 3.1 AesGenerator1R

AesGenerator1R was designed for the fastest possible generation of pseudorandom data to fill the Scratchpad. It takes advantage of hardware accelerated AES in modern CPUs. Only one AES round is performed per 16 bytes of output, which results in throughput exceeding 20 GB/s in most modern CPUs. While 1 AES round is not sufficient for a good distribution of random values, this is not an issue because the purpose is just to initialize the Scratchpad with random non-zero data.

### 3.2 AesGenerator4R

AesGenerator4R uses 4 AES rounds to generate pseudorandom data for Program Buffer initialization. Since 2 AES rounds are sufficient for full avalanche of all input bits [[28](https://csrc.nist.gov/csrc/media/projects/cryptographic-standards-and-guidelines/documents/aes-development/rijndael-ammended.pdf)], AesGenerator4R provides an excellent output distribution while maintaining very good performance.

The reversible nature of this generator is not an issue since the generator state is always initialized using the output of a non-reversible hashing function (Blake2b).

### 3.3 AesHash1R

AesHash was designed for the fastest possible calculation of the Scratchpad fingerprint. It interprets the Scratchpad as a set of AES round keys, so it's equivalent to AES encryption with 32768 rounds. Two extra rounds are performed at the end to ensure avalanche of all Scratchpad bits in each lane. The output of the AesHash is fed into the Blake2b hashing function to calculate the final PoW hash.

### 3.4 SuperscalarHash

SuperscalarHash was designed to burn as much power as possible while the CPU is waiting for data to be loaded from DRAM. The target latency of 170 cycles corresponds to the usual DRAM latency of 40-80 ns and clock frequency of 2-4 GHz. ASIC devices designed for light-mode mining with low-latency memory will be bottlenecked by SuperscalarHash when calculating Dataset items and their efficiency will be destroyed by the high power usage of SuperscalarHash.

The average SuperscalarHash function contains a total of 450 instructions, out of which 155 are 64-bit multiplications. On average, the longest dependency chain is 95 instructions long. An ASIC design for light-mode mining, with 256 MiB of on-die memory and 1-cycle latency for all operations, will need on average 95 * 8 = 760 cycles to construct a Dataset item, assuming unlimited parallelization. It will have to execute 155 * 8 = 1240 64-bit multiplications per item, which will consume energy comparable to loading 64 bytes from DRAM.

## Appendix

### A. The effect of chaining VM executions

Chapter 1.2 describes why `N` random programs are chained to prevent mining strategies that search for 'easy' programs. RandomX uses a value of `N = 8`.

Let's define `Q` as the ratio of acceptable programs in a strategy that uses filtering. For example `Q = 0.75` means that 25% of programs are rejected. 

For `N = 1`, there are no wasted program executions and the only cost is program generation and the filtering itself. The calculations below assume that these costs are zero and the only real cost is program execution. However, this is a simplification because program generation in RandomX is not free (the first program generation requires full Scratchpad initialization), but it describes a best-case scenario for an attacker.


 For `N > 1`, the first program can be filtered as usual, but after the program is executed, there is a chance of `1-Q` that the next program should be rejected and we have wasted one program execution.

For `N` chained executions, the chance is only <code>Q<sup>N</sup></code> that all programs in the chain are acceptable. However, during each attempt to find such chain, we will waste the execution of some programs. For `N = 8`, the number of wasted programs per attempt is equal to <code>(1-Q)*(1+2\*Q+3\*Q<sup>2</sup>+4\*Q<sup>3</sup>+5\*Q<sup>4</sup>+6\*Q<sup>5</sup>+7\*Q<sup>6</sup>)</code> (approximately 2.5 for `Q = 0.75`).

Let's consider 3 mining strategies:

#### Strategy I

Honest miner that doesn't reject any programs (`Q = 1`).

#### Strategy II

Miner that uses optimized custom hardware that cannot execute 25% of programs (`Q = 0.75`), but supported programs can be executed 50% faster.

#### Strategy III

Miner that can execute all programs, but rejects 25% of the slowest programs for the first program in the chain. This gives a 5% performance boost for the first program in the chain (this matches the runtime distribution from Appendix C).

#### Results

The table below lists the results for the above 3 strategies and different values of `N`. The columns **N(I)**, **N(II)** and **N(III)** list the number of programs that each strategy has to execute on average to get one valid hash result (this includes programs wasted in rejected chains). Columns **Speed(I)**, **Speed(II)** and **Speed(III)** list the average mining performance relative to strategy I.

|N|N(I)|N(II)|N(III)|Speed(I)|Speed(II)|Speed(III)|
|---|----|----|----|---------|---------|---------|
|1|1|1|1|1.00|1.50|1.05|
|2|2|2.3|2|1.00|1.28|1.02|
|4|4|6.5|4|1.00|0.92|1.01|
|8|8|27.0|8|1.00|0.44|1.00|

For `N = 8`, strategy II will perform at less than half the speed of the honest miner despite having a 50% performance advantage for selected programs. The small statistical advantage of strategy III is negligible with `N = 8`.

### B. Performance simulation

As discussed in chapter 2.7, RandomX aims to take advantage of the complex design of modern high-performance CPUs. To evaluate the impact of superscalar, out-of-order and speculative execution, we performed a simplified CPU simulation. Source code is available in [perf-simulation.cpp](../src/tests/perf-simulation.cpp).

#### CPU model

The model CPU uses a 3-stage pipeline to achieve an ideal throughput of 1 instruction per cycle:
```
        (1)                        (2)                     (3)
+------------------+       +----------------+      +----------------+
|   Instruction    |       |                |      |                |
|      fetch       | --->  | Memory access  | ---> |    Execute     |
|    + decode      |       |                |      |                |
+------------------+       +----------------+      +----------------+
```
The 3 stages are:

1. Instruction fetch and decode. This stage loads the instruction from the Program Buffer and decodes the instruction operation and operands.
2. Memory access. If this instruction uses a memory operand, it is loaded from the Scratchpad in this stage. This includes the calculation of the memory address. Stores are also performed in this stage. The value of the address register must be available in this stage.
3. Execute. This stage executes the instruction using the operands retrieved in the previous stages and writes the results into the register file.

Note that this is an optimistically short pipeline that would not allow very high clock speeds. Designs using a longer pipeline would significantly increase the benefits of speculative execution.

#### Superscalar execution

Our model CPU contains two kinds of components:

* Execution unit (EXU) - it is used to perform the actual integer or floating point operation. All RandomX instructions except ISTORE must use an execution unit in the 3rd pipeline stage. All operations are considered to take only 1 clock cycle.
* Memory unit (MEM) - it is used for loads and stores into Scratchpad. All memory instructions (including ISTORE) use a memory unit in the 2nd pipeline stage.

A superscalar design will contain multiple execution or memory units to improve performance.

#### Out-of-order execution

The simulation model supports two designs:

1. **In-order** - all instructions are executed in the order they appear in the Program Buffer. This design will stall if a dependency is encountered or the required EXU/MEM unit is not available.
2. **Out-of-order** - doesn't execute instructions in program order, but an instruction can be executed when its operands are ready and the required EXU/MEM units are available.

#### Branch handling

The simulation model supports two types of branch handling:

1. **Non-speculative** - when a branch is encountered, the pipeline is stalled. This typically adds a 3-cycle penalty for each branch.
2. **Speculative** - all branches are predicted not taken and the pipeline is flushed if a misprediction occurs (probability of 1/256).

#### Results

The following 10 designs were simulated and the average number of clock cycles to execute a RandomX program (256 instructions) was measured.

|design|superscalar config.|reordering|branch handling|execution time [cycles]|IPC|
|-------|-----------|----------|---------------|-----------------------|---|
|#1|1 EXU + 1 MEM|in-order|non-speculative|293|0.87|
|#2|1 EXU + 1 MEM|in-order|speculative|262|0.98|
|#3|1 EXU + 1 MEM|in-order|non-speculative|197|1.3|
|#4|2 EXU + 1 MEM|in-order|speculative|161|1.6|
|#5|2 EXU + 1 MEM|out-of-order|non-speculative|144|1.8|
|#6|2 EXU + 1 MEM|out-of-order|speculative|122|2.1|
|#7|4 EXU + 2 MEM|in-order|non-speculative|135|1.9|
|#8|4 EXU + 2 MEM|in-order|speculative|99|2.6|
|#9|4 EXU + 2 MEM|out-of-order|non-speculative|89|2.9|
|#10|4 EXU + 2 MEM|out-of-order|speculative|64|4.0|

The benefits of superscalar, out-of-order and speculative designs are clearly demonstrated.

### C. RandomX runtime distribution

Runtime numbers were measured on AMD Ryzen 7 1700 running at 3.0 GHz using 1 core. Source code to measure program execution and verification times is available in [runtime-distr.cpp](../src/tests/runtime-distr.cpp). Source code to measure the performance of the x86 JIT compiler is available in [jit-performance.cpp](../src/tests/jit-performance.cpp).

#### Fast mode - program execution

The following figure shows the distribution of the runtimes of a single VM program (in fast mode). This includes: program generation, JIT compilation, VM execution and Blake2b hash of the register file. Program generation and JIT compilation was measured to take 3.6 μs per program.

![Imgur](https://i.imgur.com/ikv2z2i.png)

AMD Ryzen 7 1700 can calculate 625 hashes per second in fast mode (using 1 thread), which means a single hash result takes 1600 μs (1.6 ms). This consists of (approximately):

* 1480 μs for VM execution (8 programs)
* 45 μs for initial Scratchpad fill (AesGenerator1R).
* 45 μs for final Scratchpad hash (AesHash1R).
* 30 μs for program generation and JIT compilation (8 programs)

This gives a total overhead of 7.5% (time per hash spent not executing VM).

#### Light mode - verification time

The following figure shows the distribution of times to calculate 1 hash result using the light mode. Most of the time is spent executing SuperscalarHash to calculate Dataset items (13.2 ms out of 14.8 ms). The average verification time exactly matches the performance of the CryptoNight algorithm.

![Imgur](https://i.imgur.com/VtwwJT8.png)

### D. Scratchpad entropy analysis

The average entropy of the Scratchpad after 8 program executions was approximated using the LZMA compression algorithm:

1. Hash resuls were calculated and the final scratchpads were written to disk as files with '.spad' extension (source code: [scratchpad-entropy.cpp](../src/tests/scratchpad-entropy.cpp))
2. The files were compressed using 7-Zip [[29](https://www.7-zip.org/)] in Ultra compression mode: `7z.exe a -t7z -m0=lzma2 -mx=9 scratchpads.7z *.spad`

The size of the resulting archive is approximately 99.98% of the uncompressed size of the scratchpad files. This shows that the Scratchpad retains high entropy during VM execution.

### E. SuperscalarHash analysis

SuperscalarHash is a custom function used by RandomX to generate Dataset items. It operates on 8 integer registers and uses a random sequence of instructions. About 1/3 of the instructions are multiplications.

The following figure shows the sensitivity of SuperscalarHash to changing a single bit of an input register:

![Imgur](https://i.imgur.com/ztZ0V0G.png)

This shows that SuperscalaHash has quite low sensitivity to high-order bits and somewhat decreased sensitivity to the lowest-order bits. Sensitivity is highest for bits 3-53 (inclusive).

When calculating a Dataset item, the input of the first SuperscalarHash depends only on the item number. To ensure a good distribution of results, the constants described in section 7.3 of the Specification were chosen to provide unique values of bits 3-53 for *all* item numbers in the range 0-34078718 (the Dataset contains 34078719 items). All initial register values for all Dataset item numbers were checked to make sure bits 3-53 of each register are unique and there are no collisions (source code: [superscalar-init.cpp](../src/tests/superscalar-init.cpp)). While this is not strictly necessary to get unique output from SuperscalarHash, it's a security precaution that mitigates the non-perfect avalanche properties of the randomly generated SuperscalarHash instances.

## References

[1] CryptoNote whitepaper - https://cryptonote.org/whitepaper.pdf

[2] ProgPoW: Inefficient integer multiplications - https://github.com/ifdefelse/ProgPOW/issues/16

[3] Cryptographic Hashing function - https://en.wikipedia.org/wiki/Cryptographic_hash_function

[4] randprog - https://github.com/hyc/randprog

[5] RandomJS - https://github.com/tevador/RandomJS

[6] μop cache - https://en.wikipedia.org/wiki/CPU_cache#Micro-operation_(%CE%BCop_or_uop)_cache

[7] Instruction-level parallelism - https://en.wikipedia.org/wiki/Instruction-level_parallelism

[8] Superscalar processor - https://en.wikipedia.org/wiki/Superscalar_processor

[9] Out-of-order execution - https://en.wikipedia.org/wiki/Out-of-order_execution

[10] Speculative execution - https://en.wikipedia.org/wiki/Speculative_execution

[11] Register renaming - https://en.wikipedia.org/wiki/Register_renaming

[12] Blake2 hashing function - https://blake2.net/

[13] Advanced Encryption Standard - https://en.wikipedia.org/wiki/Advanced_Encryption_Standard

[14] Log-normal distribution - https://en.wikipedia.org/wiki/Log-normal_distribution

[15] CryptoNight hash function - https://cryptonote.org/cns/cns008.txt

[16] Dynamic random-access memory - https://en.wikipedia.org/wiki/Dynamic_random-access_memory

[17] Multi-channel memory architecture - https://en.wikipedia.org/wiki/Multi-channel_memory_architecture

[18] Obelisk GRN1 chip details - https://www.grin-forum.org/t/obelisk-grn1-chip-details/4571

[19] Biryukov et al.: Tradeoff Cryptanalysis of Memory-Hard Functions - https://eprint.iacr.org/2015/227.pdf

[20] SK Hynix 20nm DRAM density - http://en.thelec.kr/news/articleView.html?idxno=20

[21] Branch predictor - https://en.wikipedia.org/wiki/Branch_predictor

[22] Predication - https://en.wikipedia.org/wiki/Predication_(computer_architecture)

[23] CPU cache - https://en.wikipedia.org/wiki/CPU_cache

[24] Cortex-A55 Microarchitecture - https://www.anandtech.com/show/11441/dynamiq-and-arms-new-cpus-cortex-a75-a55/4

[25] AMD Zen+ Microarchitecture - https://en.wikichip.org/wiki/amd/microarchitectures/zen%2B#Memory_Hierarchy

[26] Intel Skylake Microarchitecture - https://en.wikichip.org/wiki/amd/microarchitectures/zen%2B#Memory_Hierarchy

[27] Biryukov et al.: Fast and Tradeoff-Resilient Memory-Hard Functions for
Cryptocurrencies and Password Hashing - https://eprint.iacr.org/2015/430.pdf Table 2, page 8

[28] J. Daemen, V. Rijmen: AES Proposal: Rijndael - https://csrc.nist.gov/csrc/media/projects/cryptographic-standards-and-guidelines/documents/aes-development/rijndael-ammended.pdf page 28

[29] 7-Zip File archiver - https://www.7-zip.org/
