# RandomX configuration

RandomX has 45 customizable parameters (see table below). We recommend each project using RandomX to select a unique configuration to prevent network attacks from hashpower rental services.

These parameters can be modified in source file [configuration.h](../src/configuration.h).

|parameter|description|default value|
|---------|-----|-------|
|`RANDOMX_ARGON_MEMORY`|The number of 1 KiB Argon2 blocks in the Cache| `262144`|
|`RANDOMX_ARGON_ITERATIONS`|The number of Argon2d iterations for Cache initialization|`3`|
|`RANDOMX_ARGON_LANES`|The number of parallel lanes for Cache initialization|`1`|
|`RANDOMX_ARGON_SALT`|Argon2 salt|`"RandomX\x03"`|
|`RANDOMX_CACHE_ACCESSES`|The number of random Cache accesses per Dataset item|`8`|
|`RANDOMX_SUPERSCALAR_LATENCY`|Target latency for SuperscalarHash (in cycles of the reference CPU)|`170`|
|`RANDOMX_DATASET_BASE_SIZE`|Dataset base size in bytes|`2147483648`|
|`RANDOMX_DATASET_EXTRA_SIZE`|Dataset extra size in bytes|`33554368`|
|`RANDOMX_PROGRAM_SIZE`|The number of instructions in a RandomX program|`256`|
|`RANDOMX_PROGRAM_ITERATIONS`|The number of iterations per program|`2048`|
|`RANDOMX_PROGRAM_COUNT`|The number of programs per hash|`8`|
|`RANDOMX_JUMP_BITS`|Jump condition mask size in bits|`8`|
|`RANDOMX_JUMP_OFFSET`|Jump condition mask offset in bits|`8`|
|`RANDOMX_SCRATCHPAD_L3`|Scratchpad size in bytes|`2097152`|
|`RANDOMX_SCRATCHPAD_L2`|Scratchpad L2 size in bytes|`262144`|
|`RANDOMX_SCRATCHPAD_L1`|Scratchpad L1 size in bytes|`16384`|
|`RANDOMX_FREQ_*` (29x)|Instruction frequencies|multiple values|

Not all of the parameters can be changed safely and most parameters have some contraints on what values can be selected. Follow the guidelines below.

### RANDOMX_ARGON_MEMORY

This parameter determines the amount of memory needed in the light mode. Memory is specified in KiB (1 KiB = 1024 bytes).

#### Permitted values
Any integer power of 2.

#### Notes
Lower sizes will reduce the memory-hardness of the algorithm.

### RANDOMX_ARGON_ITERATIONS

Determines the number of passes of Argon2 that are used to generate the Cache.

#### Permitted values
Any positive integer.

#### Notes
The time needed to initialize the Cache is proportional to the value of this constant.

### RANDOMX_ARGON_LANES

The number of parallel lanes for Cache initialization.

#### Permitted values
Any positive integer.

#### Notes
This parameter determines how many threads can be used for Cache initialization. 

### RANDOMX_ARGON_SALT

Salt value for Cache initialization.

#### Permitted values
Any string of byte values.

#### Note
Every implementation should choose a unique salt value.

### RANDOMX_CACHE_ACCESSES

The number of random Cache access per Dataset item.

#### Permitted values
Any integer greater than 1.

#### Notes
This value directly determines the performance ratio between the 'fast' and 'light' modes. 

### RANDOMX_SUPERSCALAR_LATENCY
Target latency for SuperscalarHash, in cycles of the reference CPU.

#### Permitted values
Any positive integer.

#### Notes
The default value was tuned so that a high-performance superscalar CPU running at 2-4 GHz will execute SuperscalarHash in similar time it takes to load data from RAM (40-80 ns). Using a lower value will make Dataset generation (and light mode) more memory bound, while increasing this value will make Dataset generation (and light mode) more compute bound.

### RANDOMX_DATASET_BASE_SIZE

Dataset base size in bytes.

#### Permitted values
Integer powers of 2 in the range 64 - 4294967296 (inclusive).

#### Note
This constant affects the memory requirements in fast mode. Some values are unsafe depending on other parameters. See [Unsafe configurations](#unsafe-configurations).

### RANDOMX_DATASET_EXTRA_SIZE

Dataset extra size in bytes.

#### Permitted values
Non-negative integer divisible by 64.

#### Note
This constant affects the memory requirements in fast mode. Some values are unsafe depending on other parameters. See [Unsafe configurations](#unsafe-configurations).

### RANDOMX_PROGRAM_SIZE

The number of instructions in a RandomX program.

#### Permitted values
Any positive integer divisible by 8.

#### Notes
Smaller values will make RandomX more DRAM-latency bound, while higher values will make RandomX more compute-bound. Some values are unsafe. See [Unsafe configurations](#unsafe-configurations).

### RANDOMX_PROGRAM_ITERATIONS

The number of iterations per program.

#### Permitted values
Any positive integer.

#### Notes
Time per hash increases linearly with this constant. Smaller values will increase the overhead of program compilation, while larger values may allow more time for optimizations. Some values are unsafe. See [Unsafe configurations](#unsafe-configurations).

### RANDOMX_PROGRAM_COUNT

The number of programs per hash.

#### Permitted values
Any positive integer.

#### Notes
Time per hash increases linearly with this constant. Some values are unsafe. See [Unsafe configurations](#unsafe-configurations).

### RANDOMX_JUMP_BITS
Jump condition mask size in bits.

#### Permitted values
Positive integers. The sum of `RANDOMX_JUMP_BITS` and `RANDOMX_JUMP_OFFSET` must not exceed 16.

#### Notes
This determines the jump probability of the CBRANCH instruction. The default value of 8 results in jump probability of <code>1/2<sup>8</sup> = 1/256</code>. Increasing this constant will decrease the rate of jumps (and vice versa).

### RANDOMX_JUMP_OFFSET
Jump condition mask offset in bits.

#### Permitted values
Non-negative integers. The sum of `RANDOMX_JUMP_BITS` and `RANDOMX_JUMP_OFFSET` must not exceed 16.

#### Notes
Since the low-order bits of RandomX registers are slightly biased, this offset moves the condition mask to higher bits, which are less biased. Using values smaller than the default may result in a slightly lower jump probability than the theoretical value calculated from `RANDOMX_JUMP_BITS`.

### RANDOMX_SCRATCHPAD_L3
RandomX Scratchpad size in bytes.

#### Permitted values
Any integer power of 2. Must be larger than or equal to `RANDOMX_SCRATCHPAD_L2`.

#### Notes

The default value of 2 MiB was selected to match the typical cache/core ratio of desktop processors. Using a lower value will make RandomX more core-bound, while using larger values will make the algorithm more latency-bound. Some values are unsafe depending on other parameters. See [Unsafe configurations](#unsafe-configurations).

### RANDOMX_SCRATCHPAD_L2

Scratchpad L2 size in bytes.

#### Permitted values
Any integer power of 2. Must be larger than or equal to `RANDOMX_SCRATCHPAD_L1`.

#### Notes
The default value of 256 KiB was selected to match the typical per-core L2 cache size of desktop processors. Using a lower value will make RandomX more core-bound, while using larger values will make the algorithm more latency-bound.

### RANDOMX_SCRATCHPAD_L1

Scratchpad L1 size in bytes.

#### Permitted values
Any integer power of 2. The minimum is 64 bytes.

#### Notes
The default value of 16 KiB was selected to be about half of the per-core L1 cache size of desktop processors. Using a lower value will make RandomX more core-bound, while using larger values will make the algorithm more latency-bound.

### RANDOMX_FREQ_*

Instruction frequencies (per 256 instructions).

#### Permitted values
There is a total of 29 different instructions. The sum of frequencies must be equal to 256.

#### Notes

Making large changes to the default values is not recommended. The only exceptions are the instruction pairs IROR_R/IROL_R, FADD_R/FSUB_R and FADD_M/FSUB_M, which are functionally equivalent.

## Unsafe configurations

There are some configurations that are considered 'unsafe' because they affect the security of the algorithm against attacks. If the conditions listed below are not satisfied, the configuration is unsafe and a compilation error is emitted when building the RandomX library.

These checks can be disabled by definining `RANDOMX_UNSAFE` when building RandomX, e.g. by using `-DRANDOMX_UNSAFE` command line switch in GCC or MSVC. It is not recommended to disable these checks except for testing purposes.

### 1. Memory-time tradeoffs

#### Condition
```` 
RANDOMX_CACHE_ACCESSES * RANDOMX_ARGON_MEMORY * 1024 + 33554432 >= RANDOMX_DATASET_BASE_SIZE + RANDOMX_DATASET_EXTRA_SIZE
```` 

Configurations not satisfying this condition are vulnerable to memory-time tradeoffs, which enables efficient mining in light mode.

#### Solutions

* Increase `RANDOMX_CACHE_ACCESSES` or `RANDOMX_ARGON_MEMORY`. 
* Decrease `RANDOMX_DATASET_BASE_SIZE` or `RANDOMX_DATASET_EXTRA_SIZE`.

### 2. Insufficient Scratchpad writes

#### Condition
```` 
(128 + RANDOMX_PROGRAM_SIZE * RANDOMX_FREQ_ISTORE / 256) * (RANDOMX_PROGRAM_COUNT * RANDOMX_PROGRAM_ITERATIONS) >= RANDOMX_SCRATCHPAD_L3
```` 

Configurations not satisfying this condition are vulnerable to Scratchpad size optimizations due to low amount of writes.

#### Solutions

* Increase `RANDOMX_PROGRAM_SIZE`, `RANDOMX_FREQ_ISTORE`, `RANDOMX_PROGRAM_COUNT` or `RANDOMX_PROGRAM_ITERATIONS`. 
* Decrease `RANDOMX_SCRATCHPAD_L3`.

### 3. Program filtering strategies

#### Condition
```
RANDOMX_PROGRAM_COUNT > 1
```

Configurations not satisfying this condition are vulnerable to program filtering strategies.

#### Solution

* Increase `RANDOMX_PROGRAM_COUNT` to at least 2.

### 4. Low program entropy

#### Condition
```
RANDOMX_PROGRAM_SIZE >= 64
```

Configurations not satisfying this condition do not have a sufficient number of instruction combinations. 

#### Solution

* Increase `RANDOMX_PROGRAM_SIZE` to at least 64.

### 5. High compilation overhead

#### Condition
```
RANDOMX_PROGRAM_ITERATIONS >= 400
```

Configurations not satisfying this condition have a program compilation overhead exceeding 10%.

#### Solution

* Increase `RANDOMX_PROGRAM_ITERATIONS` to at least 400.

