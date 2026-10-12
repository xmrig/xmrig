# XMRig LoongArch performance notes

This document records the performance work of the LoongArch port in full: where the time of one hash goes, what was changed and what it bought, what was tried and why it was rejected, and what the remaining walls are. Every number below states its conditions: idle machine, offline benchmark, which build, which thread count, how many runs.

The style is deliberately conservative. A measurement that could not be repeated cleanly is marked as such, and one number that had to be thrown out mid investigation is kept in the text, with the reason, as a lesson.

## 1. Method

- **Machine**: Loongson 3A6000, 4 cores, SMT2, 1 MiB L2 per core, 16 MiB shared L3, otherwise idle.
- **Benchmark**: `xmrig --bench=1M`, the offline one million hash run. It prints a hash sum (`898B6E0431C28A6B` for this configuration) that every build and every optimization round reproduced exactly, so speed numbers never come at the cost of correctness.
- **Component benches**: separate harnesses time the scratchpad fill, the final walk, and the whole hash on one thread (`scripts/652`), race the four `hashAndFill` variants on a real 2 MiB scratchpad (`scripts/660`), and run one mining thread under `perf stat` with the dataset init window skipped (`scripts/658`).
- **The uniform session**: the headline table of section 8 comes from one back to back session on the final tree, same binary, same hour, no other work on the machine.

One rule was learned the hard way and is now standing: **nothing else runs on the machine during a benchmark, and the machine is verified idle before every run.** Two numbers had to be discarded along the way. One 8 thread run (947.0 H/s) had a rebuild and a verification suite running next to it. Two runs of the first uniform session attempt (830.7 and 556.4 H/s on the LSX only tree) came back low because the collecting script left the finished miner processes resident between runs; the redo from a verified idle machine reproduced the expected values and is what the table of section 8 shows.

## 2. Where the time of one hash goes

One full mode RandomX hash on one thread, from the uniform session's component bench:

| part | what it is | time | share |
| --- | --- | --- | --- |
| scratchpad fill | `fillAes1Rx4`, 2 MiB | 747.8 µs | 18% |
| final walk | `hashAes1Rx4`, 2 MiB | 884.0 µs | 21% |
| program execution | 8 programs, 2048 iterations each | ~2.6 ms | 61% |

The two AES passes are pure instruction throughput. The program execution is the interesting part, and the first belief about it was wrong.

**The wrong belief.** The generated program body is about 975 words, and with the loop and the tail an iteration looked like ~290 instructions. Divided by the measured cycles per iteration this gave an IPC of 0.56, which said: latency bound, nothing to win by cutting instructions. Both the premise and the conclusion were false.

**The correction.** `perf stat` on one mining thread says:

| counter | per hash |
| --- | --- |
| instructions | ~50.6 M |
| cycles | ~14.0 M |
| **IPC** | **3.61 out of 4 issue slots** |

An iteration is really about 3000 instructions, not 290: the program body is ~975 words, but taken branches re-execute parts of it, and the fused AES pass of the mining path adds its own stream. The core keeps this at 3.61 IPC, which is 90% of its issue width. **The loop is issue bound, and every instruction removed is a real slot freed.** That correction is what round 4 acted on.

## 3. Round 1: the software AES round

No LoongArch CPU carries AES instructions, so the scratchpad fill and the final walk both run a T-table software round. The tree had two implementations:

- `aesenc<1>`: the state goes through a `volatile` stack array and is reloaded byte by byte. About 70 to 90 instructions per block, half of them pure traffic: the store, 16 byte loads, 16 redundant `andi` after `ld.bu`.
- `aesenc<2>`: the four words are picked out once and the bytes are sliced with shifts. About 43 instructions per block, which is the practical floor for a T-table on this ISA: the 16 table reads per block cannot be folded into vectors because LoongArch has no gather instruction.

The port switched every virtual machine to the fast variant (`124b432`):

| part | before | after | speedup |
| --- | --- | --- | --- |
| fillAes1Rx4, 2 MiB | 1390.8 µs | 745.8 µs | 1.86x |
| hashAes1Rx4, 2 MiB | 1272.0 µs | 790.5 µs | 1.61x |
| randomx_calculate_hash, whole | 5396.6 µs | 4316.8 µs | 1.25x |

**Why the miner gained far less than 25%.** The mining steady state uses `hashAndFill`, which fuses the final walk with the next fill and picks its implementation at startup through a measured race (`SelectSoftAESImpl`). That race had already been choosing a fast variant all along; the slow `<1>` round only sat on the non mining API path and the first hash. The switch unified all paths on the fast family and cost nothing, but the mining gain from it was under one percent.

The four candidates were also raced on the real working set (2 MiB, not the 10 KiB arena the startup race uses): `<2,1>` 1530 µs, `<2,2>` 1545 µs, `<2,4>` 1562 µs, `<1,1>` 2682 µs. The startup race picks within the fast family correctly; the spread between the `<2,x>` variants is under 2%.

## 4. Round 2: the LASX walls

The 256 bit LASX layer was investigated for the per hash path and rejected on three independent grounds, each verified rather than assumed:

1. **The integer registers are semantically scalar.** The dataset read of the tail xors into `r0`-`r7`; moving it into vectors would change what the register file means, not just how it is computed. No.
2. **There is no gather.** Pairing two AES blocks per 256 bit register would halve the extraction work, but the 16 table reads per block still come back one by one, and the byte extraction round trip (vector to GPR) costs more than it saves: an estimated 57 instructions per block against the 43 of the current scalar extraction. No.
3. **128 bit operations do not preserve the upper half.** The promising plan was to pair the FP group traffic of the JIT template: 12 `vld` plus 8 conversion plus 8 mask operations plus 4 `vst` collapse into 14 wide operations. The whole plan rests on one assumption: that an LSX instruction on `$vr4` leaves the upper half of `$xr4` (the partner register's data) intact. A probe (`scripts/661`) loaded a four word pattern into `$xr0`, ran an identity `vand.v` on `$vr0`, and stored the full 256 bits: the upper half came back as garbage (`fffffffbcebcf1f`), not as the pattern. On x86, SSE operations preserve the upper half of a YMM register; on LoongArch they do not. The design was discarded the same day, zero commits.

Where LASX does pay is the dataset item function: one cache line across four lanes. Dataset init drops from 6.6 s to 2.8 s. That runs once per epoch and once per restart, never inside a hash.

## 5. Round 3: the emitter diet

The generated memory opcodes (`IADD_M`, `ISUB_M`, `IMUL_M`, `IMULH_M`, `ISMULH_M`, `IXOR_M`, `ISTORE`, the FP loads) rebuilt their scratchpad address mask with `lu12i.w` + `ori` + `and`, three words per access, about 59 accesses per iteration. The AArch64 backend keeps its masks in registers; now the LA64 one does too (`7585715`):

- five callee saved registers, `$r24`-`$r28`, materialized once per program by a patched prologue slot;
- each access becomes a single `and` with its register, two words saved per access;
- the frame grows from 32 to 64 bytes to save and restore them, once per program, not per iteration.

One trap on the way: `AddressMask_Calculated` is a four entry table indexed by two mod bits, not three entries as first assumed. The first version left `modMem` 2 and 3 reading an uninitialized `$r27`, which turned masks into garbage and addresses into SIGBUS. The verification gate caught it immediately (five trees diverged, a generated program crashed); the five register version passes everything.

The measured effect:

| what | before | after |
| --- | --- | --- |
| instructions per 30 s window | 248.92 G | 246.90 G in the uniform session (the diet itself removed 1.91 G, exactly the 59 x 2 words x 16384 iterations x hashes) |
| cycles per 30 s window, default `<1,1>` harness | 68.92 G | 68.85 G |
| 1M benchmark, 4 threads | 881.5 H/s | 895.0 and 907.8 H/s, and 903.9 to 909.1 H/s in the uniform session |
| 1M benchmark, 8 threads | 969.8 H/s | 1011.7 H/s, and 999.8 to 1000.9 H/s in the uniform session |

The two honest readings of this table: on the harness (which runs the slow default AES path and has more total instructions) the removed words were partly executing in shadow, so cycles barely moved. On the miner, which runs the lean `<2,x>` fused path and sits closer to the issue ceiling, the same removal is worth 1.5 to 3%. And on SMT the gain grows to about 4%: two threads share the issue slots of one physical core, so an instruction one thread does not execute is an instruction the sibling thread gets.

## 6. Threads

The thread curve (30 s per point, workers pinned physical core first; the sibling pairs are 0-1, 2-3, 4-5, 6-7):

| threads | H/s | versus one thread |
| --- | --- | --- |
| 1 | 189.9 | 1.00x |
| 2 | 393.8 | 2.07x |
| 3 | 583.5 | 3.07x |
| 4 | 758.1 | 3.99x |
| 5 | 797.9 | 4.20x |
| 6 | 817.7 | 4.31x |
| 7 | 836.5 | 4.40x |
| 8 | 851.7 | 4.48x |

Linear up to the four physical cores, then about 20 H/s per SMT sibling. Two measurement notes worth keeping:

- Pinning two threads to logical CPUs 0 and 1 halves the throughput: those are siblings of one core. The scheduler knows this; explicit pinning must too.
- The 30 s harness window reads about 14% below the 1M benchmark steady state on the same machine (TLB and frequency settling, thread setup). Use the harness for the shape of the curve and the 1M run for absolute numbers.

Deployment guidance follows directly: one thread per physical core is near optimal, `-t 8` buys about 10% more total throughput at half the per thread efficiency, and that is a power and heat decision rather than a software one.

## 7. The walls that remain

- **The RandomX program itself.** 65% of every hash is the latency structure the algorithm prescribes: dependent chains, scratchpad and dataset latency, long FP operations. The port cannot change it without changing the algorithm.
- **The issue width.** At IPC 3.61 the generated loop leaves about 10% of the issue slots on the table, and the remaining fat is thin: the CBRANCH emitter could still lose about one word per occurrence (~0.8%), and the loop top materializes the same L3 mask twice (~0.07%).
- **The huge page accounting.** The dataset sits on 32 MiB transparent huge pages the whole time, but the miner counts explicit 2 MiB pages and prints `huge pages 0% 0/1168`. Accounting only, not performance, but wrong and worth fixing.
- **The CryptoNight family is verified but not benchmarked here.** All 105 reference inputs of `CryptoNight_test.h` match (see the port notes), but this document measures RandomX only, so nothing about CN speed is claimed.

## 8. The uniform session

Final tree, one session on an idle machine, nothing else running, every run a full 1M benchmark. The two discarded runs of the first attempt are left out; see section 1 for how they failed.

| build | threads | run 1 | run 2 |
| --- | --- | --- | --- |
| LSX + LASX | 4 | 909.1 H/s | 903.9 H/s |
| LSX + LASX | 8 | 999.8 H/s | 1000.9 H/s |
| LSX only | 4 | 904.3 H/s | 906.0 H/s |

Every run printed the hash sum `898B6E0431C28A6B`. The dataset init of the same session: 2817 to 2968 ms for the LASX tree, 6576 to 6703 ms for the LSX only tree, about 2.3x apart. The run to run spread inside one condition is under 0.6%, which is the noise floor of this benchmark on this machine.

Two conclusions read directly off the table:

- **The two builds mine at the same speed.** 904.3 against 903.9 to 909.1 H/s is inside the noise, on the same binary generation, same hour. The choice between the trees is decided by the dataset init time and by whether the machine has LASX at all, never by the hash rate.
- **SMT buys about 10%.** 1000.35 against 906.5 H/s on the midpoints is a 10.4% gain for doubling the thread count, at half the per thread efficiency.

## 9. What this means in practice

- On a machine with LASX, build the main tree: same mining speed as the LSX only tree, 2.3x faster dataset init.
- On a machine without LASX, build with `-DWITH_LASX=OFF`: the binary carries no LASX instruction and mines at the same rate.
- Run one thread per physical core by default; add the SMT siblings only if the extra ~10% total throughput is worth half the per thread efficiency.
