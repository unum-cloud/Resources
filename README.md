# Unum Resources

Instruction latencies and throughputs measured on the chips Unum's libraries target, and a guide tying each technology to the code that uses it.
Every row comes from the `measure` binary in this repository, with its method recorded beside the numbers; nothing is copied from vendor guides or third-party tables.
Code and rows are licensed under Apache 2.0.

## Technology Guide

### Arm SME

Outer products into the ZA tile array, entered with `SMSTART` and left with `SMSTOP`.
NumKong uses it for dense and binary dot products, spatial distances, attention and MaxSim on Apple M4 and M5, and USearch reaches it through NumKong's tile kits.
Measured on [Apple M5 Pro](https://github.com/unum-cloud/Resources/blob/main/data/arm/apple-m5-pro/sme2.csv), per core kind.
References: [Arm intrinsics](https://developer.arm.com/architectures/instruction-sets/intrinsics/) and the [A64 instruction set](https://developer.arm.com/documentation/ddi0602/latest/).
Articles: [NumKong](https://ashvardanian.com/posts/numkong/) and the Arm kernels of [less_slow.cpp](https://github.com/ashvardanian/less_slow.cpp/blob/main/less_slow_aarch64.S).

### Arm SVE

Scalable vectors with predicate registers; Apple chips run them only in streaming mode, inside the SME unit.
NumKong and StringZilla ship SVE and SVE2 kernels for Arm servers, and SmashTable probes its rows with SVE.
Streaming SVE is measured on [Apple M5 Pro](https://github.com/unum-cloud/Resources/blob/main/data/arm/apple-m5-pro/sve-streaming.csv), per core kind.
Reference: [Arm intrinsics](https://developer.arm.com/architectures/instruction-sets/intrinsics/).
Articles: [2x Faster Hashes on AWS Graviton: NEON → SVE2](https://ashvardanian.com/posts/aws-graviton-checksums-on-neon-vs-sve/) and [set intersections on SVE2 and AVX-512](https://ashvardanian.com/posts/simd-set-intersections-sve2-avx512/).

### Arm NEON

Fixed 128-bit vectors on every 64-bit Arm core, extended with dot products, I8MM and BF16 on newer ones.
NumKong, StringZilla, SmashTable and USearch use it as their baseline Arm tier.
Measured on [Apple M5 Pro](https://github.com/unum-cloud/Resources/blob/main/data/arm/apple-m5-pro/neon.csv), per core kind.
References: [Arm intrinsics](https://developer.arm.com/architectures/instruction-sets/intrinsics/) and the FMA and dot-product peaks of [less_slow.cpp](https://github.com/ashvardanian/less_slow.cpp).
Articles: [2x Faster Hashes on AWS Graviton: NEON → SVE2](https://ashvardanian.com/posts/aws-graviton-checksums-on-neon-vs-sve/) and the Arm kernels of [less_slow.cpp](https://github.com/ashvardanian/less_slow.cpp/blob/main/less_slow_aarch64.S).

### AVX2 & AVX-512

256-bit vectors on every x86 core since Haswell, and 512-bit vectors with mask registers since Skylake-X and Zen 4.
NumKong, StringZilla and SmashTable build their x86 kernels on both, and USearch calls them for its own distance kernels.
References: [Intel intrinsics](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html) and the x86 peaks of [less_slow.cpp](https://github.com/ashvardanian/less_slow.cpp).
Articles: [Hiding x86 Port Latency for 330 GB/s/core Reductions](https://ashvardanian.com/posts/cpu-ports/), [GCC Compiler vs Human, 119x Faster Assembly](https://ashvardanian.com/posts/gcc-12-vs-avx512fp16/), [Full Unicode Search at 50x ICU Speed with AVX-512](https://ashvardanian.com/posts/search-utf8/), [set intersections on SVE2 and AVX-512](https://ashvardanian.com/posts/simd-set-intersections-sve2-avx512/) and the x86 kernels of [less_slow.cpp](https://github.com/ashvardanian/less_slow.cpp/blob/main/less_slow_amd64.S).

### Intel AMX

Tile registers and a tile matrix-multiply unit on Sapphire Rapids and newer Xeons, which Linux enables per process on request.
NumKong uses it for dense dot products, attention and MaxSim, and USearch reaches it through NumKong's tile kits.
Reference: [Intel intrinsics](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html).

### WebAssembly SIMD

128-bit vectors in browsers and server runtimes, with Relaxed SIMD adding fused multiply-adds and dot products.
NumKong and StringZilla ship v128 and Relaxed SIMD kernels, and SmashTable probes its rows with v128.
Article: [the complexity of SIMD code](https://ashvardanian.com/posts/understanding-simd-complexity/).

### Wait Instructions

`UMWAIT` and `TPAUSE` on x86 and `WFE` and `WFET` on Arm park a core until a watched cache line changes or a deadline passes.
ForkUnion's thread pools wait on them instead of spinning.
References: [Concurrency Costs](https://travisdowns.github.io/blog/2020/07/06/concurrency-costs) by Travis Downs and the [Experimental Concurrency](https://github.com/s9w/experimental_concurrency) benchmarks.

### io_uring & NVMe

Submission and completion rings shared with the Linux kernel, for sockets and for direct reads and writes on NVMe namespaces.
UCall serves its requests over io_uring, UStore keeps blobs, strings and graphs on raw NVMe, and USearch reads its NVMe tier through the same rings.
References: [UCSB](https://github.com/unum-cloud/UCSB) for persistent storage benchmarks and [LWN](https://lwn.net) for kernel news.

## Measuring Rows

The `measure` binary times every form on each core kind of the host in turn, on one thread.
On Linux it pins itself to the first CPU of the kind, with hybrid x86 kinds read from the `cpu_*` PMU folders.
On macOS, which has no affinity API on Apple chips, a QoS class steers the thread: user-interactive for the fastest level, background for the slowest, and runs that start or end on another level are discarded.
Related benchmarks live in [less_slow.cpp](https://github.com/ashvardanian/less_slow.cpp) and [less_slow.rs](https://github.com/ashvardanian/less_slow.rs), and port pressure is worked through in [Hiding x86 Port Latency for 330 GB/s/core Reductions](https://ashvardanian.com/posts/cpu-ports/) and [the complexity of SIMD code](https://ashvardanian.com/posts/understanding-simd-complexity/).

### Kernels

Each form is one line of the `forms!` table in `arm.rs` or `x86.rs`, and the line's step template becomes two kernels of assembly text, with `UNROLL` and with `2 × UNROLL` copies per loop iteration.
The difference of their times, divided by the iterations and by `UNROLL`, cancels the loop, the call and any mode switch around it.
In a template, `\v` is the rotated operand and `\fixed` the operand its register class reserves, the class's highest-numbered registers.
Latency kernels make every copy depend on the previous one through `\v`.
Throughput kernels rotate `\v` over the whole class: 31 vector registers, every ZA tile of the element size, 16 ZA slices, the 8 vector-group offsets of the ZA array, 8 predicates, 15 VEX or 31 EVEX vector registers, 6 tiles, or the cache lines of an atomic form.
`independent` lines feed only `\fixed` to their throughput kernels, and `throughput` lines, such as `MOVA` between Z and ZA, have no latency kernel.
`UNROLL` is the least common multiple of every class's rotation, so each throughput kernel splits evenly.
Forms that no single template can time are written out below the table: round trips such as `MOVA` through ZA and `LDR q` then `FMOV`, the `SMSTART` and `SMSTOP` pairs, exclusive load and store pairs, and register-pair `CASP`.

Atomic forms come in three kinds of kernels.
Uncontended rows run on one core: each line holds its own address, so a latency kernel feeds the value an atomic returns into the next one's address, and a throughput kernel rotates over lines that stay in L1 across repeats.
Arm atomics narrower than 64 bits, whose values cannot hold an address, chain the value they return into the next one's operand instead.
Contended rows, with two pinned threads passing one cache line, and wait rows, with one thread parked on a monitored line by `WFE`, `WFET`, `UMWAIT` or `TPAUSE` while another writes it after a known delay, come with the fleet.
Both kinds report nanoseconds and core cycles per round trip, name the core pair in `topology`, and wait rows add the deadline they asked for and how far past it the core woke.

### References

No cycle counter is read.
Each kernel pair is timed interleaved with a reference chain of dependent adds in its own clock domain, and the ratio of the two differences is the cycle count.
Core forms take the 1-cycle scalar add chain.
On x86 that chain is `add r64, r64`, because Golden Cove and later fold dependent adds of an immediate at rename.
SME and streaming SVE forms execute in the SME unit, which clocks apart from the core, so they take a streaming `ADD z.s` chain, whose own cost in core cycles is measured against the scalar chain once per core kind and run, and their rows count core cycles at that ratio.

### Settling

The only chosen number is the precision the table claims, 1%; everything else is derived from it, the timer and the form set.
The smallest step between two timer readings, the tick or the cost of a reading, sets the run length: four minima enter every ratio, each off by up to one step, so the shorter kernel of each pair runs for four steps divided by the precision.
Rounds double the samples of the four interleaved kernels, 1, 2, 4 and so on, and a form settles when two consecutive rounds' minima give ratios within 1% of each other.
A form's rounds share one instance hour divided by the forms, core kinds and runs, and a form whose next round would overrun it is reported as unstable and dropped from that run.
Each row is the median of `--runs` passes over every form and core kind, 3 by default, published only when most passes settled.

```sh
cargo run --release                      # writes data/<isa>/<chip>/<extension>.csv
cargo run --release -- --runs 5          # the median of five passes
```

The chip folder defaults to the lowercased CPU brand string, and `--chip` overrides it.

## Row Columns

- `mnemonic`: the instruction, as the vendor manual spells it.
- `form`: operand shapes with register numbers dropped; a `->` chain names a round trip.
- `input_type`: the type of the first source operand, such as `u8` for `USMOPA` and `VPDPBUSD`, which multiply unsigned by signed bytes.
- `output_type`: the type of the result, the input type again unless the form widens or narrows.
- `core`: the core kind, as the operating system names it; WSL hides the hybrid layout of the Intel i9-13950HX, and its rows match the Raptor Cove P-core, so they name it.
- `mode`: `standard`, or `streaming` for Arm forms that run in streaming mode inside the SME unit.
- `latency_cycles`: core cycles from an input to the dependent result; for `SMSTART` rows and round trips, per pair.
- `throughput_cycles`: reciprocal throughput, core cycles per instruction with the whole class in flight.
- `nanoseconds`: the length of one core cycle while the row was measured, so cycles times it give time.
- `reference`: the add chain the row was timed against, `add x` or `add r64` on the core, `add z.s` in streaming mode.
- `reference_cycles`: core cycles per add of that chain, 1 on the core.
- `rounds`: doubling rounds until the slower of the row's kernels settled, the most among the passes.
- `intrinsics`: the intrinsics that compile to the form, space-separated.
- `uops` and `ports`: micro-ops and execution ports in uops.info notation, such as `1*p05+1*p23`, empty where not measured.

Contended and wait rows add three columns when they arrive with the fleet.

- `topology`: the core pair, such as `smt-sibling`, `same-cluster`, `cross-cluster`, `same-ccd`, `cross-ccd`, `same-die` or `cross-socket`.
- `requested_nanoseconds`: the deadline a wait row asked for.
- `oversleep_nanoseconds`: how far past that deadline the core woke.
