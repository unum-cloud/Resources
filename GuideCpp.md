# Code Guidelines for C++

Modern C++ has a huge language specification, often with too many ways to implement the same functionality.
This file contains guidelines and selects a subset of C++ that we use across our core libraries, with supporting performance and design considerations.

We follow those where high quality is expected, such as implementing our Databases or the Networking step.
We have to loosen the constraints at the boundary between our code and third-party software, which doesn't fully conform.
We occasionally compromise some rules in auxiliary components, such as language bindings, or additional tooling, such as Python wrappers for C++ code or CLI DBMS management tools.

- [Code Guidelines for C++](#code-guidelines-for-c)
  - [Language Features](#language-features)
    - [`throw`](#throw)
    - [`virtual`](#virtual)
    - [`goto`](#goto)
    - [`noexcept` and `noexcept(false)`](#noexcept-and-noexceptfalse)
  - [Styling](#styling)
    - [Project Directory Structure](#project-directory-structure)
    - [Naming Conventions](#naming-conventions)
    - [Other Preferences](#other-preferences)
  - [Dependencies](#dependencies)
    - [Standard Templates Library](#standard-templates-library)
    - [Frequently used Third Party Libraries](#frequently-used-third-party-libraries)
      - [Google Test](#google-test)
      - [Google Benchmark](#google-benchmark)
      - [FMT](#fmt)
  - [Performance Optimizations](#performance-optimizations)
    - [Avoid Dynamic Memory Allocation](#avoid-dynamic-memory-allocation)
    - [Thread Synchronization is Expensive](#thread-synchronization-is-expensive)
    - [Branch-less Computing](#branch-less-computing)
    - [Use SIMD Intrinsics](#use-simd-intrinsics)
    - [Befriend your Compiler](#befriend-your-compiler)
      - [GCC `target`](#gcc-target)
      - [GCC `malloc`, `alloc_size`](#gcc-malloc-alloc_size)
      - [GCC `pure`, `const`](#gcc-pure-const)
      - [GCC `hot`](#gcc-hot)
      - [GCC `regparm` (number), `sseregparm`](#gcc-regparm-number-sseregparm)

## Language Features

To avoid going through every part of the C++ standard, let's just specify the most obvious differences between our other codebases.

|                  |       We don't use |          We use |
| :--------------- | -----------------: | --------------: |
| Others use       | `throw`, `virtual` |             ... |
| Others don't use |                ... | `goto`, `union` |

### `throw`

Our systems should survive high load with grace.
Even when the user has a Terabyte of RAM, he may analyze a 100 TB dataset, so we must always expect RAM starvation.
Throwing exceptions, including `std::bad_alloc`, would slow us down, especially on 100-core systems.

### `virtual`

Polymorphism comes in 2 forms - static and dynamic.
Combining meta-programming and dynamic polymorphism almost universally leads to ugly code as the project matures.
Choosing between the two - we prefer the zero ~~runtime~~ cost variant.

It has another benefit as well.
When dealing with heterogeneous binaries intended for CPU+GPU setups, the lack of dynamic polymorphism makes kernels more portable and makes it easier to reason about the addresses of the called functions.

### `goto`

This language feature is generally considered a code-smell, but given the low-level nature of our code, using it often leads to shorter, cleaner, and more performant code, especially when implementing complex state automata.

### `noexcept` and `noexcept(false)`

In case we are forced to use exceptions, such as for compatibility with STL, we explicitly mark those functions `noexcept(false)`.

## Styling

We use `.clang-format` for automatic formatting.
We derive from the LLVM code style. Here are the primary settings that we share across codebases.

```yaml
Language: Cpp
BasedOnStyle:  LLVM
IndentWidth: 4
TabWidth: 4
NamespaceIndentation: None
ColumnLimit: 120
ReflowComments: true
UseTab: Never
PointerAlignment: Left
```

The remaining parameters are optional.
Here is a much more detailed config in project [UStore](https://github.com/unum-cloud/ustore/blob/main/.clang-format).

### Project Directory Structure

- `include/` for your library headers.
- `src/` for implementation code.
- `CMakeLists.txt` in your projects root directory.
- `cmake/` for supporting CMake scripts for third-party dependencies.

We use VSCode as the default IDE.
You can expect the following VSCode-specific root level directories:

- `.vscode/` for build tasks and launchers.
- `.devcontainer/` for in-Docker development environment.

### Naming Conventions

Like the Standard Templates Library, we use `snake_case` for all the class and variable names.
Variable names should be nouns, function names — verbs.

Parsing and highlighting C++ code is harder for modern IDEs than most other languages, as it requires compiling the AST to differentiate the ambiguous symbols.
We use the following suffixes to distinguish the symbols by their names.

| Suffix | Meaning                                             |
| :----- | :-------------------------------------------------- |
| `_t`   | Class, Struct, or Union **T**ype.                   |
| `_k`   | Compile-time (**K**) Constants.                     |
| `_gt`  | Templates = **G**eneric **T**ypes.                  |
| `_at`  | Template type arguments = **A**ny **T**ype.         |
| `_ak`  | Template non-type arguments = **A**ny **K**onstant. |
| `_`    | Private class member.                               |

Examples:

```cpp
template <typename key_at>
class set_gt {
    using key_t = key_at;
    size_t count_ {};
};
using integer_set_t = set_gt<int64_t>;
constexpr static size_t page_size_k = 4096;
```

### Other Preferences

We also prefer the following constructs, where possible.

| Classic                      | Preferred                    |
| :--------------------------- | :--------------------------- |
| `#ifdef X`                   | `#if defined(X)`             |
| `#define FILE_NAME_H`        | `#pragma once`               |
| `typedef x y;`               | `using y = x;`               |
| `auto v = expr(); if (v) {}` | `if (auto v = expr(); v) {}` |
| `if (x) { a; } else { b; }`  | `x ? a : b`                  |

We also recommend using "pedantic" flags across compilers and using following standard attributes:

- `[[nodiscard]]` where the result of function execution can't be silently dropped.
- `[[maybe_unused]]` where the opposite effect is needed.

## Dependencies

We use CMake both for builds, tests, packaging, and dependency management.
It provides two mechanisms:

- `FetchContent_Declare` to add CMake-based dependencies.
- `ExternalProject_Add` to add non-CMake projects as dependencies.

When possible, then former should be used.
In general, we avoid using large frameworks, preferring small manageable dependencies.
This means avoiding:

- Boost.
- Poco.
- Folly.
- Qt.

### Standard Templates Library

STL provides a lot of basic components, but not all of them are born equal.
Here are the suggested alternatives for the ones we dislike.

| STL Component        | Replacement            |
| :------------------- | :--------------------- |
| `std::iostream`      | FMT                    |
| `std::unordered_map` | Abseil                 |
| `std::regex`         | PCRE2, Intel HyperScan |

### Frequently used Third Party Libraries

#### Google Test

```cmake
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.13.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(googletest)
```

#### Google Benchmark

```cmake
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.7.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(benchmark)
```

#### FMT

```cmake
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 9.1.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(fmt)
```

## Performance Optimizations

### Avoid Dynamic Memory Allocation

This is by far the most common issue in most programs we encounter.
Using a `std::vector<std::string>` or `std::vector<std::vector<int>>` may be much easier than allocating once, and having to do manual addressing over a "string tape" or an "integer matrix".

Still, the underlying memory allocator is solving a very complex "resource allocation" problem every time you call it.
It is trying to solve the general case, searching for continuous blocks of needed size in fragmented arenas.
This is already a non-constant time operation.
When it fails, a system call is performed to request more address space from the Operating System.
Once this pessimistic scenario happens, your operation of appending a `char` to a `string` suddenly costs thousands of CPU cycles.

Having  application-specific or scope-specific allocators, will help reuse memory and save CPU time.

- [Cost of System Calls](https://gms.tf/on-the-costs-of-syscalls.html)

### Thread Synchronization is Expensive

### Branch-less Computing

Branching can be disastrous in HPC, especially on cores without a deep branch predictor, such as GPU cores.
In such cases bit-hacks or even trivial table lookups can be faster:

| Ternary Conditional | Table Lookup                   |
| :------------------ | :----------------------------- |
| `x ? a : b`         | `auto arr[2] = {a, b}; arr[x]` |

Similarly if you have two branches that are trivial to evaluate, you may prefer bitwise operators over conditionals:

| Conditional | Bitwise              | Optimal             |
| :---------- | :------------------- | :------------------ |
| `a \|\| b`  | `bool(a) \| bool(b)` | `bool(a) + bool(b)` |

### Use SIMD Intrinsics

It has been shown a number of times, that compilers can't properly vectorize complex code.
Both in our theoretical studies and lectures:

- [Substring Search](https://github.com/ashvardanian/SubstringSearchBenchmark) 10x faster than `libc`.

As well as with well known industry-standard SIMD-accelerated libraries:

- [SIMDJSON](https://github.com/simdjson/simdjson) for JSON parsing.
- [HyperScan](https://github.com/intel/hyperscan) for RegEx matching.

To benefit from Intra-core parallelism, familiarize yourself with following materials:

- [AVX/AVX2 on x86](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#techs=AVX,AVX2)
- [NEON on ARM](https://developer.arm.com/architectures/instruction-sets/simd-isas/neon/intrinsics)

Beware, that even on x86 there are some pitfalls:

1. A considerable number of documented intrinsics may be [missing](https://www.mail-archive.com/gcc-bugs@gcc.gnu.org/msg659597.html) in you compiler toolchain.
2. In AVX2 many instructions have extremely confusing names. For example, `_mm256_srli_epi64` shifts right the bits in every 64-bit integer in the register, while `_mm256_srli_si256` shift the 128-bit lanes inside the 256-bit register. See remaining [`si256` intrinsics here](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=260,260,4201,4185,4179,1876,5534,1382,2438,2457,3630,3603,260,260,632,1877,1877,1874,1874,5156,5535&techs=AVX,AVX2&text=si256).

### Befriend your Compiler

The first step is to define target-wide compilation flags.

|                                     |       [GCC][gcc-flags] | LLVM |
| :---------------------------------- | ---------------------: | ---: |
| Don't care about math precision?    |          `-ffast-math` |      |
| Unrolling `for` loops               |       `-funroll-loops` |      |
| Aligning functions for faster calls | `-falign-functions=64` |      |
| Aligning labels and branches        |     `-falign-jumps=64` |      |

[gcc-flags]: https://gcc.gnu.org/onlinedocs/gcc-12.2.0/gcc/Optimize-Options.html

Once that is done, you can annotate the code to hint further optimizations.

#### GCC `target`

The `target` attribute is used to specify that a function is to be compiled with different `target` options than specified on the command line.
This can be used for instance to have functions compiled with a different ISA (instruction set architecture) than the default.
You can also use the `#pragma GCC target` pragma to set more than one function to be compiled with specific `target` options.
See Function Specific Option Pragmas, for details about the `#pragma GCC target` pragma.

#### GCC `malloc`, `alloc_size`

The `malloc` attribute is used to tell the compiler that a function may be treated as if any non-NULL
pointer it returns cannot alias any other pointer valid when the function returns and that the memory
has undefined content. This will often improve optimization. Standard functions with this property include `malloc` and calloc.
realloc-like functions do not have this property as the memory pointed to does not have undefined content.
The `alloc_size` attribute is used to tell the compiler that the function return value points to memory, where the size is given by one or two of the functions parameters.
GCC uses this information to improve the correctness of `__builtin_object_size`.

#### GCC `pure`, `const`

Many functions have no effects except the return value and their return value depends only on the parameters and/or global variables.
Such a function can be subject to common sub-expression elimination and loop optimization just as an arithmetic operator would be.
These functions should be declared with the attribute `pure`.
The `const` is just slightly more strict class than the `pure` attribute, since function is not allowed to read global memory.

#### GCC `hot`

The `hot` attribute is used to inform the compiler that a function is a `hot` spot of the compiled program.
The function is optimized more aggressively and on many target it is placed into special subsection of the text section so all `hot` functions appears close together improving locality.
When profile feedback is available, via `-fprofile-use`, `hot` functions are automatically detected and this attribute is ignored.

#### GCC `regparm` (number), `sseregparm`

On the Intel 386, the `regparm` attribute causes the compiler to pass arguments number one to number if they are of integral type in registers EAX, EDX, and ECX instead of on the stack.
On the Intel 386 with SSE support, the `sseregparm` attribute causes the compiler to pass up to 3 floating point arguments in SSE registers instead of on the stack.
Functions that take a variable number of arguments will continue to be passed all of their arguments on the stack.
