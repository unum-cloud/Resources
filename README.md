# Unum-related Resources

Hereby we present a list of awesome resources and blogs on topics related to Unum!
Suggested for everyone wishing to join the core Unum team or contribute to our Open-Source projects!

- :octocat: means source code
- :tv: means video channels
- :scroll: means textual blogs
- 📖 means complete books
- 🇷🇺 implies content is in Russian

## Learning Python

- :octocat: [WTF Python](https://github.com/satwikkansal/wtfpython): Understanding Python through surprising snippets..
- :tv: [mCoding](https://www.youtube.com/c/mCodingWithJamesMurphy) by James Murphy.
- 📖 [Python Design Patterns](https://python-patterns.guide/) by Brandon Rhodes.
- :scroll: [Lesser known parts of Python standard library](https://www.trickster.dev/post/lesser-known-parts-of-python-standard-library/) by "Trickster Dev".
- :octocat: [Numba](https://numba.readthedocs.io/) - JIT compiler for Python, translates Python to optimized machine code.
- :octocat: [CuPy](https://docs.cupy.dev/en/stable/user_guide/kernel.html) - NumPy-compatible array library for GPU-accelerated computing.
- 📜 [Python C API](https://docs.python.org/3/c-api/) - official documentation for writing C extensions.
- :octocat: [NanoBind](https://github.com/wjakob/nanobind) - minimal Python bindings, successor to PyBind11.
- :octocat: [Cppyy](https://cppyy.readthedocs.io/) - JIT compiler for Python interfacing with C++.

## Learning C++

Learning C++ can be tricky and few good courses exist:

- :tv: 🇷🇺 [MIPT Course in Basic C++](https://youtube.com/playlist?list=PL3BR09unfgciJ1_K_E914nohpiOiHnpsK)

After covering the basics, use [CppReference.com](https://en.cppreference.com/w/) as a reference for the C++ standard library.
It's generally more [reliable than CPlusPlus.com](https://www.reddit.com/r/cpp/comments/1b0lp7u/comment/ks8vgy7/?utm_source=share&utm_medium=web2x&context=3).
Other useful manuals for low-level engineering include:

- [Arm intrinsics](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
- [Intel intrinsics](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html) for x86 CPUs
- [CUDA intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/modules.html#modules) for Nvidia GPUs
- [Clang built-ins](https://clang.llvm.org/docs/LanguageExtensions.html#builtin-functions)
- [GCC built-ins](https://gcc.gnu.org/onlinedocs/gcc-4.7.2/gcc/Other-Builtins.html#Other-Builtins)
- [C++ Header Sizes](https://artificial-mind.net/projects/compile-health/)
- :octocat: [Compiler Explorer (Godbolt)](https://godbolt.org/) - online tool for comparing assembly output across compilers

## Engineering & Performance

- 📜 [Algorithms for Competitive Programming](https://cp-algorithms.com/#data-structures) blog
- :octocat: [C++ Benchmarking Tutorial](https://github.com/ashvardanian/BenchmarkingTutorial) for Google toolchain: `std::rand`, `std::execution`
- :octocat: [Experimental Concurrency](https://github.com/s9w/experimental_concurrency) benchmarks for C++ STL concurrent primitives: `std::thread`, `std::atomic`, `std::mutex`
- :octocat: [Parallel Reductions](https://github.com/unum-cloud/ParallelReductions): CUDA, Intel TBB, SIMD, x86 AVX2, ARM Neon
- :octocat: [Foreign Function Interface Overheads](https://github.com/dyu/ffi-overhead)
- :octocat: [Intel oneTBB](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onetbb.html) - Threading Building Blocks for parallel programming
- 📜 [Linux perf](https://perf.wiki.kernel.org/) - profiling tool for performance analysis
- 📜 [eBPF](https://ebpf.io/) - running sandboxed programs in Linux kernel for tracing
- 📜 [Concurrency Costs](https://travisdowns.github.io/blog/2020/07/06/concurrency-costs) by Travis Downs - analysis of concurrency performance
- 📜 [LWN](https://lwn.net): latest news from the kernel space
- 📜 [Linux Insides](https://github.com/0xAX/linux-insides) book
- :octocat: [UCSB Benchmark](https://github.com/unum-cloud/UCSB) for persistent storage: io_uring, eBPF, RocksDB, LevelDB, WiredTiger.
- :octocat: [SMHasher](https://github.com/aappleby/smhasher/wiki) by Austin Appleby - test suite for hash functions

## AI & Data Science

- 📜 [Parallelism strategies for training large Neural Networks](https://openai.com/blog/techniques-for-training-large-neural-networks/) blog by OpenAI
- 📜 [Semantic search with embeddings](https://rom1504.medium.com/semantic-search-with-embeddings-index-anything-8fb18556443c) blog by Romain Beaumont  - googler, Laion5B dataset and open_clip contributor

## DevOps & Tooling

- 📜 [Common Applications of k8s](https://blog.adamchalmers.com/kubernetes-problems/)
- :octocat: [cibuildwheel](https://cibuildwheel.readthedocs.io/) - tool for building Python wheels across platforms
