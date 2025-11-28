# Fast Concurrent Programming - Complete Documentation

This repository includes comprehensive documentation on concurrent and GPU programming. Due to GitHub's limited support for RST (reStructuredText) files, we provide documentation in two formats:

## Quick Navigation

### 1. HTML Documentation (Recommended)
The best viewing experience is via the built HTML documentation:

```bash
cd docs
make html
# Then open build/html/index.html in your browser
```

This provides full navigation, proper formatting for tables, code blocks, and diagrams.

---

## 2. Documentation Structure

### CPU Concurrency Guides
Learn fundamental concurrent programming concepts with Python threading, asyncio, and multiprocessing.

| Topic | Focus | Level |
|-------|-------|-------|
| [Key Concepts](docs/source/cpu-concurrency/key_concepts.rst) | Concurrency vs parallelism, GIL, event loops, coroutines | Beginner |
| [Hardware Parallelism](docs/source/cpu-concurrency/hardware_parallelism.rst) | Multi-core architecture, NUMA, cache hierarchy | Intermediate |
| [Threading Basics](docs/source/cpu-concurrency/threading_basics.rst) | Thread creation, management, lifecycle | Beginner |
| [Asyncio Event Loop](docs/source/cpu-concurrency/asyncio_event_loop.rst) | Event-driven programming, async I/O patterns | Intermediate |
| [Asyncio Coroutines](docs/source/cpu-concurrency/asyncio_coroutine.rst) | Coroutine definition, async/await syntax | Intermediate |
| [Asyncio and Futures](docs/source/cpu-concurrency/asyncio_and_futures.rst) | Future-based concurrency, callbacks | Intermediate |
| [Asyncio Task Manipulation](docs/source/cpu-concurrency/asyncio_task_manipulation.rst) | Task creation, management, cancellation | Advanced |
| [Concurrent Futures](docs/source/cpu-concurrency/concurrent_futures_pooling.rst) | ThreadPoolExecutor, ProcessPoolExecutor, futures | Intermediate |
| [Queue Explained](docs/source/cpu-concurrency/queue_explained.rst) | Queue data structure, thread-safe operations | Intermediate |
| [Queue Internals](docs/source/cpu-concurrency/queue_internal_mechanics.rst) | Queue implementation, condition variables | Advanced |
| [Task Done Queue](docs/source/cpu-concurrency/task_done_queue_explained.rst) | Task synchronization, done signaling | Intermediate |
| [RLock Explained](docs/source/cpu-concurrency/rlock_explained.rst) | Reentrant locks, recursive synchronization | Intermediate |
| [Semaphore Explained](docs/source/cpu-concurrency/semaphore_explained.rst) | Semaphore patterns, resource counting | Intermediate |
| [Patterns & Problems](docs/source/cpu-concurrency/patterns_problems_mapping.rst) | Common patterns, deadlock, race conditions | Advanced |

### GPU Programming Guides
Master GPU architecture, optimization techniques, and Triton programming.

| Topic | Focus | Level |
|-------|-------|-------|
| [GPU Fundamentals](docs/source/gpu-concepts/gpu-fundamentals.rst) | GPU architecture, threads, warps, blocks | Beginner |
| [Memory Hierarchy](docs/source/gpu-concepts/memory-hierarchy.rst) | Global, shared, local memory; cache; bandwidth | Intermediate |
| [Execution Model](docs/source/gpu-concepts/execution-model.rst) | Thread execution, synchronization, atomics | Advanced |
| [Performance Optimization](docs/source/gpu-concepts/performance-optimization.rst) | Memory access patterns, occupancy, roofline | Advanced |
| [Triton Concepts](docs/source/gpu-concepts/triton-concepts.rst) | Kernel launching, grid specification, constexpr | Intermediate |

### GPU Tutorials (Step-by-Step Examples)
Progressive tutorials from basic to advanced GPU kernel programming.

| # | Topic | Concepts | Level |
|---|-------|----------|-------|
| 1 | [Vector Addition](docs/source/gpu-tutorials/01-vector-add.rst) | Basic kernel, grid launch, masking | Beginner |
| 2 | [Fused Softmax](docs/source/gpu-tutorials/02-fused-softmax.rst) | Row-wise reductions, numerical stability | Intermediate |
| 3 | [Matrix Multiplication](docs/source/gpu-tutorials/03-matrix-multiplication.rst) | Block tiling, memory hierarchy, cache | Intermediate |
| 4 | [Low Memory Dropout](docs/source/gpu-tutorials/04-low-memory-dropout.rst) | Memory efficiency, masking patterns | Intermediate |
| 5 | [Layer Normalization](docs/source/gpu-tutorials/05-layer-norm.rst) | Normalization kernels, fusion | Intermediate |
| 6 | [Fused Attention](docs/source/gpu-tutorials/06-fused-attention.rst) | Complex kernel fusion, online softmax | Advanced |
| 7 | [Extern Functions](docs/source/gpu-tutorials/07-extern-functions.rst) | CUDA integration, custom ops | Advanced |
| 8 | [Grouped GEMM](docs/source/gpu-tutorials/08-grouped-gemm.rst) | Batched operations, dynamic shapes | Advanced |
| 9 | [Persistent MatMul](docs/source/gpu-tutorials/09-persistent-matmul.rst) | Persistent kernel pattern, work distribution | Advanced |
| 10 | [Block Scaled MatMul](docs/source/gpu-tutorials/10-block-scaled-matmul.rst) | Block-wise scaling, quantization-aware | Advanced |

### Triton Compiler Deep Dive
Understanding Triton's compilation pipeline and optimization.

| Topic | Focus | Level |
|-------|-------|-------|
| [Compiler Overview](docs/source/triton-compiler/01-overview.rst) | Pipeline stages, IR representations | Intermediate |
| [JIT Decorator](docs/source/triton-compiler/02-jit-decorator.rst) | Specialization, caching, kernel variants | Advanced |
| [Compilation Pipeline](docs/source/triton-compiler/03-compilation-pipeline.rst) | Python→Triton IR→MLIR→LLVM→PTX→Cubin | Advanced |
| [CUDA Comparison](docs/source/triton-compiler/04-cuda-comparison.rst) | Triton vs CUDA: code, compilation, optimization | Intermediate |
| [MLIR Concepts](docs/source/triton-compiler/05-mlir-concepts.rst) | MLIR dialects, transformations, lowering | Advanced |

### Learning Resources
Structured learning paths and reference materials.

| Resource | Content |
|----------|---------|
| [Learning Paths](docs/source/learning-paths.rst) | Beginner → Intermediate → Advanced progressions |
| [Troubleshooting](docs/source/troubleshooting.rst) | Common issues, debugging, performance problems |
| [References](docs/source/references.rst) | External links, official documentation, papers |

---

## Building Documentation Locally

### Prerequisites
```bash
pip install sphinx sphinx-rtd-theme
```

### Build HTML Documentation
```bash
cd docs
make html
open build/html/index.html  # macOS
# or
xdg-open build/html/index.html  # Linux
# or
start build/html/index.html  # Windows
```

### Build Other Formats
```bash
# PDF (requires LaTeX)
make latexpdf

# Man pages
make man

# EPUB
make epub
```

---

## Key Concepts Quick Reference

### Concurrent Programming Patterns
- **Mutual Exclusion (Locks)**: Prevent concurrent access to shared resources
- **Condition Variables**: Wait/notify synchronization between threads
- **Producer-Consumer**: Data flow between threads with bounded queues
- **Thread Pools**: Reusable worker threads for task execution
- **Async/Await**: Efficient I/O-bound concurrency without explicit threading

### GPU Programming Fundamentals
- **SPMD Model**: Single Program, Multiple Data execution
- **Thread Organization**: Threads → Warps → Blocks → Grids
- **Memory Hierarchy**: Global → Shared → Local memory with different speeds
- **Synchronization**: `__syncthreads()` for block-level, atomic ops for device-level
- **Occupancy**: Ratio of active warps to maximum possible warps

### Triton Concepts
- **@triton.jit**: JIT decorator for GPU kernel compilation
- **Grid Specification**: `lambda meta: (cdiv(N, BLOCK_SIZE),)` for 1D grids
- **tl.constexpr**: Compile-time constants enabling kernel specialization
- **Block Tiling**: Processing data in blocks for cache efficiency
- **Masking**: Safe handling of boundary conditions with `mask` parameter

---

## Quick Start

### For CPU Concurrency
1. Read: [Key Concepts](docs/source/cpu-concurrency/key_concepts.rst)
2. Study: [Threading Basics](docs/source/cpu-concurrency/threading_basics.rst)
3. Practice: See `/python` directory for practice problems

### For GPU Programming
1. Read: [GPU Fundamentals](docs/source/gpu-concepts/gpu-fundamentals.rst)
2. Study: [Vector Addition Tutorial](docs/source/gpu-tutorials/01-vector-add.rst)
3. Practice: Work through tutorials 1-5, then tackle 6+

### For Triton
1. Read: [Triton Concepts](docs/source/gpu-concepts/triton-concepts.rst)
2. Study: [Compiler Overview](docs/source/triton-compiler/01-overview.rst)
3. Dive Deep: [MLIR Concepts](docs/source/triton-compiler/05-mlir-concepts.rst)

---

## RST File Locations

All source documentation is in `docs/source/`:

```
docs/source/
├── cpu-concurrency/      (14 files on threading, asyncio, synchronization)
├── gpu-concepts/         (5 files on GPU architecture)
├── gpu-tutorials/        (10 step-by-step tutorials)
├── triton-compiler/      (5 files on Triton compilation)
├── learning-paths.rst
├── troubleshooting.rst
├── references.rst
└── index.rst            (Main documentation index)
```

---

## Contributing

To improve documentation:

1. Edit RST files in `docs/source/`
2. Build locally: `cd docs && make html`
3. Verify formatting in browser
4. Commit changes

For new documentation:
- Create `.rst` file in appropriate folder
- Add entry to `docs/source/index.rst` toctree
- Follow existing formatting and structure

---

## Additional Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [reStructuredText Primer](https://docutils.sourceforge.io/rst.html)
- [Triton Documentation](https://triton-lang.org/)
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/)

---

**Note**: For best viewing experience, build and view the HTML documentation locally. GitHub's web interface doesn't fully support RST formatting for tables, complex code blocks, and diagrams.
