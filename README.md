# Fast Concurrent Programs - Practice Problems

This repository contains a curated set of concurrent programming problems designed to help you master concurrent patterns across Python, C++, and Go, as well as GPU programming with Triton/CUDA.

## Documentation

Comprehensive guides and tutorials on concurrent and GPU programming. This repository includes documentation on concurrent and GPU programming in multiple formats.

### Quick Navigation

#### HTML Documentation (Recommended)
The best viewing experience is via the built HTML documentation:

```bash
cd docs
make html
# Then open build/html/index.html in your browser
```

This provides full navigation, proper formatting for tables, code blocks, and diagrams.

---

### Documentation Structure

#### CPU Concurrency Guides
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

#### GPU Programming Guides
Master GPU architecture, optimization techniques, and Triton programming.

| Topic | Focus | Level |
|-------|-------|-------|
| [GPU Fundamentals](docs/source/gpu-concepts/gpu-fundamentals.rst) | GPU architecture, threads, warps, blocks | Beginner |
| [Memory Hierarchy](docs/source/gpu-concepts/memory-hierarchy.rst) | Global, shared, local memory; cache; bandwidth | Intermediate |
| [Execution Model](docs/source/gpu-concepts/execution-model.rst) | Thread execution, synchronization, atomics | Advanced |
| [Performance Optimization](docs/source/gpu-concepts/performance-optimization.rst) | Memory access patterns, occupancy, roofline | Advanced |
| [Triton Concepts](docs/source/gpu-concepts/triton-concepts.rst) | Kernel launching, grid specification, constexpr | Intermediate |

#### GPU Tutorials (Step-by-Step Examples)
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

#### Triton Compiler Deep Dive
Understanding Triton's compilation pipeline and optimization.

| Topic | Focus | Level |
|-------|-------|-------|
| [Compiler Overview](docs/source/triton-compiler/01-overview.rst) | Pipeline stages, IR representations | Intermediate |
| [JIT Decorator](docs/source/triton-compiler/02-jit-decorator.rst) | Specialization, caching, kernel variants | Advanced |
| [Compilation Pipeline](docs/source/triton-compiler/03-compilation-pipeline.rst) | Python→Triton IR→MLIR→LLVM→PTX→Cubin | Advanced |
| [CUDA Comparison](docs/source/triton-compiler/04-cuda-comparison.rst) | Triton vs CUDA: code, compilation, optimization | Intermediate |
| [MLIR Concepts](docs/source/triton-compiler/05-mlir-concepts.rst) | MLIR dialects, transformations, lowering | Advanced |

#### Learning Resources
Structured learning paths and reference materials.

| Resource | Content |
|----------|---------|
| [Learning Paths](docs/source/learning-paths.rst) | Beginner → Intermediate → Advanced progressions |
| [Troubleshooting](docs/source/troubleshooting.rst) | Common issues, debugging, performance problems |
| [References](docs/source/references.rst) | External links, official documentation, papers |

---

### Building Documentation Locally

#### Prerequisites
```bash
pip install sphinx sphinx-rtd-theme
```

#### Build HTML Documentation
```bash
cd docs
make html
open build/html/index.html  # macOS
# or
xdg-open build/html/index.html  # Linux
# or
start build/html/index.html  # Windows
```

#### Build Other Formats
```bash
# PDF (requires LaTeX)
make latexpdf

# Man pages
make man

# EPUB
make epub
```

---

### Key Concepts Quick Reference

#### Concurrent Programming Patterns
- **Mutual Exclusion (Locks)**: Prevent concurrent access to shared resources
- **Condition Variables**: Wait/notify synchronization between threads
- **Producer-Consumer**: Data flow between threads with bounded queues
- **Thread Pools**: Reusable worker threads for task execution
- **Async/Await**: Efficient I/O-bound concurrency without explicit threading

#### GPU Programming Fundamentals
- **SPMD Model**: Single Program, Multiple Data execution
- **Thread Organization**: Threads → Warps → Blocks → Grids
- **Memory Hierarchy**: Global → Shared → Local memory with different speeds
- **Synchronization**: `__syncthreads()` for block-level, atomic ops for device-level
- **Occupancy**: Ratio of active warps to maximum possible warps

#### Triton Concepts
- **@triton.jit**: JIT decorator for GPU kernel compilation
- **Grid Specification**: `lambda meta: (cdiv(N, BLOCK_SIZE),)` for 1D grids
- **tl.constexpr**: Compile-time constants enabling kernel specialization
- **Block Tiling**: Processing data in blocks for cache efficiency
- **Masking**: Safe handling of boundary conditions with `mask` parameter

---

### Quick Start

#### For CPU Concurrency
1. Read: [Key Concepts](docs/source/cpu-concurrency/key_concepts.rst)
2. Study: [Threading Basics](docs/source/cpu-concurrency/threading_basics.rst)
3. Practice: See `/python` directory for practice problems

#### For GPU Programming
1. Read: [GPU Fundamentals](docs/source/gpu-concepts/gpu-fundamentals.rst)
2. Study: [Vector Addition Tutorial](docs/source/gpu-tutorials/01-vector-add.rst)
3. Practice: Work through tutorials 1-5, then tackle 6+

#### For Triton
1. Read: [Triton Concepts](docs/source/gpu-concepts/triton-concepts.rst)
2. Study: [Compiler Overview](docs/source/triton-compiler/01-overview.rst)
3. Dive Deep: [MLIR Concepts](docs/source/triton-compiler/05-mlir-concepts.rst)

---

### Documentation File Locations

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

Markdown versions for GitHub viewing in `docs/markdown/`:

```
docs/markdown/
└── gpu-concepts/        (5 Markdown files for GitHub web viewing)
```

---

### Contributing to Documentation

To improve documentation:

1. Edit RST files in `docs/source/`
2. Build locally: `cd docs && make html`
3. Verify formatting in browser
4. Commit changes

For new documentation:
- Create `.rst` file in appropriate folder
- Add entry to `docs/source/index.rst` toctree
- Follow existing formatting and structure
- Create corresponding `.md` file for GitHub viewing

---

### Additional Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [reStructuredText Primer](https://docutils.sourceforge.io/rst.html)
- [Triton Documentation](https://triton-lang.org/)
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/)


## Key Concepts by Language

### Python
- `threading.Lock` / `threading.RLock` - Mutual exclusion
- `threading.Condition` - Wait/notify synchronization
- `threading.Event` - Simple signaling
- `queue.Queue` - Thread-safe queue
- `asyncio` - Asynchronous programming
- `threading.Semaphore` - Resource counting

### C++
- `std::mutex` - Mutual exclusion
- `std::condition_variable` - Wait/notify
- `std::lock_guard` / `std::unique_lock` - RAII wrappers
- `std::atomic` - Atomic operations
- `std::future` / `std::promise` - Async results
- Memory ordering: `acquire`, `release`, `relaxed`, `seq_cst`

### Go
- `sync.Mutex` - Mutual exclusion
- `sync.RWMutex` - Reader-writer lock
- `sync.WaitGroup` - Coordination
- Channels - Communication and synchronization
- `select` statement - Channel multiplexing
- `goroutines` - Lightweight concurrency


## Structure of problems

### Python (`/python`)
Introduction to concurrent programming concepts using Python's threading and asyncio modules.

1. **01_thread_safe_counter.py** - Thread-safe counter with mutex
   - Learn: Locks, race conditions, synchronization primitives
   - Difficulty: Beginner
   - Key concepts: `threading.Lock`, mutual exclusion

2. **02_producer_consumer_queue.py** - Thread-safe bounded queue
   - Learn: Condition variables, blocking operations, signaling
   - Difficulty: Intermediate
   - Key concepts: `threading.Condition`, wait/notify patterns

3. **03_thread_pool.py** - Thread pool executor
   - Learn: Thread lifecycle, work distribution, futures
   - Difficulty: Intermediate
   - Key concepts: Worker threads, task queues, graceful shutdown

4. **04_dining_philosophers.py** - Classic synchronization problem
   - Learn: Deadlock avoidance, fair scheduling, resource allocation
   - Difficulty: Intermediate
   - Key concepts: Deadlock prevention, fairness

5. **05_async_web_scraper.py** - Asynchronous concurrent fetching
   - Learn: Async/await, coroutines, rate limiting
   - Difficulty: Intermediate
   - Key concepts: `asyncio`, event loops, semaphores

### C++ (`/cpp`)
Low-level concurrent programming with fine-grained control over synchronization.

1. **01_thread_safe_counter.cpp** - Atomic counter with synchronization
   - Learn: `std::mutex`, `std::atomic`, RAII patterns
   - Difficulty: Beginner
   - Key concepts: Lock guards, atomic operations

2. **02_producer_consumer_queue.cpp** - Template-based queue with condition variables
   - Learn: `std::condition_variable`, template metaprogramming
   - Difficulty: Intermediate
   - Key concepts: Wait/notify, timeout handling

3. **03_thread_pool.cpp** - Thread pool with futures
   - Learn: `std::future`, `std::promise`, thread management
   - Difficulty: Intermediate
   - Key concepts: Async task execution, result handling

4. **04_read_write_lock.cpp** - Reader-writer lock implementation
   - Learn: Multiple synchronized access patterns
   - Difficulty: Advanced
   - Key concepts: Shared/exclusive locks, writer priority

5. **05_lock_free_queue.cpp** - Lock-free queue using atomics
   - Learn: CAS operations, memory ordering, lock-free data structures
   - Difficulty: Advanced
   - Key concepts: `std::atomic`, compare-and-swap, ABA problem

### Go (`/go`)
Idiomatic concurrent programming using goroutines and channels.

1. **01_goroutine_counter.go** - Counter with mutex
   - Learn: `sync.Mutex`, goroutines, basic synchronization
   - Difficulty: Beginner
   - Key concepts: Goroutines, mutex patterns

2. **02_pipeline_pattern.go** - Multi-stage data pipeline
   - Learn: Channel communication, goroutine pipelines
   - Difficulty: Intermediate
   - Key concepts: Channels as pipes, concurrent stages

3. **03_worker_pool.go** - Worker pool pattern
   - Learn: Work distribution, goroutine management
   - Difficulty: Intermediate
   - Key concepts: Worker goroutines, task queues

4. **04_fan_out_fan_in.go** - Distributing and aggregating work
   - Learn: Fan-out/fan-in patterns, channel multiplexing
   - Difficulty: Intermediate
   - Key concepts: Work distribution, result aggregation

5. **05_rate_limiter.go** - Token bucket rate limiting
   - Learn: Time-based coordination, token patterns
   - Difficulty: Intermediate
   - Key concepts: Rate control, channels with time

### DL Modeling (`/dl_modeling`)
Implementation of core deep learning components from scratch in Python/NumPy.

1. **01_linear_layer.py** - Fully-connected layer
   - Learn: Forward/backward passes, weight initialization, gradient computation
   - Difficulty: Beginner
   - Key concepts: Matrix operations, backpropagation, numerical gradients

2. **02_conv2d_layer.py** - 2D Convolution layer
   - Learn: Convolution operation, padding/stride, gradient computation
   - Difficulty: Intermediate
   - Key concepts: Tensor operations, spatial convolutions, im2col

3. **03_attention_mechanism.py** - Scaled dot-product attention
   - Learn: Softmax, multi-head patterns, causal masking
   - Difficulty: Intermediate
   - Key concepts: Attention weights, temperature scaling, numerical stability

4. **04_batch_normalization.py** - Batch normalization
   - Learn: Training vs eval modes, running statistics, layer optimization
   - Difficulty: Intermediate
   - Key concepts: Internal covariate shift, momentum, affine transformations

5. **05_lstm_cell.py** - LSTM recurrent cell
   - Learn: Gating mechanisms, hidden state tracking, sequence processing
   - Difficulty: Intermediate
   - Key concepts: Cell state, gradient flow, vanishing gradients

### DL LLM Systems (`/dl_llm_systems`)
Systems-level problems for building LLM inference engines and agent systems.

1. **01_dynamic_batching.py** - Request batching for inference
   - Learn: Queue management, batch formation strategies, latency optimization
   - Difficulty: Intermediate
   - Key concepts: FCFS/SJF scheduling, padding efficiency, throughput optimization

2. **02_kv_cache_manager.py** - KV cache management
   - Learn: Memory allocation, eviction policies, sequence management
   - Difficulty: Intermediate
   - Key concepts: LRU/LFU eviction, cache fragmentation, memory limits

3. **03_agent_system.py** - Agent framework with tools
   - Learn: ReAct pattern, tool calling, conversation memory
   - Difficulty: Intermediate
   - Key concepts: Multi-step reasoning, state management, error handling

### Triton/CUDA (`/triton_cuda`)
GPU kernel programming using Triton and CUDA concepts.

1. **01_matrix_multiply.py** - Optimized matrix multiplication
   - Learn: Block tiling, memory hierarchy, GPU optimization
   - Difficulty: Intermediate
   - Key concepts: Blocking strategy, cache locality, coalesced memory access

2. **02_softmax_kernel.py** - Softmax kernel implementation
   - Learn: Row-wise reductions, numerical stability, online computation
   - Difficulty: Intermediate
   - Key concepts: Block operations, final normalization, attention kernels

3. **03_flash_attention.py** - IO-aware attention implementation
   - Learn: IO-optimal algorithms, tiling strategy, online softmax
   - Difficulty: Advanced
   - Key concepts: Block-wise computation, running statistics, memory efficiency

## Learning Path

### Beginner Path (Concurrent Programming)
1. Start with Python basics:
   - `python/01_thread_safe_counter.py`
   - `python/02_producer_consumer_queue.py`

2. Move to DL Modeling basics:
   - `dl_modeling/01_linear_layer.py`

3. Try Go basics:
   - `go/01_goroutine_counter.go`
   - `go/02_pipeline_pattern.go`

### Intermediate Path (Concurrent & DL)
1. Expand concurrent programming:
   - `python/03_thread_pool.py` & `cpp/03_thread_pool.cpp`
   - `go/03_worker_pool.go` & `go/04_fan_out_fan_in.go`

2. Expand DL Modeling:
   - `dl_modeling/02_conv2d_layer.py`
   - `dl_modeling/03_attention_mechanism.py`
   - `dl_modeling/04_batch_normalization.py`

3. Learn GPU programming:
   - `triton_cuda/01_matrix_multiply.py`
   - `triton_cuda/02_softmax_kernel.py`

### Advanced Path (Systems & GPU)
1. Advanced concurrent:
   - `cpp/04_read_write_lock.cpp`
   - `cpp/05_lock_free_queue.cpp`
   - `python/04_dining_philosophers.py`

2. Advanced DL systems:
   - `dl_modeling/05_lstm_cell.py`
   - `dl_llm_systems/01_dynamic_batching.py`
   - `dl_llm_systems/02_kv_cache_manager.py`
   - `dl_llm_systems/03_agent_system.py`

3. Advanced GPU:
   - `triton_cuda/03_flash_attention.py`

## How to Use

Each problem file contains:
- **Problem Description** - Detailed requirements and learning objectives
- **Boilerplate Code** - Skeleton implementation with TODO comments
- **Test Cases** - Comprehensive tests to verify correctness
- **Performance Notes** - Expected performance characteristics

### Getting Started

1. Read the problem description at the top of the file
2. Understand the requirements and test cases
3. Implement the TODOs (marked with `// TODO:` or `# TODO:`)
4. Run the tests to verify correctness
5. Benchmark your implementation

### Running Tests

**Python (Concurrent):**
```bash
python python/01_thread_safe_counter.py
```

**Python (DL Modeling):**
```bash
python dl_modeling/01_linear_layer.py
```

**Python (DL Systems):**
```bash
python dl_llm_systems/01_dynamic_batching.py
```

**Python (GPU):**
```bash
python triton_cuda/01_matrix_multiply.py
```

**C++:**
```bash
cd cpp
g++ -std=c++17 -pthread -o counter 01_thread_safe_counter.cpp
./counter
```

**Go:**
```bash
go run go/01_goroutine_counter.go
```

## Performance Expectations

| Problem | Language | Expected Throughput |
|---------|----------|-------------------|
| Thread-safe Counter | Python | 10K-50K ops/sec |
| Thread-safe Counter | C++ | 1M-10M ops/sec |
| Producer-Consumer | Python | 1K-5K items/sec |
| Producer-Consumer | C++ | 100K-500K items/sec |
| Thread Pool | Python | 100-1K tasks/sec |
| Thread Pool | C++ | 10K-100K tasks/sec |
| Rate Limiter | Go | Configurable (accurate) |

## Tips for Success

1. **Start Simple** - Get basic tests passing before optimizing
2. **Understand the Problem** - Read comments carefully
3. **Test Thoroughly** - Run all tests, including edge cases
4. **Measure Performance** - Use benchmarks to verify optimization
5. **Avoid Common Pitfalls**:
   - Deadlocks (circular wait)
   - Race conditions (unsynchronized access)
   - Goroutine leaks (unreachable goroutines)
   - Memory leaks (unclosed channels/resources)

## Common Challenges

### Deadlock
- Caused by circular wait for resources
- Prevention: consistent lock ordering, timeout-based acquisition

### Race Conditions
- Unsynchronized access to shared data
- Solution: proper synchronization primitives

### Starvation
- One thread continuously blocked while others proceed
- Solution: fair scheduling, writer/reader priority

### Performance
- Lock contention reduces throughput
- Solution: reduce critical section, use lock-free structures

## References

### Python
- [threading documentation](https://docs.python.org/3/library/threading.html)
- [asyncio documentation](https://docs.python.org/3/library/asyncio.html)

### C++
- [std::thread documentation](https://en.cppreference.com/w/cpp/thread/thread)
- [std::atomic documentation](https://en.cppreference.com/w/cpp/atomic/atomic)
- [Memory ordering guide](https://en.cppreference.com/w/cpp/atomic/memory_order)

### Go
- [sync package](https://golang.org/pkg/sync/)
- [Channels](https://golang.org/ref/spec#Channel_types)
- [Effective Go - Concurrency](https://golang.org/doc/effective_go#concurrency)

### GPU Programming
- [Triton Documentation](https://triton-lang.org/)
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/)

---

Good luck with your concurrent programming journey! Remember, mastering concurrency takes time and practice. Start with simple problems and gradually tackle more complex ones.
