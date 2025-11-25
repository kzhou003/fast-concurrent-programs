# Fast Concurrent Programs - Complete Structure

## Overview
A comprehensive practice problem repository with **27 problems** across **6 domains**:

- **Concurrent Programming**: 15 problems (Python, C++, Go)
- **DL Modeling**: 5 problems (Neural network components)
- **DL LLM Systems**: 3 problems (Inference engines, agents)
- **Triton/CUDA**: 3 problems (GPU kernels)

## File Organization

```
fast-concurrent-programs/
├── README.md                           # Main guide and learning paths
├── STRUCTURE.md                        # This file
│
├── python/                             # Python concurrent programming
│   ├── 01_thread_safe_counter.py      # Locks, mutual exclusion
│   ├── 02_producer_consumer_queue.py  # Condition variables
│   ├── 03_thread_pool.py              # Thread management
│   ├── 04_dining_philosophers.py      # Deadlock avoidance
│   └── 05_async_web_scraper.py        # Async/await patterns
│
├── cpp/                                # C++ concurrent programming
│   ├── 01_thread_safe_counter.cpp     # Mutex + atomic operations
│   ├── 02_producer_consumer_queue.cpp # Condition variables
│   ├── 03_thread_pool.cpp             # Futures + promises
│   ├── 04_read_write_lock.cpp         # Advanced synchronization
│   └── 05_lock_free_queue.cpp         # Atomic CAS operations
│
├── go/                                 # Go concurrent programming
│   ├── 01_goroutine_counter.go        # Goroutines + mutex
│   ├── 02_pipeline_pattern.go         # Channel pipelines
│   ├── 03_worker_pool.go              # Worker goroutines
│   ├── 04_fan_out_fan_in.go           # Channel multiplexing
│   └── 05_rate_limiter.go             # Token bucket rate limiting
│
├── dl_modeling/                        # Deep learning components
│   ├── 01_linear_layer.py             # Fully-connected layers
│   ├── 02_conv2d_layer.py             # 2D convolution
│   ├── 03_attention_mechanism.py      # Scaled dot-product attention
│   ├── 04_batch_normalization.py      # Batch normalization
│   └── 05_lstm_cell.py                # LSTM recurrent cell
│
├── dl_llm_systems/                     # LLM inference systems
│   ├── 01_dynamic_batching.py         # Request batching for inference
│   ├── 02_kv_cache_manager.py         # Cache management & eviction
│   └── 03_agent_system.py             # ReAct agent framework
│
└── triton_cuda/                        # GPU programming
    ├── 01_matrix_multiply.py          # Optimized matrix multiplication
    ├── 02_softmax_kernel.py           # Softmax kernel implementation
    └── 03_flash_attention.py          # IO-aware attention
```

## Problem Categories

### Concurrent Programming (15 problems)

**Python (5 problems)**
- Basic synchronization → Advanced async patterns
- Focus: threading, condition variables, asyncio

**C++ (5 problems)**
- Mutex-based → Lock-free algorithms
- Focus: RAII, atomics, memory ordering

**Go (5 problems)**
- Goroutines → Complex channel patterns
- Focus: idiomatic channels, concurrency

### Deep Learning Modeling (5 problems)

Build ML components from scratch in NumPy:
1. Linear transformations
2. Convolutions
3. Attention mechanisms
4. Normalization layers
5. Recurrent networks

### DL LLM Systems (3 problems)

System design for LLM inference:
1. Dynamic batching strategies
2. Memory management (KV cache)
3. Agent framework with tools

### GPU Programming (3 problems)

Optimize kernels using Triton/CUDA concepts:
1. Matrix multiplication tiling
2. Reduction operations (softmax)
3. Memory-efficient attention (Flash)

## Difficulty Levels

| Level | Count | Examples |
|-------|-------|----------|
| Beginner | 5 | Counter, Linear layer, Goroutine counter |
| Intermediate | 17 | Queue, Conv2D, Attention, Batching, Softmax |
| Advanced | 5 | Lock-free, ReadWrite lock, LSTM, KV cache, Flash attention |

## Learning Paths

### Path 1: Concurrent Programming Master
1. Concurrent basics (Python/Go)
2. Advanced synchronization (C++)
3. Performance optimization
**Duration**: 2-3 weeks

### Path 2: Deep Learning Engineer
1. Core components (Linear, Conv)
2. Advanced layers (Attention, LSTM)
3. System design (Batching, Caching)
**Duration**: 3-4 weeks

### Path 3: GPU Programming Specialist
1. Basic kernels (MatMul, Softmax)
2. Optimized kernels (Flash attention)
3. Benchmarking and profiling
**Duration**: 2-3 weeks

### Path 4: Full Stack (All Domains)
- Start with concurrent programming
- Move to DL modeling
- Learn DL systems
- Master GPU programming
**Duration**: 8-10 weeks

## Key Features

Each problem includes:

✅ **Detailed problem description** - What to build and why
✅ **Requirements list** - Specific functionality needed
✅ **Performance notes** - Expected metrics and benchmarks
✅ **Boilerplate code** - TODO markers for implementation
✅ **Comprehensive tests** - 5-10 test cases per problem
✅ **Gradient checking** - For ML problems
✅ **Benchmarks** - Performance measurement code

## Language Coverage

| Language | Problems | Focus |
|----------|----------|-------|
| Python | 15 | Threading, asyncio, DL modeling, GPU concepts |
| C++ | 5 | Low-level concurrency, atomics, memory ordering |
| Go | 5 | Goroutines, channels, idiomatic concurrency |

## Topics Covered

### Concurrency Topics
- Mutual exclusion and locks
- Condition variables
- Atomic operations
- Memory ordering
- Lock-free data structures
- Async/await
- Goroutines and channels

### Deep Learning Topics
- Forward/backward passes
- Gradient computation
- Weight initialization
- Attention mechanisms
- Normalization techniques
- Sequence models

### Systems Topics
- Request batching
- Cache management
- Eviction policies
- Memory allocation
- Agent reasoning

### GPU Topics
- Block tiling
- Memory hierarchy
- Cache optimization
- Kernel fusion
- IO-aware algorithms

## Getting Started

1. **Read the README** - Understand the structure
2. **Pick a learning path** - Beginner, ML, GPU, or Full Stack
3. **Start with basics** - Try simple problems first
4. **Implement TODOs** - Fill in the skeleton code
5. **Run tests** - Verify correctness
6. **Benchmark** - Measure performance
7. **Iterate** - Optimize and refactor

## Expected Timeline

| Path | Beginner | Intermediate | Advanced |
|------|----------|--------------|----------|
| Concurrent | 1 week | 2 weeks | 1 week |
| DL Modeling | - | 2 weeks | 1 week |
| GPU | - | 1 week | 1 week |
| Total | 1 week | 5 weeks | 3 weeks |

## Prerequisites

- **Python path**: Python 3.8+, NumPy, understand of threading/async
- **C++ path**: C++17, understanding of POSIX threads, template basics
- **Go path**: Go 1.16+, understanding of goroutines
- **DL path**: NumPy, linear algebra, calculus basics
- **GPU path**: Understanding of GPU architecture (helpful but not required)

## Success Indicators

For each problem, you should be able to:

✅ Pass all provided tests
✅ Achieve within 10% of target performance
✅ Explain the design choices
✅ Handle edge cases correctly
✅ Use proper synchronization/optimization techniques

## Common Mistakes to Avoid

- **Concurrency**: Deadlocks, race conditions, goroutine leaks
- **DL**: Incorrect gradient computation, numerical instability
- **Systems**: Memory leaks, cache thrashing, unfair scheduling
- **GPU**: Uncoalesced memory access, synchronization overhead

## Resources & References

### Concurrency
- [The Little Book of Concurrency](https://www.cnblogs.com/jiayy/p/3246192.html)
- Language documentation (Python/C++/Go)
- [Lock-Free Programming Papers](https://www.1024cores.net/)

### Deep Learning
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [Backpropagation Tutorial](http://colah.github.io/posts/2015-08-Backprop/)
- Paper: "Attention Is All You Need"

### GPU Programming
- [Triton Documentation](https://triton-lang.org/)
- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- Paper: "Flash-2: Faster Causal Masked Attention with Rotary Embeddings"

### Systems
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [Flash Attention](https://arxiv.org/abs/2205.14135)
- [Orca: A Distributed Serving Engine](https://arxiv.org/abs/2211.07558)

---

**Total: 27 problems across 6 domains, designed for progressive skill building from basics to advanced systems.**

Happy learning! 🚀
