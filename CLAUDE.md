# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Fast Concurrent Programs is a comprehensive practice problem repository with 27+ problems across concurrent programming (Python/C++/Go), deep learning modeling, LLM systems, and GPU programming (Triton/CUDA). Each problem contains TODO markers for learners to implement.

## Repository Structure

```
fast-concurrent-programs/
├── python/              # Python concurrent programming (5 problems)
├── cpp/                 # C++ concurrent programming (5 problems)
├── go/                  # Go concurrent programming (5 problems)
├── basics/              # Educational materials on concurrency concepts
├── dl_modeling/         # Deep learning components from scratch (5 problems)
├── dl_llm_systems/      # LLM inference systems (3 problems)
├── triton_cuda/         # GPU kernel programming (3+ problems)
└── docs/                # Sphinx documentation for concepts
```

## Running and Testing

### Python Problems
Each Python problem file is self-contained with test cases at the bottom:

```bash
# Run a single problem
python python/01_thread_safe_counter.py

# Run all Python concurrent problems
for f in python/*.py; do python "$f"; done

# Run DL modeling problems
python dl_modeling/01_linear_layer.py
python dl_llm_systems/01_dynamic_batching.py
python triton_cuda/01_matrix_multiply.py
```

Tests are built-in using `if __name__ == "__main__"` blocks. No external test runner is needed.

### C++ Problems
C++ problems require compilation with C++17 and threading support:

```bash
cd cpp
g++ -std=c++17 -pthread -o counter 01_thread_safe_counter.cpp && ./counter
g++ -std=c++17 -pthread -o queue 02_producer_consumer_queue.cpp && ./queue
# Lock-free queue may need additional memory ordering considerations
g++ -std=c++17 -pthread -O2 -o lock_free 05_lock_free_queue.cpp && ./lock_free
```

### Go Problems
Go problems can be run directly:

```bash
go run go/01_goroutine_counter.go
go run go/02_pipeline_pattern.go
```

## Problem Structure and Requirements

Each problem follows a consistent format:

1. **Problem Description** - At the top in docstring/comment block, explains what to build
2. **TODO Markers** - Lines marked with `# TODO:` or `// TODO:` indicate where to implement
3. **Test Cases** - Built-in tests verify correctness (marked with `test_` prefix in Python, main functions in C++/Go)
4. **Performance Expectations** - Problems specify throughput/latency targets in the comments

### Key Patterns to Follow

- **Python**: Use `threading.Lock`, `threading.Condition`, `asyncio`, or `queue.Queue` depending on the problem
- **C++**: Prefer RAII patterns with `std::lock_guard` or `std::unique_lock`; use `std::atomic` for lock-free variants
- **Go**: Use channels for communication; leverage goroutines; follow idiomatic Go patterns

## Architecture Notes

### Concurrency Problems (Python/C++/Go)
- **Beginner**: Basic synchronization (locks, mutexes)
- **Intermediate**: Condition variables, producer-consumer patterns, thread pools
- **Advanced**: Lock-free structures, reader-writer locks, complex channel patterns

### Deep Learning Problems
- **DL Modeling**: Neural network components implemented from scratch in NumPy (forward/backward passes)
- **DL LLM Systems**: Higher-level systems for inference (batching, caching, memory management)
- **Triton/CUDA**: GPU kernel optimization focusing on memory hierarchy and blocking strategies

### Documentation
- `docs/` contains Sphinx-built documentation
- `docs/basics/` has conceptual materials explaining threading, asyncio, queues, etc.
- `docs/triton/` has GPU programming tutorials
- Build docs with `cd docs && make html`

## Common Development Patterns

### Testing and Validation
- All problems include comprehensive test cases within the problem file
- Tests check correctness, edge cases, and performance (where applicable)
- Gradient checking is included in ML problems for numerical correctness
- Benchmarks are provided to measure throughput/latency

### Implementation Approach
1. Read the problem description thoroughly (first comment block)
2. Understand the test cases to know what's expected
3. Implement the TODO sections
4. Run the file directly to test: `python file.py` or `go run file.go` or compile and run C++
5. Verify all tests pass
6. Check performance against specified targets

## Important Implementation Considerations

### Thread Safety (Concurrency Problems)
- Always protect shared state with appropriate synchronization
- Avoid deadlocks through consistent lock ordering
- Be careful with condition variable wait/notify patterns
- In C++, prefer `std::lock_guard` over manual unlock for exception safety

### Numerical Stability (ML Problems)
- Use techniques like log-softmax to prevent overflow
- Implement batch normalization with epsilon values
- Check gradients numerically for correctness
- Handle edge cases in normalization/attention (zero denominators, etc.)

### GPU Programming (Triton)
- Focus on memory access patterns and blocking strategies
- Understand cache hierarchies and coalesced access
- Use online computation techniques for reductions
- Benchmark memory throughput vs arithmetic intensity

## Linting and Code Quality

The repository doesn't enforce strict linting; focus on:
- Using appropriate language idioms (Python's threading module, C++'s RAII, Go's channels)
- Following the patterns already in the codebase
- Clear variable names that indicate purpose (e.g., `lock`, `cv` for condition variable)
- Comments where algorithm complexity isn't self-evident

## Quick Reference: Problem Categories

| Category | Language | Difficulty | Count |
|----------|----------|------------|-------|
| Basic Concurrency | Py/C++/Go | Beginner | 3 each |
| Advanced Concurrency | Py/C++/Go | Intermediate/Advanced | 2 each |
| DL Components | Python | Intermediate | 5 |
| LLM Systems | Python | Intermediate/Advanced | 3 |
| GPU Kernels | Python (Triton) | Intermediate/Advanced | 3+ |

## Documentation Build

The repository uses Sphinx for generating documentation:

```bash
cd docs
pip install -r requirements.txt
make html        # Build HTML documentation
make clean        # Clean build artifacts
```

The HTML output is generated in `docs/_build/html/`.
