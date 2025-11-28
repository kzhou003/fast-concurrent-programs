Fast Concurrent Programming Guide

A comprehensive guide to concurrent and parallel programming, covering both CPU-based concurrency
(threading, asyncio, multiprocessing) and GPU-based parallelism (Triton, CUDA).

Welcome! This documentation provides in-depth tutorials and explanations of concurrent programming
techniques for modern Python applications, from multi-threaded CPU code to massively parallel GPU
kernels.

What is Concurrent Programming?

Concurrent programming allows multiple tasks to make progress simultaneously, improving performance
and responsiveness. This guide covers two main approaches:

**CPU Concurrency** (Threads, Async, Processes)

* Threading for I/O-bound tasks
* Asyncio for async I/O operations
* Multiprocessing for CPU-bound parallel tasks
* Synchronization primitives (locks, semaphores, queues)

**GPU Parallelism** (Triton, CUDA)

* Massively parallel computation on GPUs
* Custom kernels for deep learning
* High-performance numerical computing
* Triton language for accessible GPU programming

Who This Guide Is For

* **Python Developers**: Learn concurrent programming patterns
* **ML Engineers**: Optimize neural network operations on GPUs
* **Performance Engineers**: Achieve maximum utilization
* **Students**: Learn both CPU and GPU parallel programming

Prerequisites

**For CPU Concurrency:**

* Python programming experience
* Basic understanding of threads and processes
* Familiarity with Python standard library

**For GPU Programming:**

* Python and PyTorch knowledge
* Basic linear algebra
* Access to NVIDIA or AMD GPU (for GPU sections)

Getting Started

**New to concurrent programming?** Start with :doc:`cpu-concurrency/key_concepts`.

**Want to learn GPU programming?** Begin with :doc:`gpu-concepts/gpu-fundamentals`.

**Experienced with one area?** Jump directly to the section you need.

Documentation Structure

.. toctree::
   :maxdepth: 0
   :caption: CPU Concurrency

   Key Concepts <cpu-concurrency/key_concepts>
   Hardware Parallelism <cpu-concurrency/hardware_parallelism>
   Threading Basics <cpu-concurrency/threading_basics>
   Asyncio Event Loop <cpu-concurrency/asyncio_event_loop>
   Asyncio Coroutine <cpu-concurrency/asyncio_coroutine>
   Asyncio and Futures <cpu-concurrency/asyncio_and_futures>
   Asyncio Task Manipulation <cpu-concurrency/asyncio_task_manipulation>
   Concurrent Futures Pooling <cpu-concurrency/concurrent_futures_pooling>
   Queue Explained <cpu-concurrency/queue_explained>
   Queue Internal Mechanics <cpu-concurrency/queue_internal_mechanics>
   Task Done Queue Explained <cpu-concurrency/task_done_queue_explained>
   RLock Explained <cpu-concurrency/rlock_explained>
   Semaphore Explained <cpu-concurrency/semaphore_explained>
   Patterns and Problems Mapping <cpu-concurrency/patterns_problems_mapping>

.. toctree::
   :maxdepth: 0
   :caption: GPU Concepts

   GPU Fundamentals <gpu-concepts/gpu-fundamentals>
   Memory Hierarchy <gpu-concepts/memory-hierarchy>
   Execution Model <gpu-concepts/execution-model>
   Performance Optimization <gpu-concepts/performance-optimization>
   Triton Concepts <gpu-concepts/triton-concepts>

.. toctree::
   :maxdepth: 0
   :caption: GPU Tutorials

   Vector Add <gpu-tutorials/01-vector-add>
   Fused Softmax <gpu-tutorials/02-fused-softmax>
   Matrix Multiplication <gpu-tutorials/03-matrix-multiplication>
   Low Memory Dropout <gpu-tutorials/04-low-memory-dropout>
   Layer Norm <gpu-tutorials/05-layer-norm>
   Fused Attention <gpu-tutorials/06-fused-attention>
   Extern Functions <gpu-tutorials/07-extern-functions>
   Grouped GEMM <gpu-tutorials/08-grouped-gemm>
   Persistent MatMul <gpu-tutorials/09-persistent-matmul>
   Block Scaled MatMul <gpu-tutorials/10-block-scaled-matmul>

.. toctree::
   :maxdepth: 0
   :caption: Triton Compiler

   Overview <triton-compiler/01-overview>
   JIT Decorator <triton-compiler/02-jit-decorator>
   Compilation Pipeline <triton-compiler/03-compilation-pipeline>
   CUDA Comparison <triton-compiler/04-cuda-comparison>
   MLIR Concepts <triton-compiler/05-mlir-concepts>

.. toctree::
   :maxdepth: 0
   :caption: Resources

   Learning Paths <learning-paths>
   Troubleshooting <troubleshooting>
   References <references>

Quick Navigation

**CPU Concurrency Quick Start:**

1. :doc:`cpu-concurrency/key_concepts` - Understand concurrency fundamentals
2. :doc:`cpu-concurrency/threading_basics` - Multi-threaded programming
3. :doc:`cpu-concurrency/asyncio_event_loop` - Async I/O programming

**GPU Programming Quick Start:**

1. :doc:`gpu-concepts/gpu-fundamentals` - GPU architecture basics
2. :doc:`gpu-tutorials/01-vector-add` - Your first GPU kernel
3. :doc:`gpu-tutorials/02-fused-softmax` - Kernel optimization

**Common Patterns:**

* **I/O-bound tasks** -> Asyncio or Threading
* **CPU-bound tasks** -> Multiprocessing
* **Massive parallelism** -> GPU programming
* **Deep learning** -> GPU kernels with Triton

Indices and Tables

* :ref:`genindex`
* :ref:`search`
