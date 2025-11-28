Fast Concurrent Programming Guide
===================================

A comprehensive guide to concurrent and parallel programming, covering both CPU-based concurrency
(threading, asyncio, multiprocessing) and GPU-based parallelism (Triton, CUDA).

Welcome! This documentation provides in-depth tutorials and explanations of concurrent programming
techniques for modern Python applications, from multi-threaded CPU code to massively parallel GPU
kernels.

What is Concurrent Programming?
--------------------------------

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
---------------------

* **Python Developers**: Learn concurrent programming patterns
* **ML Engineers**: Optimize neural network operations on GPUs
* **Performance Engineers**: Achieve maximum utilization
* **Students**: Learn both CPU and GPU parallel programming

Prerequisites
-------------

**For CPU Concurrency:**

* Python programming experience
* Basic understanding of threads and processes
* Familiarity with Python standard library

**For GPU Programming:**

* Python and PyTorch knowledge
* Basic linear algebra
* Access to NVIDIA or AMD GPU (for GPU sections)

Getting Started
---------------

**New to concurrent programming?** Start with :doc:`cpu-concurrency/key_concepts`.

**Want to learn GPU programming?** Begin with :doc:`gpu-concepts/gpu-fundamentals`.

**Experienced with one area?** Jump directly to the section you need.

Documentation Structure
-----------------------

.. toctree::
   :maxdepth: 2
   :caption: CPU Concurrency

   cpu-concurrency/key_concepts
   cpu-concurrency/hardware_parallelism
   cpu-concurrency/threading_basics
   cpu-concurrency/asyncio_event_loop
   cpu-concurrency/asyncio_coroutine
   cpu-concurrency/asyncio_and_futures
   cpu-concurrency/asyncio_task_manipulation
   cpu-concurrency/concurrent_futures_pooling
   cpu-concurrency/queue_explained
   cpu-concurrency/queue_internal_mechanics
   cpu-concurrency/task_done_queue_explained
   cpu-concurrency/rlock_explained
   cpu-concurrency/semaphore_explained
   cpu-concurrency/patterns_problems_mapping

.. toctree::
   :maxdepth: 2
   :caption: GPU Concepts & Fundamentals

   gpu-concepts/gpu-fundamentals
   gpu-concepts/memory-hierarchy
   gpu-concepts/execution-model
   gpu-concepts/performance-optimization

.. toctree::
   :maxdepth: 2
   :caption: GPU Tutorials - Beginner Level

   gpu-tutorials/01-vector-add
   gpu-tutorials/02-fused-softmax

.. toctree::
   :maxdepth: 2
   :caption: GPU Tutorials - Intermediate Level

   gpu-tutorials/03-matrix-multiplication
   gpu-tutorials/04-low-memory-dropout

.. toctree::
   :maxdepth: 2
   :caption: GPU Tutorials - Advanced Level

   gpu-tutorials/05-layer-norm
   gpu-tutorials/06-fused-attention
   gpu-tutorials/07-extern-functions
   gpu-tutorials/08-grouped-gemm
   gpu-tutorials/09-persistent-matmul
   gpu-tutorials/10-block-scaled-matmul

.. toctree::
   :maxdepth: 2
   :caption: Triton Compiler & Internals

   triton-compiler/01-overview
   triton-compiler/02-jit-decorator
   triton-compiler/03-compilation-pipeline
   triton-compiler/04-cuda-comparison
   triton-compiler/05-mlir-concepts

.. toctree::
   :maxdepth: 2
   :caption: Resources & Support

   learning-paths
   troubleshooting
   references

Quick Navigation
----------------

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
==================

* :ref:`genindex`
* :ref:`search`
