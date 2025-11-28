Learning Paths
==============

Choose a learning path based on your goals and experience level.

Path 1: Fast Track (Essentials)
--------------------------------

**Goal**: Get productive with Triton quickly

**Time**: 4-6 hours

**Prerequisites**: Basic Python, familiar with PyTorch

**Sequence**:

1. :doc:`gpu-concepts/gpu-fundamentals` (30 min)

   * Understand SPMD model
   * Learn GPU hierarchy

2. :doc:`gpu-tutorials/01-vector-add` (1 hour)

   * Write your first kernel
   * Understand parallelism
   * Learn memory patterns

3. :doc:`gpu-tutorials/02-fused-softmax` (1.5 hours)

   * Master kernel fusion
   * Understand SRAM vs DRAM
   * Learn reduction operations

4. :doc:`gpu-tutorials/03-matrix-multiplication` (2-3 hours)

   * Understand tiling
   * Learn auto-tuning
   * Use Tensor Cores

**Outcome**: You can write and optimize basic GPU kernels for common operations.

Path 2: Deep Understanding (Comprehensive)
------------------------------------------

**Goal**: Become a Triton/GPU programming expert

**Time**: 12-16 hours

**Prerequisites**: Path 1 or equivalent knowledge

**Sequence**:

1. **Foundations** (2 hours)

   * :doc:`gpu-concepts/gpu-fundamentals`
   * :doc:`gpu-concepts/memory-hierarchy`
   * :doc:`gpu-concepts/execution-model`

2. **Basic Kernels** (3 hours)

   * :doc:`gpu-tutorials/01-vector-add`
   * :doc:`gpu-tutorials/02-fused-softmax`

3. **Compute Optimization** (3 hours)

   * :doc:`gpu-tutorials/03-matrix-multiplication`
   * :doc:`gpu-concepts/performance-optimization`

4. **Advanced Memory Techniques** (2 hours)

   * :doc:`gpu-tutorials/04-low-memory-dropout`

5. **Training Loop Implementation** (3 hours)

   * :doc:`gpu-tutorials/05-layer-norm`

6. **State-of-the-Art** (3-4 hours)

   * :doc:`gpu-tutorials/06-fused-attention`

7. **Extensions** (1 hour)

   * :doc:`gpu-tutorials/07-extern-functions`

**Outcome**: You can implement complex, production-ready GPU kernels and optimize them for maximum performance.

Path 3: Transformer Focus (For LLM/NLP)
----------------------------------------

**Goal**: Optimize Transformers and attention mechanisms

**Time**: 8-10 hours

**Prerequisites**: Familiar with Transformers (BERT, GPT, etc.)

**Sequence**:

1. **GPU Basics** (1 hour)

   * :doc:`gpu-concepts/gpu-fundamentals`
   * :doc:`gpu-concepts/memory-hierarchy`

2. **Foundation Kernel** (1 hour)

   * :doc:`gpu-tutorials/01-vector-add`

3. **Attention Building Blocks** (2 hours)

   * :doc:`gpu-tutorials/02-fused-softmax`
   * Learn softmax optimization (key for attention)

4. **Normalization** (2 hours)

   * :doc:`gpu-tutorials/05-layer-norm`
   * Essential Transformer component

5. **Efficient Attention** (3-4 hours)

   * :doc:`gpu-tutorials/06-fused-attention`
   * Flash Attention for long sequences

6. **Memory Efficiency** (1-2 hours)

   * :doc:`gpu-tutorials/04-low-memory-dropout`
   * Techniques for large models

**Outcome**: You can optimize Transformer models, implement efficient attention, and handle long sequences.

Path 4: Computer Vision Focus
------------------------------

**Goal**: Optimize CNN and vision models

**Time**: 8-10 hours

**Sequence**:

1. **Fundamentals** (2 hours)

   * :doc:`gpu-concepts/gpu-fundamentals`
   * :doc:`gpu-concepts/memory-hierarchy`
   * :doc:`gpu-tutorials/01-vector-add`

2. **Compute-Heavy Operations** (3 hours)

   * :doc:`gpu-tutorials/03-matrix-multiplication`
   * Tiling techniques apply to convolutions

3. **Activation and Normalization** (2 hours)

   * :doc:`gpu-tutorials/02-fused-softmax`
   * :doc:`gpu-tutorials/05-layer-norm`

4. **Data Augmentation** (1 hour)

   * :doc:`gpu-tutorials/04-low-memory-dropout`

5. **Performance** (2 hours)

   * :doc:`gpu-concepts/performance-optimization`
   * :doc:`gpu-concepts/execution-model`

**Outcome**: Optimize convolutions, pooling, and other vision-specific operations.

Path 5: Performance Engineering
--------------------------------

**Goal**: Maximize GPU utilization and performance

**Time**: 10-12 hours

**Prerequisites**: Comfortable with GPU programming

**Sequence**:

1. **Core Concepts** (3 hours)

   * All documents in :doc:`gpu-concepts/gpu-fundamentals`
   * :doc:`gpu-concepts/memory-hierarchy`
   * :doc:`gpu-concepts/execution-model`
   * :doc:`gpu-concepts/performance-optimization`

2. **Practical Optimization** (4 hours)

   * :doc:`gpu-tutorials/02-fused-softmax` - Memory optimization
   * :doc:`gpu-tutorials/03-matrix-multiplication` - Compute optimization
   * :doc:`gpu-tutorials/06-fused-attention` - Advanced techniques

3. **Profiling and Tuning** (2 hours)

   * :doc:`troubleshooting`
   * Practice with real kernels
   * Use ``nsys`` and ``ncu``

4. **Case Studies** (2-3 hours)

   * Analyze and optimize existing kernels
   * Compare with PyTorch/cuBLAS
   * Implement variants

**Outcome**: Expert-level performance analysis and optimization skills.

By Topic
--------

If you want to learn specific topics:

Memory Optimization
~~~~~~~~~~~~~~~~~~~

1. :doc:`gpu-concepts/memory-hierarchy`
2. :doc:`gpu-tutorials/02-fused-softmax`
3. :doc:`gpu-tutorials/06-fused-attention`
4. :doc:`gpu-tutorials/04-low-memory-dropout`

Compute Optimization
~~~~~~~~~~~~~~~~~~~~

1. :doc:`gpu-concepts/execution-model`
2. :doc:`gpu-tutorials/03-matrix-multiplication`
3. :doc:`gpu-concepts/performance-optimization`

Backward Pass / Training
~~~~~~~~~~~~~~~~~~~~~~~~~

1. :doc:`gpu-tutorials/05-layer-norm`
2. :doc:`gpu-tutorials/06-fused-attention` (backward)

Advanced Techniques
~~~~~~~~~~~~~~~~~~~

1. :doc:`gpu-tutorials/06-fused-attention` - Online algorithms
2. :doc:`gpu-tutorials/04-low-memory-dropout` - Recomputation
3. :doc:`gpu-tutorials/07-extern-functions` - External libraries

Learning Tips
-------------

1. **Run the Code**

   Don't just read - execute examples::

       cd triton_cuda/triton_practice
       python 01-vector-add.py

2. **Modify and Experiment**

   * Change ``BLOCK_SIZE`` values
   * Try different input sizes
   * Add print statements
   * Break things intentionally!

3. **Profile Your Code**

   Use profiling tools::

       # NVIDIA
       nsys profile python script.py
       ncu --set full python script.py

       # AMD
       rocprof python script.py

4. **Compare with PyTorch**

   * Verify correctness
   * Measure speedup
   * Understand trade-offs

5. **Join the Community**

   * `Triton Discussions <https://github.com/openai/triton/discussions>`_
   * Share your kernels
   * Ask questions

Assessment Checkpoints
----------------------

After Path 1 (Fast Track)
~~~~~~~~~~~~~~~~~~~~~~~~~~

You should be able to:

☐ Explain SPMD execution model
☐ Write a simple element-wise kernel
☐ Understand memory coalescing
☐ Implement basic kernel fusion
☐ Use auto-tuning

After Path 2 (Comprehensive)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You should be able to:

☐ Implement forward and backward passes
☐ Optimize for both memory and compute
☐ Use Tensor Cores effectively
☐ Write persistent kernels
☐ Achieve 80%+ of PyTorch performance

After Path 3 (Transformer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You should be able to:

☐ Implement efficient attention mechanisms
☐ Handle long sequences (16K+ tokens)
☐ Optimize layer normalization
☐ Fuse operations in Transformers
☐ Understand Flash Attention algorithm

Next Steps After Completing a Path
-----------------------------------

1. **Build a Project**

   * Optimize your own model
   * Implement a research paper
   * Contribute to open source

2. **Advanced Topics**

   * Multi-GPU kernels
   * Quantization (INT8, FP8)
   * Sparse operations
   * Custom backward passes

3. **Contribute**

   * Share your kernels
   * Write tutorials
   * Help others in community

Resources for Continued Learning
---------------------------------

* :doc:`references` - Papers and documentation
* :doc:`troubleshooting` - Common issues and solutions
* `Triton GitHub <https://github.com/openai/triton>`_ - Latest updates
* `CUDA Programming Guide <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>`_ - Deep dive

Choose Your Path
----------------

Ready to start? Pick the path that matches your goals:

* **Quick start?** → :doc:`gpu-tutorials/01-vector-add`
* **Deep learning?** → Path 2 (Comprehensive)
* **Transformers?** → Path 3 (Transformer Focus)
* **Performance?** → Path 5 (Performance Engineering)

Happy learning! 🚀
