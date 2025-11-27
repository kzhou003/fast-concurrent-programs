Triton GPU Programming Guide
============================

A comprehensive guide to GPU programming with Triton, from fundamentals to advanced optimization techniques.

Welcome to the Triton GPU Programming Guide! This documentation provides in-depth tutorials
and explanations of GPU programming concepts using Triton, a language and compiler for writing
highly efficient custom Deep Learning primitives.

What is Triton?
---------------

Triton is a language and compiler for parallel programming that aims to provide a Python-based
programming surface that is both more productive and more portable than CUDA, while still
achieving comparable performance on modern GPU hardware.

Key advantages:

* **Easy to learn**: Python-like syntax, high-level abstractions
* **High performance**: Achieves 90-95% of hand-tuned CUDA performance
* **Portable**: Works on NVIDIA and AMD GPUs
* **Productive**: Write complex kernels in 1/3 the code of CUDA

Who This Guide Is For
---------------------

* **ML Engineers**: Optimize neural network operations
* **Researchers**: Implement custom operations for novel architectures
* **Performance Engineers**: Achieve maximum GPU utilization
* **Students**: Learn GPU programming concepts

Prerequisites
-------------

* Python programming experience
* Basic understanding of linear algebra
* Familiarity with PyTorch (helpful but not required)
* Access to NVIDIA or AMD GPU

Getting Started
---------------

If you're new to GPU programming, start with :doc:`concepts/gpu-fundamentals`.
Then follow the tutorial sequence beginning with :doc:`tutorials/01-vector-add`.

For experienced GPU programmers, you might jump directly to advanced topics
like :doc:`tutorials/06-fused-attention`.

.. toctree::
   :maxdepth: 2
   :caption: Core Concepts

   concepts/gpu-fundamentals
   concepts/memory-hierarchy
   concepts/execution-model
   concepts/performance-optimization

.. toctree::
   :maxdepth: 2
   :caption: Beginner Tutorials

   tutorials/01-vector-add
   tutorials/02-fused-softmax

.. toctree::
   :maxdepth: 2
   :caption: Intermediate Tutorials

   tutorials/03-matrix-multiplication
   tutorials/04-low-memory-dropout

.. toctree::
   :maxdepth: 2
   :caption: Advanced Tutorials

   tutorials/05-layer-norm
   tutorials/06-fused-attention
   tutorials/07-extern-functions

.. toctree::
   :maxdepth: 2
   :caption: Additional Resources

   learning-paths
   troubleshooting
   references

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
