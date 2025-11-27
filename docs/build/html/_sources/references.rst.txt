References and Resources
========================

Essential resources for learning more about GPU programming, Triton, and related topics.

Official Documentation
-----------------------

Triton
~~~~~~

* `Triton Documentation <https://triton-lang.org/>`_ - Official docs
* `Triton GitHub Repository <https://github.com/openai/triton>`_ - Source code and examples
* `Triton Discussions <https://github.com/openai/triton/discussions>`_ - Community Q&A
* `Triton Issues <https://github.com/openai/triton/issues>`_ - Bug reports and feature requests

CUDA
~~~~

* `CUDA C Programming Guide <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>`_ - Comprehensive CUDA reference
* `CUDA C Best Practices Guide <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/>`_ - Optimization techniques
* `NVIDIA Developer Blog <https://developer.nvidia.com/blog>`_ - Latest updates and tutorials
* `cuBLAS Documentation <https://docs.nvidia.com/cuda/cublas/>`_ - Matrix operations library
* `cuDNN Documentation <https://docs.nvidia.com/deeplearning/cudnn/>`_ - Deep learning primitives

ROCm (AMD)
~~~~~~~~~~

* `ROCm Documentation <https://rocmdocs.amd.com/>`_ - AMD GPU programming
* `HIP Programming Guide <https://rocmdocs.amd.com/projects/HIP/>`_ - CUDA-like programming for AMD
* `AMD Developer Resources <https://developer.amd.com/>`_

PyTorch
~~~~~~~

* `PyTorch Documentation <https://pytorch.org/docs/stable/>`_ - Deep learning framework
* `PyTorch Custom C++/CUDA Extensions <https://pytorch.org/tutorials/advanced/cpp_extension.html>`_
* `PyTorch Internals <https://pytorch.org/blog/>`_

Research Papers
---------------

Flash Attention
~~~~~~~~~~~~~~~

**Flash Attention** - Dao et al., 2022

   Seminal paper on O(N) memory attention algorithm

   * Paper: `arXiv:2205.14135 <https://arxiv.org/abs/2205.14135>`_
   * Key contribution: Online softmax algorithm
   * Impact: Enables 16K+ token context windows

**Flash Attention-2** - Dao, 2023

   Improved version with better parallelism

   * Paper: `arXiv:2307.08691 <https://arxiv.org/abs/2307.08691>`_
   * Improvements: 2x faster than original
   * Techniques: Warp specialization, better scheduling

Normalization
~~~~~~~~~~~~~

**Layer Normalization** - Ba et al., 2016

   Foundation for Transformer architectures

   * Paper: `arXiv:1607.06450 <https://arxiv.org/abs/1607.06450>`_
   * Key idea: Normalize across features, not batch
   * Usage: BERT, GPT, all modern Transformers

**RMSNorm** - Zhang & Sennrich, 2019

   Simplified layer normalization

   * Paper: `arXiv:1910.07467 <https://arxiv.org/abs/1910.07467>`_
   * Simplification: No mean subtraction
   * Used in: LLaMA, GPT-NeoX

Optimization Techniques
~~~~~~~~~~~~~~~~~~~~~~~

**Automatic Differentiation** - Baydin et al., 2018

   Survey of autodiff techniques

   * Paper: `arXiv:1502.05767 <https://arxiv.org/abs/1502.05767>`_
   * Covers forward and reverse mode
   * Essential for understanding backward passes

**Mixed Precision Training** - Micikevicius et al., 2018

   Training with FP16 for speedup

   * Paper: `arXiv:1710.03740 <https://arxiv.org/abs/1710.03740>`_
   * Techniques: Loss scaling, master weights
   * Impact: 2-3x training speedup

Triton
~~~~~~

**Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations** - Tillet et al., 2019

   Original Triton paper

   * Paper: `ACM MAPL 2019 <https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf>`_
   * Key insight: Block-level programming model
   * Impact: Made GPU programming accessible

Books
-----

GPU Programming
~~~~~~~~~~~~~~~

**Programming Massively Parallel Processors** - Kirk & Hwu

   * Comprehensive introduction to GPU programming
   * Covers CUDA fundamentals through advanced topics
   * Excellent for understanding hardware

**CUDA by Example** - Sanders & Kandrot

   * Practical, example-driven approach
   * Good for beginners
   * Covers basic to intermediate topics

**Professional CUDA C Programming** - Cheng et al.

   * Advanced CUDA techniques
   * Performance optimization
   * Real-world case studies

Deep Learning
~~~~~~~~~~~~~

**Deep Learning** - Goodfellow, Bengio & Courville

   * Comprehensive ML theory
   * Mathematical foundations
   * Available free online

**Dive into Deep Learning** - Zhang et al.

   * Interactive textbook
   * Code examples with PyTorch/MXNet
   * `Available online <https://d2l.ai/>`_

Tools and Profilers
-------------------

NVIDIA Tools
~~~~~~~~~~~~

**Nsight Systems**

   * System-wide profiling
   * Timeline visualization
   * CPU-GPU interaction

   Installation::

       # Included with CUDA toolkit
       nsys profile -o output python script.py
       nsys-ui output.qdrep  # View results

**Nsight Compute**

   * Kernel-level profiling
   * Detailed metrics (memory, compute, occupancy)
   * Optimization suggestions

   Usage::

       ncu --set full -o output python script.py
       ncu-ui output.ncu-rep  # View results

**NVIDIA Visual Profiler (nvvp)**

   * Legacy tool (being replaced by Nsight)
   * Still useful for older GPUs

AMD Tools
~~~~~~~~~

**ROCProfiler**

   * AMD's profiling tool
   * Similar to NVIDIA tools

   Usage::

       rocprof --stats python script.py

**Radeon GPU Profiler**

   * GUI-based profiler
   * Visualization tools

PyTorch Profiler
~~~~~~~~~~~~~~~~

**torch.profiler**

   * Profile PyTorch operations
   * Integrated with TensorBoard

   Example::

       with torch.profiler.profile() as prof:
           output = model(input)

       print(prof.key_averages().table())

Benchmarking
~~~~~~~~~~~~

**Triton Built-in Benchmarking**::

    @triton.testing.perf_report(configs)
    def benchmark(...):
        ms = triton.testing.do_bench(lambda: kernel(...))
        return performance_metric(ms)

**PyTorch Benchmark**::

    import torch.utils.benchmark as benchmark

    t = benchmark.Timer(stmt='operation()', globals={...})
    print(t.timeit(100))

Online Resources
----------------

Tutorials and Courses
~~~~~~~~~~~~~~~~~~~~~

**NVIDIA Deep Learning Institute**

   * `DLI Courses <https://www.nvidia.com/en-us/training/>`_
   * Hands-on GPU programming courses
   * Free and paid options

**Coursera - GPU Programming Specialization**

   * University courses on GPU programming
   * Theory and practice

**YouTube - NVIDIA Developer Channel**

   * Conference talks
   * Tutorial videos
   * Latest technology updates

Community
~~~~~~~~~

**Triton Community**

   * `GitHub Discussions <https://github.com/openai/triton/discussions>`_
   * Active community
   * Get help, share kernels

**PyTorch Forums**

   * `discuss.pytorch.org <https://discuss.pytorch.org/>`_
   * Questions on PyTorch + Triton integration

**Reddit**

   * r/CUDA - CUDA programming
   * r/MachineLearning - ML discussions
   * r/computergraphics - GPU graphics

Blogs and Articles
~~~~~~~~~~~~~~~~~~

**Lil'Log**

   * `lilianweng.github.io <https://lilianweng.github.io/>`_
   * Excellent ML explanations
   * Flash Attention breakdown

**Jay Alammar's Blog**

   * `jalammar.github.io <https://jalammar.github.io/>`_
   * Visual guides to Transformers
   * Attention mechanisms explained

**Hugging Face Blog**

   * `huggingface.co/blog <https://huggingface.co/blog>`_
   * ML engineering articles
   * Optimization techniques

Example Repositories
--------------------

Triton Examples
~~~~~~~~~~~~~~~

* `OpenAI Triton Tutorials <https://github.com/openai/triton/tree/main/python/tutorials>`_ - Official tutorials
* `Triton Puzzles <https://github.com/srush/Triton-Puzzles>`_ - Learn by solving puzzles
* `Awesome Triton <https://github.com/Dao-AILab/awesome-triton>`_ - Curated list of Triton resources

Production Usage
~~~~~~~~~~~~~~~~

* `vLLM <https://github.com/vllm-project/vllm>`_ - LLM inference engine using Triton
* `Flash Attention <https://github.com/Dao-AILab/flash-attention>`_ - Official Flash Attention implementation
* `xformers <https://github.com/facebookresearch/xformers>`_ - Efficient Transformer components

Hardware Documentation
----------------------

NVIDIA GPUs
~~~~~~~~~~~

**Architecture Whitepapers**

   * `Volta Architecture <https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf>`_
   * `Ampere Architecture <https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf>`_
   * `Hopper Architecture <https://resources.nvidia.com/en-us-tensor-core>`_

**GPU Specifications**

   * `NVIDIA Data Center GPUs <https://www.nvidia.com/en-us/data-center/>`_
   * A100, H100, L40S specifications
   * Compute capabilities reference

AMD GPUs
~~~~~~~~

**CDNA Architecture**

   * `MI200 Architecture <https://www.amd.com/en/products/accelerators/instinct/mi200.html>`_
   * `MI300 Series <https://www.amd.com/en/products/accelerators/instinct/mi300.html>`_

Performance Databases
---------------------

**MLPerf**

   * `mlcommons.org/benchmarks <https://mlcommons.org/benchmarks/>`_
   * Standardized ML benchmarks
   * Compare different hardware

**Tensor Core Performance**

   * NVIDIA's published FLOPS numbers
   * Vendor benchmarks

Keeping Up to Date
------------------

Subscribe To
~~~~~~~~~~~~

* `Triton Releases <https://github.com/openai/triton/releases>`_ - New features
* `NVIDIA Developer Blog <https://developer.nvidia.com/blog>`_ - GPU news
* `PyTorch Blog <https://pytorch.org/blog/>`_ - Framework updates
* `Papers with Code <https://paperswithcode.com/>`_ - Latest research

Conferences
~~~~~~~~~~~

* **GTC (GPU Technology Conference)** - NVIDIA's annual conference
* **NeurIPS** - Neural Information Processing Systems
* **ICML** - International Conference on Machine Learning
* **MLSys** - Machine Learning and Systems

Academic Courses
----------------

**Stanford CS149 - Parallel Computing**

   * `cs149.stanford.edu <https://cs149.stanford.edu/>`_
   * GPU programming fundamentals
   * Assignments and lecture notes

**Stanford CS231n - CNNs for Visual Recognition**

   * Covers optimization and efficiency
   * GPU acceleration topics

**MIT 6.S965 - TinyML and Efficient Deep Learning**

   * Efficiency techniques
   * Includes GPU optimization

Contributing
------------

Want to contribute to the ecosystem?

* **Triton**: Submit kernels, fix bugs, improve docs
* **PyTorch**: Integrate Triton kernels
* **Research**: Publish new techniques
* **Education**: Write tutorials, create videos

See `Contributing Guidelines <https://github.com/openai/triton/blob/main/CONTRIBUTING.md>`_

Citation
--------

If you use Triton in research, cite::

    @inproceedings{tillet2019triton,
      author={Tillet, Philippe and Kung, H. T. and Cox, David},
      title={Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations},
      booktitle={Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages},
      year={2019},
      pages={10--19}
    }

For Flash Attention::

    @inproceedings{dao2022flashattention,
      title={Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
      author={Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
      booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
      year={2022}
    }

Quick Reference
---------------

**Most Important**:

1. Triton docs: https://triton-lang.org/
2. CUDA guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
3. Flash Attention paper: https://arxiv.org/abs/2205.14135

**For Help**:

1. Triton discussions: https://github.com/openai/triton/discussions
2. PyTorch forums: https://discuss.pytorch.org/

**For Profiling**:

1. NVIDIA: ``nsys`` and ``ncu``
2. AMD: ``rocprof``

Stay curious and keep optimizing! 🚀
