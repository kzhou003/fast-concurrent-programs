# Triton GPU Programming Tutorials

Comprehensive guides to GPU programming with Triton, covering fundamental concepts through advanced techniques. Each tutorial builds on previous ones while introducing new GPU/CUDA/Triton concepts.

## 📚 Tutorial Overview

### Beginner Level

#### 1. [Vector Addition](01-vector-add.md)
**The "Hello World" of GPU Programming**

Learn the fundamentals:
- SPMD (Single Program, Multiple Data) execution model
- Program IDs and block processing
- Memory coalescing and bandwidth
- Grid launch and parallel execution
- Masking for boundary conditions

**Key Concepts**: Bandwidth-bound operations, memory hierarchy, parallel patterns

---

#### 2. [Fused Softmax](02-fused-softmax.md)
**Introduction to Kernel Fusion**

Optimize memory-bound operations:
- Kernel fusion for 4x speedup
- DRAM vs SRAM memory hierarchy
- Row-wise reduction operations
- Power-of-2 block sizes
- Numerical stability techniques

**Key Concepts**: Memory bandwidth, reduction operations, fusion benefits

---

### Intermediate Level

#### 3. [Matrix Multiplication](03-matrix-multiplication.md)
**High-Performance Computing Fundamentals**

Master compute-bound operations:
- Tiling/blocking strategies
- Multi-dimensional pointer arithmetic
- L2 cache optimization (swizzling)
- Auto-tuning configurations
- Tensor Cores and specialized hardware

**Key Concepts**: Compute-bound vs bandwidth-bound, arithmetic intensity, roofline model

---

#### 4. [Low-Memory Dropout](04-low-memory-dropout.md)
**Advanced Memory Management**

Efficient randomness on GPU:
- Counter-based PRNG (Philox algorithm)
- Seeded dropout for 50% memory reduction
- Deterministic reproducibility
- Parallel random number generation
- Memory vs computation trade-offs

**Key Concepts**: PRNG, memory optimization, trade-offs

---

### Advanced Level

#### 5. [Layer Normalization](05-layer-norm.md)
**Forward and Backward Passes**

Complete training loop implementation:
- Forward pass with mean/variance computation
- Backward pass with gradient computation
- Parallel reduction strategies
- Atomic operations and locks
- Two-stage gradient accumulation

**Key Concepts**: Backward pass, gradient computation, synchronization primitives

---

#### 6. [Fused Attention (Flash Attention)](06-fused-attention.md)
**State-of-the-Art Transformer Optimization**

Revolutionary algorithm for attention:
- Reducing O(N²) to O(N) memory
- Online softmax algorithm
- Tiling with correction factors
- Causal masking for autoregressive models
- Recomputation strategies
- Warp specialization

**Key Concepts**: Flash Attention, online algorithms, memory-compute trade-offs

---

### Specialized Topics

#### 7. [External Functions (libdevice)](07-extern-functions.md)
**Leveraging Optimized Libraries**

Use external mathematical functions:
- CUDA libdevice and AMD device libraries
- Linking LLVM bitcode
- Type dispatch and calling conventions
- Custom external functions
- Cross-platform compatibility

**Key Concepts**: Library linking, external functions, portability

---

## 🎯 Learning Path

### Path 1: Fast Track (Essentials Only)
1. Vector Addition → Understanding SPMD and parallelism
2. Fused Softmax → Kernel fusion and memory optimization
3. Matrix Multiplication → Compute optimization and tiling

**Time**: ~4-6 hours | **Level**: Get productive quickly

### Path 2: Deep Understanding (Comprehensive)
1. Vector Addition → Foundations
2. Fused Softmax → Memory optimization
3. Matrix Multiplication → Compute optimization
4. Low-Memory Dropout → Advanced memory techniques
5. Layer Normalization → Training loop implementation
6. Fused Attention → State-of-the-art optimization
7. External Functions → Extending capabilities

**Time**: ~12-16 hours | **Level**: Become an expert

### Path 3: Transformer Focus (For LLM/NLP)
1. Vector Addition → Basics
2. Fused Softmax → Attention building block
3. Layer Normalization → Transformer component
4. Fused Attention → Core optimization
5. Low-Memory Dropout → Memory efficiency

**Time**: ~8-10 hours | **Level**: Optimize Transformers

---

## 🔑 Key GPU/CUDA Concepts Covered

### Memory Hierarchy
- **DRAM (HBM)**: 40-80 GB, 1-2 TB/s bandwidth, ~400 cycle latency
- **L2 Cache**: 40-60 MB, ~5 TB/s, ~100 cycle latency
- **L1/Shared Memory (SRAM)**: 128-256 KB per SM, 10-20 TB/s, ~10 cycle latency
- **Registers**: 256 KB per SM, ~100 TB/s, 1 cycle latency

### Execution Model
- **SPMD**: Single Program, Multiple Data
- **Warps**: 32 threads (NVIDIA) or 64 threads (AMD) executing in lockstep
- **Thread Blocks**: Groups of warps, up to 1024 threads
- **Streaming Multiprocessors (SM)**: 80-140 SMs on modern GPUs
- **Occupancy**: Active thread blocks per SM

### Performance Metrics
- **FLOPS**: Floating Point Operations Per Second
  - A100: 312 TFLOPS (FP16 with Tensor Cores)
  - H100: 1000+ TFLOPS (FP8 with Tensor Cores)
- **Memory Bandwidth**: Data transfer rate
  - A100: 2 TB/s
  - H100: 3.35 TB/s
- **Arithmetic Intensity**: FLOPs / Bytes transferred
  - Low (<10): Memory-bound (vector add, softmax)
  - High (>100): Compute-bound (matmul, convolution)

### Optimization Techniques
1. **Kernel Fusion**: Combine operations to reduce memory traffic
2. **Tiling/Blocking**: Reuse data in fast memory
3. **Coalescing**: Align memory accesses for efficiency
4. **Occupancy Tuning**: Balance resource usage
5. **Auto-tuning**: Automatically find best configurations
6. **Warp Specialization**: Different warps do different tasks
7. **Software Pipelining**: Overlap compute and memory

### Hardware Features
- **Tensor Cores**: Specialized matrix multiply units (10-100x speedup)
- **Warp Shuffles**: Share data within warp without memory
- **Atomic Operations**: Thread-safe updates
- **Tensor Descriptors**: Hardware-assisted memory access (Hopper+)

---

## 📊 Performance Expectations

### Typical Speedups (vs PyTorch)
| Operation | Speedup | Why |
|-----------|---------|-----|
| Vector Add | 1x | Both are memory-bound, hitting bandwidth limit |
| Fused Softmax | 2-4x | Reduced memory traffic through fusion |
| Matrix Multiplication | 0.9-1x | Both use Tensor Cores, Triton ≈ cuBLAS |
| Dropout | 1-2x | Slightly slower but 50% less memory |
| Layer Norm | 1.5-2x | Fusion and optimized reduction |
| Fused Attention | 2-4x | Massive memory reduction enables speedup |

### Memory Savings
| Operation | Standard | Optimized | Savings |
|-----------|----------|-----------|---------|
| Dropout | 2-3x input | 1x input | 50-66% |
| Attention (2K seq) | 8 GB | 0.5 GB | 94% |
| Attention (16K seq) | 512 GB | 4 GB | 99%+ |

---

## 🛠️ Prerequisites

### Software
- Python 3.8+
- PyTorch 2.0+
- Triton 2.0+
- CUDA 11.8+ (NVIDIA) or ROCm 5.0+ (AMD)

### Hardware
- NVIDIA GPU with Compute Capability 7.0+ (Volta, Turing, Ampere, Hopper, Blackwell)
  - Recommended: A100, H100, RTX 4090, RTX 3090
- AMD GPU with CDNA or RDNA architecture
  - Recommended: MI200, MI300

### Knowledge
- **Essential**: Python programming, basic linear algebra
- **Helpful**: Understanding of neural networks, basic CUDA concepts
- **Not required**: No need to know C++ or CUDA programming

---

## 💡 Tips for Learning

### 1. Run the Code
Don't just read – execute the examples:
```bash
cd triton_cuda/triton_practice
python 01-vector-add.py
python 02-fused-softmax.py
# etc.
```

### 2. Modify and Experiment
- Change `BLOCK_SIZE` values
- Try different input sizes
- Add print statements to see intermediate values
- Break things to understand errors!

### 3. Use Profiling Tools
```bash
# NVIDIA
nsys profile python 03-matrix-multiplication.py
ncu --set full python 03-matrix-multiplication.py

# AMD
rocprof python 03-matrix-multiplication.py
```

### 4. Visualize Performance
All tutorials include benchmarking code:
```python
benchmark.run(save_path=".", print_data=True, show_plots=True)
```

### 5. Compare with PyTorch
Each tutorial compares Triton implementation with PyTorch:
- Verify correctness
- Measure speedup
- Understand trade-offs

---

## 🐛 Common Issues and Solutions

### Out of Memory (OOM)
**Symptom**: `CUDA out of memory` error

**Solutions**:
- Reduce batch size
- Decrease `BLOCK_SIZE_M`, `BLOCK_SIZE_N`
- Reduce `num_stages`
- Check for memory leaks

### Out of Shared Memory
**Symptom**: `out of resource: shared memory`

**Solutions**:
- Reduce block sizes
- Decrease `num_stages`
- Adjust auto-tune configurations

### Wrong Results
**Symptom**: Output doesn't match PyTorch

**Solutions**:
- Check masking logic
- Verify pointer arithmetic
- Use `float32` for accumulation
- Check numerical stability (subtract max in softmax)

### Slow Performance
**Symptom**: Triton slower than PyTorch

**Solutions**:
- Profile with `nsys` or `ncu`
- Check occupancy (might be too low)
- Adjust `num_warps` and `num_stages`
- Ensure data is contiguous
- Enable auto-tuning

---

## 📖 Additional Resources

### Official Documentation
- [Triton Documentation](https://triton-lang.org/)
- [Triton GitHub](https://github.com/openai/triton)
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [ROCm Documentation](https://rocmdocs.amd.com/)

### Research Papers
- [Flash Attention](https://arxiv.org/abs/2205.14135) - Dao et al., 2022
- [Flash Attention 2](https://arxiv.org/abs/2307.08691) - Dao, 2023
- [Triton](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf) - Tillet et al., 2019

### Community
- [Triton Discussions](https://github.com/openai/triton/discussions)
- [OpenAI Discord](https://discord.gg/openai)

---

## 🎓 What's Next?

After completing these tutorials, you can:

1. **Optimize Your Own Models**
   - Write custom kernels for bottleneck operations
   - Fuse operations for efficiency
   - Reduce memory usage for longer sequences

2. **Contribute to Open Source**
   - Triton itself
   - vLLM (LLM inference engine)
   - PyTorch (custom ops)

3. **Advanced Topics**
   - Multi-GPU kernels
   - Quantization (INT8, FP8)
   - Sparse operations
   - Custom backward passes

4. **Research**
   - Novel fusion patterns
   - New attention variants
   - Hardware-specific optimizations

---

## 🙏 Acknowledgments

These tutorials are based on:
- Official Triton tutorials by OpenAI
- Flash Attention by Tri Dao
- Community contributions and best practices

---

## 📝 License

These educational materials are provided as-is for learning purposes. The code examples follow the same license as Triton (MIT).

---

**Happy GPU Programming! 🚀**

Remember: The best way to learn GPU programming is to write code, break things, profile, and iterate. Don't be afraid to experiment!
