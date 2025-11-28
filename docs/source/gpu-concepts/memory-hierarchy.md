# GPU Memory Hierarchy

Understanding the GPU memory hierarchy is crucial for writing high-performance kernels. Memory access patterns often determine whether your kernel is fast or slow.

## Overview

GPUs have a multi-level memory hierarchy, similar to CPUs but with different characteristics:

```
Registers (fastest, smallest)
    ↓ ~1 cycle latency
L1 Cache / Shared Memory (SRAM)
    ↓ ~10 cycles latency
L2 Cache
    ↓ ~100 cycles latency
Global Memory / HBM (slowest, largest)
    ↓ ~400 cycles latency
```

### Memory Hierarchy Details

| Level | Size | Bandwidth | Latency | Scope |
|-------|------|-----------|---------|-------|
| Registers | 256 KB/SM | ~100 TB/s | 1 cycle | Per thread |
| Shared/L1 | 128-256 KB/SM | 10-20 TB/s | ~10 cycles | Per thread block |
| L2 Cache | 40-60 MB | ~5 TB/s | ~100 cycles | Whole GPU |
| Global/HBM | 40-80 GB | 1-3 TB/s | ~400 cycles | Whole GPU |

## Global Memory (HBM/DRAM)

### Characteristics

- Largest memory (40-80 GB on modern GPUs)
- Slowest access (~400 cycle latency)
- Highest bandwidth (1-3 TB/s)
- Off-chip (separate memory chips)

### Best Practices

1. **Minimize accesses**: Each access costs ~400 cycles
2. **Coalesce accesses**: Adjacent threads should access adjacent addresses
3. **Maximize bandwidth utilization**: Transfer large chunks, not individual elements

### Memory Coalescing

**Uncoalesced Access (Slow)**:
```
Thread 0: Read address 0
Thread 1: Read address 1000
Thread 2: Read address 2000
Thread 3: Read address 3000
-> 4 separate memory transactions
```

**Coalesced Access (Fast)**:
```
Thread 0: Read address 0
Thread 1: Read address 4
Thread 2: Read address 8
Thread 3: Read address 12
-> 1 combined memory transaction (128 bytes)
```

**In Triton**:
```python
# This automatically coalesces!
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
data = tl.load(ptr + offsets)  # Adjacent threads -> adjacent addresses
```

## L2 Cache

### Characteristics

- 40-60 MB on modern GPUs
- Shared across all SMs
- Caches global memory accesses
- ~100 cycle latency (4x faster than global memory)

### Optimization Strategy

- Reuse data across thread blocks
- Process data in patterns that maximize L2 hit rate
- Use "swizzling" techniques for matrix operations

**Example in matrix multiplication:**
- Bad: Process blocks in row-major order
- Good: Process blocks in grouped order (better L2 reuse)

See [Matrix Multiplication Tutorial](../gpu-tutorials/03-matrix-multiplication.md) for details.

## L1 Cache / Shared Memory (SRAM)

### Characteristics

- 128-256 KB per SM
- Configurable split between L1 cache and shared memory
- ~10 cycle latency (40x faster than global memory!)
- **Explicitly managed** in CUDA, **automatic** in Triton

### Shared Memory

Explicitly allocated memory shared by threads in a block:

```python
# CUDA (explicit)
__shared__ float shared_data[256];

# Triton (automatic when loading blocks)
data = tl.load(ptr + offsets)  # Triton manages SRAM automatically
```

### Key Uses

1. **Staging area**: Load from global → SRAM → process → write back
2. **Data reuse**: Multiple threads access same data from SRAM
3. **Communication**: Threads in block share results via SRAM

### Example: Matrix Multiplication

```python
# Load A block into SRAM (reused BLOCK_N times)
a = tl.load(a_ptrs)  # Loaded to SRAM

# Load B block into SRAM (reused BLOCK_M times)
b = tl.load(b_ptrs)  # Loaded to SRAM

# Compute using SRAM data (very fast!)
accumulator = tl.dot(a, b, accumulator)
```

## Registers

### Characteristics

- Fastest memory (1 cycle access)
- 256 KB per SM (divided among threads)
- Private to each thread
- Limited supply!

### Register Pressure

Each thread needs registers for:
- Local variables
- Intermediate computations
- Accumulation

**Too many registers** → fewer threads per SM → lower occupancy → lower performance

**Example**:
```python
# Each thread needs registers for:
accumulator = tl.zeros([BLOCK_M, BLOCK_N], dtype=float32)
# BLOCK_M=128, BLOCK_N=128 -> 16K elements per thread!
# Each element needs 1 register -> 16K registers
# But only 255 registers available per thread!

# Solution: Divide work among multiple threads in block
```

## Memory Access Patterns

### Pattern 1: Streaming (Bandwidth-Bound)

Read data once, process, write once:

```python
x = tl.load(x_ptr + offsets)
y = x + 1  # Simple operation
tl.store(y_ptr + offsets, y)
```

**Characteristics**:
- Memory bandwidth is bottleneck
- Examples: vector add, element-wise operations
- Goal: Achieve high % of peak bandwidth

### Pattern 2: Staged Computation (Compute-Bound)

Load to SRAM, heavy computation, write result:

```python
# Load blocks to SRAM
a = tl.load(a_ptrs)  # MxK block
b = tl.load(b_ptrs)  # KxN block

# Heavy computation on SRAM data
for k in range(0, K, BLOCK_K):
    accumulator += tl.dot(a, b)  # Many ops per load!

# Write result
tl.store(c_ptrs, accumulator)
```

**Characteristics**:
- Computation is bottleneck (good!)
- Data reused many times from SRAM
- Examples: matrix multiplication, convolution
- Goal: Maximize compute utilization

### Pattern 3: Reduction (Mixed)

Load data, reduce to smaller output:

```python
row = tl.load(input_ptrs + offsets, mask=mask)
max_val = tl.max(row, axis=0)  # Many inputs -> one output
sum_val = tl.sum(row, axis=0)  # Many inputs -> one output
```

**Characteristics**:
- Reads more data than writes
- Benefits from keeping data in SRAM
- Examples: softmax, layer norm
- Goal: Minimize memory traffic, maximize SRAM use

## Optimizing for Memory Hierarchy

### The Golden Rules

1. **Minimize global memory accesses**
   - Load once, use many times
   - Fuse operations to avoid intermediate writes

2. **Maximize SRAM usage**
   - Keep data in SRAM as long as possible
   - Reuse data across threads

3. **Ensure coalesced accesses**
   - Adjacent threads access adjacent addresses
   - Triton usually handles this automatically

4. **Consider register pressure**
   - Don't allocate too much per thread
   - Balance parallelism vs resources

5. **Optimize for L2 cache**
   - Reuse data across blocks
   - Use swizzling for better locality

### Example: Naive vs Optimized Softmax

**Naive (Multiple passes through global memory)**:
```python
x_max = x.max(dim=1)              # Pass 1: Read all
z = x - x_max                      # Pass 2: Read all, write all
numerator = torch.exp(z)           # Pass 3: Read all, write all
denominator = numerator.sum(dim=1) # Pass 4: Read all
output = numerator / denominator   # Pass 5: Read all, write all

# Total: 5 reads + 3 writes = 8 global memory passes!
```

**Optimized (Single pass with SRAM)**:
```python
@triton.jit
def softmax_kernel(...):
    # Load row once into SRAM
    row = tl.load(input_ptrs)

    # All computation in SRAM
    max_val = tl.max(row)
    numerator = tl.exp(row - max_val)
    denominator = tl.sum(numerator)
    output = numerator / denominator

    # Write result once
    tl.store(output_ptrs, output)

# Total: 1 read + 1 write = 2 global memory passes!
# 4x reduction in memory traffic = 4x speedup!
```

## Measuring Memory Performance

### Key Metrics

**Arithmetic Intensity**:
```
AI = FLOPs / Bytes Transferred
```
- Low AI (<10): Memory-bound
- High AI (>100): Compute-bound

**Memory Bandwidth Utilization**:
```
Achieved Bandwidth / Peak Bandwidth
```
- Good performance: 60-80% for memory-bound ops
- Poor performance: <30% suggests optimization opportunities

**Example**:
```
Vector Add: 1 FLOP, 12 bytes (read x, read y, write z)
AI = 1/12 ~ 0.08 FLOPs/byte
-> Heavily memory-bound!

Matrix Multiply: 2MNK FLOPs, 2(MK + KN + MN) bytes
For N=1024: AI = 512 FLOPs/byte
-> Compute-bound!
```

### Tools for Profiling

**NVIDIA**:
```bash
nsys profile python your_script.py  # Timeline analysis
ncu --set full python your_script.py  # Detailed metrics
```

**AMD**:
```bash
rocprof python your_script.py
```

**Key metrics to watch**:
- Memory throughput (GB/s)
- L1/L2 cache hit rates
- Occupancy
- Register usage per thread

## Summary

The memory hierarchy is the key to GPU performance:

- **Registers**: Fastest, use for temporaries
- **Shared Memory**: Fast, use for data reuse within block
- **L2 Cache**: Automatic, but can optimize access patterns
- **Global Memory**: Slow, minimize accesses

**Core Strategy**: Load data from slow memory → process in fast memory → write result

**Next**: Learn about [Execution Model](execution-model.md) to understand how GPUs schedule work.
