Fused Attention (Flash Attention) in Triton
===========================================

Overview
--------
This implements **Flash Attention v2**, a revolutionary algorithm for computing attention in Transformers. It reduces memory usage from O(N²) to O(N) and achieves 2-4x speedup by using **tiling**, **online softmax**, and **recomputation** strategies. This is one of the most advanced GPU kernels you'll encounter.

What You'll Learn
-----------------
- The **quadratic memory problem** in standard attention
- **Flash Attention algorithm** and its innovations
- **Online softmax**: Computing softmax without storing QK^T matrix
- **Causal masking** for autoregressive models
- **Tensor descriptors** for advanced memory access
- **Shared memory management** for large K/V blocks
- Why attention is the bottleneck in long-sequence Transformers

Background: The Attention Mechanism
-----------------------------------

Standard Attention Formula
~~~~~~~~~~~~~~~~~~~~~~~~~~

::

Attention(Q, K, V) = softmax(QK^T / √d) V
::


Where:
- Q (Query): (batch, heads, seq*len, head*dim)
- K (Key): (batch, heads, seq*len, head*dim)
- V (Value): (batch, heads, seq*len, head*dim)
- Output: (batch, heads, seq*len, head*dim)

Step-by-Step Computation
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Compute similarity scores**:
   ::

   S = QK^T / √d*k
   S shape: (seq*len, seq*len)  # N×N matrix!
   ::


2. **Apply softmax**:
   ::

   P = softmax(S)
   P shape: (seq*len, seq*len)  # Still N×N!
   ::


3. **Weighted sum of values**:
   ::

   O = PV
   O shape: (seq*len, head*dim)
   ::


The Memory Problem
~~~~~~~~~~~~~~~~~~

For sequence length N=2048, head*dim=64, fp16:

**Intermediate matrices**:
- S (scores): 2048 × 2048 × 2 bytes = 8 MB
- P (attention weights): 2048 × 2048 × 2 bytes = 8 MB
- Total per head: 16 MB

**For a Transformer**:
- Batch size: 32
- Heads: 16
- Total: 32 × 16 × 16 MB = **8 GB** just for attention matrices!

**For N=16384** (long sequences):
- Per head: 16384² × 2 bytes = 512 MB
- Total: 32 × 16 × 512 MB = **256 GB** 😱

**Clearly unsustainable!**

Flash Attention: The Key Insights
---------------------------------

Insight 1: We Don't Need to Store S and P
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Standard approach**:
1. Compute full S = QK^T
2. Store S in HBM (slow)
3. Compute full P = softmax(S)
4. Store P in HBM
5. Compute O = PV

**Flash Attention**:
1. Compute and process S and P in **blocks**
2. Keep blocks in SRAM (fast)
3. Never materialize full S or P in HBM
4. Only store final output O

**Memory**: O(N²) → O(N) 🎉

Insight 2: Online Softmax
~~~~~~~~~~~~~~~~~~~~~~~~~

**Challenge**: How to compute softmax without storing full matrix?

**Standard softmax** requires two passes:
.. code-block:: python

Pass 1: Find max
================
max*score = max(S)

Pass 2: Compute softmax
=======================
P = exp(S - max*score) / sum(exp(S - max*score))
::


**Online algorithm**: Update running statistics as we process blocks!

Insight 3: Tiling
~~~~~~~~~~~~~~~~~

Process attention in blocks:
::

For each block of Q (BLOCK*M rows):
    For each block of K,V (BLOCK*N columns):
        Compute block of S
        Update running softmax statistics
        Accumulate into output
::


**Benefit**: Each block fits in SRAM (fast)!

The Online Softmax Algorithm
----------------------------

Standard Softmax
~~~~~~~~~~~~~~~~

For a row S*i = [s₁, s₂, ..., sₙ]:

.. code-block:: python

m = max(s₁, s₂, ..., sₙ)
numerator = [exp(s₁-m), exp(s₂-m), ..., exp(sₙ-m)]
l = sum(numerator)
P*i = numerator / l
::


Block-wise Computation
~~~~~~~~~~~~~~~~~~~~~~

Suppose we process in two blocks: [s₁, s₂] and [s₃, s₄]:

**After block 1**:
::

m*old = max(s₁, s₂)
l*old = exp(s₁ - m*old) + exp(s₂ - m*old)
::


**After block 2**:
::

m*new = max(m*old, max(s₃, s₄))
::


**Problem**: How to update l*old when m changes?

**Solution**: Correction factor!
::

α = exp(m*old - m*new)
l*new = α * l*old + exp(s₃ - m*new) + exp(s₄ - m*new)
::


**Why α?**
::

Old contribution: exp(s₁ - m*old) + exp(s₂ - m*old)
With new max:      exp(s₁ - m*new) + exp(s₂ - m*new)
                 = exp(s₁ - m*old - (m*new - m*old)) + exp(s₂ - m*old - (m*new - m*old))
                 = [exp(s₁ - m*old) + exp(s₂ - m*old)] * exp(m*old - m*new)
                 = l*old * α
::


Updating the Output
~~~~~~~~~~~~~~~~~~~

Similarly, the accumulated output needs correction:
::

output*old = (1/l*old) * [exp(s₁-m*old)*v₁ + exp(s₂-m*old)*v₂]
::


When we update max:
::

output*new = α * output*old + (1/l*new) * [exp(s₃-m*new)*v₃ + exp(s₄-m*new)*v₄]
::


**This is the heart of Flash Attention!**

Triton Implementation Details
-----------------------------

The Inner Loop
~~~~~~~~~~~~~~

.. code-block:: python

def *attn*fwd*inner(acc, l*i, m*i, q, desc*k, desc*v, ...):
    for start*n in range(lo, hi, BLOCK*N):
        # Load K block
        k = desc*k.load([offsetk*y, 0]).T

        # Compute QK^T for this block
        qk = tl.dot(q, k)

        # Apply scaling and mask (if causal)
        qk = qk * qk*scale
        if STAGE == 2:  # Causal
            mask = offs*m[:, None] >= (start*n + offs*n[None, :])
            qk = qk + tl.where(mask, 0, -1.0e6)

        # Update max
        m*ij = tl.maximum(m*i, tl.max(qk, 1))

        # Compute softmax probabilities for this block
        p = tl.exp(qk - m*ij[:, None])

        # Correction factor
        alpha = tl.exp(m*i - m*ij)

        # Update sum
        l*ij = tl.sum(p, 1)

        # Update output with correction
        acc = acc * alpha[:, None]

        # Load V block
        v = desc*v.load([offsetv*y, 0])

        # Accumulate
        acc = tl.dot(p, v, acc)

        # Update running statistics
        l*i = l*i * alpha + l*ij
        m*i = m*ij
::


**Key variables**:
- ``m*i``: Current max for each row
- ``l*i``: Current sum of exponentials for each row
- ``acc``: Current weighted sum output
- ``alpha``: Correction factor when max changes

Causal Masking
~~~~~~~~~~~~~~

For autoregressive models (GPT), prevent attending to future tokens:

.. code-block:: python

mask = offs*m[:, None] >= (start*n + offs*n[None, :])
qk = qk + tl.where(mask, 0, -1.0e6)
::


**Visual example** (BLOCK*M=4, BLOCK*N=4):

::

Query rows: [0, 1, 2, 3]
Key columns: [0, 1, 2, 3]

Mask (1 = allowed, 0 = masked):
[[1, 0, 0, 0],   # Query 0 can only attend to Key 0
 [1, 1, 0, 0],   # Query 1 can attend to Keys 0-1
 [1, 1, 1, 0],   # Query 2 can attend to Keys 0-2
 [1, 1, 1, 1]]   # Query 3 can attend to Keys 0-3
::


Setting masked positions to ``-1e6`` → ``exp(-1e6) ≈ 0`` → effectively ignored.

Stages for Causal Attention
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

if STAGE == 1:
    lo, hi = 0, start*m * BLOCK*M  # Before diagonal
elif STAGE == 2:
    lo, hi = start*m * BLOCK*M, (start*m + 1) * BLOCK*M  # On diagonal (causal)
else:
    lo, hi = 0, N*CTX  # Full attention (non-causal)
::


**For causal attention**:
- **Stage 1**: Process all blocks before the diagonal (full blocks, no masking needed)
- **Stage 2**: Process diagonal block (needs causal masking)

**Optimization**: Stage 1 can skip masking checks, runs faster!

Memory Optimizations
--------------------

Tensor Descriptors
~~~~~~~~~~~~~~~~~~

Modern GPUs (Hopper, Blackwell) support **tensor descriptors**:

.. code-block:: python

desc*q = TensorDescriptor(q, shape=[y*dim, HEAD*DIM],
                          strides=[HEAD*DIM, 1],
                          block*shape=[BLOCK*M, HEAD*DIM])
::


**Benefits**:
- Hardware-assisted bounds checking
- Automatic address calculation
- Faster than manual pointer arithmetic

**Triton abstracts this**:
.. code-block:: python

q = desc*q.load([offset*y, 0])
::


Looks simple, but uses advanced hardware features!

Warp Specialization
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

warp*specialize = True
::


**Concept**: Different warps in a block do different tasks
- Some warps: Load data from DRAM
- Other warps: Compute on data
- Overlap communication and computation!

**Hopper/Blackwell specific**: These GPUs have dedicated units for loads, so specialization helps.

Reduced Shared Memory
~~~~~~~~~~~~~~~~~~~~~

By using online softmax:
- Don't store full QK^T matrix
- Only need current blocks in SRAM
- Typical usage: ~100 KB vs MBs for naive implementation

FP8 Support
~~~~~~~~~~~

For Blackwell GPUs with FP8 Tensor Cores:

.. code-block:: python

if dtype == tl.float8e5:
    v = desc*v.load([0, offsetv*y]).T  # Transposed layout for FP8
else:
    v = desc*v.load([offsetv*y, 0])
::


**FP8 benefits**:
- 2x memory bandwidth (8 bits vs 16 bits)
- 2x compute throughput (Tensor Cores)
- Trade-off: Slight accuracy loss (acceptable for ML)

Backward Pass
-------------

The backward pass is even more complex!

Preprocess Step
~~~~~~~~~~~~~~~

.. code-block:: python

def *attn*bwd*preprocess(O, DO, Delta, ...):
    # Compute delta = sum(O * DO) for each row
    o = tl.load(O + ...)
    do = tl.load(DO + ...)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + ..., delta)
::


**Delta**: Row-wise dot product of output and output gradient
- Used in both dK/dV and dQ computation
- Precompute once, use multiple times

Computing dK and dV
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

def *attn*bwd*dkdv(dk, dv, Q, k, v, DO, M, D, ...):
    for each block of Q:
        # Recompute attention weights
        qk = tl.dot(k, Q)
        p = tl.exp(qk - M)  # M was saved from forward pass

        # Compute dV
        dv += tl.dot(p, DO)

        # Compute dP (gradient of attention weights)
        dp = tl.dot(v, DO.T)

        # Compute dS (before softmax)
        ds = p * (dp - D)  # D is delta from preprocess

        # Compute dK
        dk += tl.dot(ds, Q.T)
::


**Key insight**: Recompute attention weights from Q, K (which we have)
- Don't need to store P from forward pass
- Classic memory-compute trade-off

Computing dQ
~~~~~~~~~~~~

Similar structure, but processes different blocks:

.. code-block:: python

def *attn*bwd*dq(dq, q, K, V, DO, M, D, ...):
    for each block of K,V:
        # Recompute attention
        qk = tl.dot(q, K)
        p = tl.exp(qk - M)

        # Compute dP
        dp = tl.dot(DO, V.T)

        # Compute dS
        ds = p * (dp - D)

        # Compute dQ
        dq += tl.dot(ds, K)
::


Why Recomputation?
~~~~~~~~~~~~~~~~~~

**Traditional approach**:
- Forward: Compute and store P (N² memory)
- Backward: Load P, compute gradients

**Flash Attention**:
- Forward: Don't store P (O(N) memory)
- Backward: Recompute P from Q, K (extra compute, but much less memory)

**Trade-off**:
- 2x compute (recompute in backward)
- But enables 100x longer sequences!

Performance Characteristics
---------------------------

Memory Complexity
~~~~~~~~~~~~~~~~~

**Standard attention**:
- O(N²) for S and P matrices
- Becomes prohibitive for long sequences

**Flash Attention**:
- O(N) for Q, K, V, O
- No intermediate N² matrices
- Enables sequences of 16K, 32K, even 100K+ tokens

Compute Complexity
~~~~~~~~~~~~~~~~~~

Still O(N²) FLOPs (can't change the math), but:
- Better memory access patterns
- Higher arithmetic intensity
- Better cache utilization

**Result**: 2-4x faster despite same FLOPs!

Arithmetic Intensity
~~~~~~~~~~~~~~~~~~~~

With tiling (BLOCK*M=128, BLOCK*N=64):
- Load Q block: 128 × 64 elements
- Load K block: 64 × 64 elements
- Load V block: 64 × 64 elements
- Compute QK: 128 × 64 × 64 = 512K FLOPs
- Compute PV: 128 × 64 × 64 = 512K FLOPs
- Total: ~1M FLOPs per ~20K bytes loaded

Arithmetic Intensity: 1M / 20K ≈ 50 FLOPs/byte

**Compare to naive** (no tiling):
- Each element of S: 1 load Q row, 1 load K row
- Each element of P: 1 load/store
- AI ≈ 2 FLOPs/byte

Flash Attention is **25x more compute-efficient** in memory access!

Auto-Tuning Configurations
--------------------------

.. code-block:: python

configs = [
    triton.Config({'BLOCK*M': BM, 'BLOCK*N': BN}, num*stages=s, num*warps=w)
    for BM in [64, 128]
    for BN in [32, 64, 128]
    for s in [2, 3, 4]
    for w in [4, 8]
]
::


**Block size trade-offs**:
- **Larger blocks**: More reuse, but more SRAM usage
- **Smaller blocks**: Less SRAM, but more overhead

**Typical good configs**:
- BLOCK*M=128, BLOCK*N=64: Balanced
- BLOCK*M=64, BLOCK*N=32: Smaller SRAM GPUs
- More stages: Better for Hopper/Blackwell with more SRAM

Common Pitfalls
---------------

1. SRAM Overflow
~~~~~~~~~~~~~~~~

::

Error: out of resource: shared memory
::


**Cause**: Blocks too large for available SRAM

**Solution**:
- Reduce BLOCK*M or BLOCK*N
- Reduce num*stages
- Adjust auto-tune configs

2. Numerical Instability
~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Exp overflow without proper max subtraction

.. code-block:: python

Wrong
=====
p = tl.exp(qk)

Right
=====
m*ij = tl.max(qk, 1)
p = tl.exp(qk - m*ij[:, None])
::


3. Causal Mask Errors
~~~~~~~~~~~~~~~~~~~~~

**Common mistake**: Using ``>`` instead of ``>=``

.. code-block:: python

Wrong for causal (token can't attend to itself!)
================================================
mask = offs*m[:, None] > (start*n + offs*n[None, :])

Right
=====
mask = offs*m[:, None] >= (start*n + offs*n[None, :])
::


4. Forgetting to Update Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

Must update both m*i and l*i!
=============================
l*i = l*i * alpha + l*ij
m*i = m*ij

And correct accumulator
=======================
acc = acc * alpha[:, None]
::


Forgetting any of these → wrong results.

Extensions and Variants
-----------------------

Multi-Query Attention (MQA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use same K, V for all heads:
- K, V shape: (batch, 1, seq*len, head*dim)
- Q shape: (batch, num*heads, seq*len, head*dim)
- Memory savings: num*heads× less for K, V

Grouped Query Attention (GQA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Groups of heads share K, V:
- Middle ground between MQA and full attention
- Used in Llama 2, Mistral

Sliding Window Attention
~~~~~~~~~~~~~~~~~~~~~~~~

Only attend to nearby tokens:
.. code-block:: python

mask = abs(offs*m[:, None] - offs*n[None, :]) <= window*size
::


Reduces complexity to O(N × window_size).

Flash Attention 3
~~~~~~~~~~~~~~~~~

Latest version (Hopper H100+):
- Warp-specialization optimizations
- Even better overlap of compute and memory
- FP8 support with dynamic scaling
- 1.5-2x faster than Flash Attention 2

Comparison to Standard Attention
--------------------------------

| Aspect | Standard | Flash Attention |
|--------|----------|-----------------|
| Memory | O(N²) | O(N) |
| Speed (2K seq) | 1x | 2-3x faster |
| Speed (16K seq) | OOM | 10x+ faster |
| Max sequence | ~2048 | 100K+ |
| Implementation | Simple | Complex |

Key Takeaways
-------------

1. **Flash Attention solves the O(N²) memory problem**: Enables long sequences
2. **Online softmax is the key innovation**: Compute softmax without full materialization
3. **Tiling + correction factors**: Update statistics as we process blocks
4. **Recomputation in backward pass**: Trade compute for memory
5. **SRAM management is critical**: Blocks must fit in fast memory
6. **Causal masking**: Essential for autoregressive models (GPT, etc.)
7. **Hardware features matter**: Tensor descriptors, warp specialization, FP8
8. **This enables modern LLMs**: GPT-4, Llama, etc. all use variants of Flash Attention

Flash Attention is arguably the most important algorithmic innovation for Transformers in recent years. It enabled the jump from ~2K to 100K+ token context windows!
