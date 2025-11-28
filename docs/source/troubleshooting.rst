Troubleshooting Guide
=====================

Common issues and their solutions when working with Triton and GPU programming.

Out of Memory Errors
--------------------

CUDA Out of Memory
~~~~~~~~~~~~~~~~~~

**Error Message**::

    RuntimeError: CUDA out of memory. Tried to allocate X MB

**Causes**:

1. Batch size too large
2. Sequence length too long
3. Too many intermediate tensors
4. Memory leak

**Solutions**:

**Reduce batch size**::

    # Before
    batch_size = 64

    # After
    batch_size = 32  # or 16

**Use gradient checkpointing**::

    # Recompute activations instead of storing
    from torch.utils.checkpoint import checkpoint

    output = checkpoint(my_function, input)

**Clear cache**::

    torch.cuda.empty_cache()

**Check for memory leaks**::

    # Detach tensors when not needed
    loss = compute_loss(output, target)
    loss_value = loss.item()  # Convert to Python number
    del loss  # Free memory

Out of Shared Memory
~~~~~~~~~~~~~~~~~~~~

**Error Message**::

    triton.runtime.errors.OutOfResources: out of resource: shared memory,
    Required: 109568, Hardware limit: 101376

**Causes**:

* Block sizes too large
* Too many pipeline stages (``num_stages``)

**Solutions**:

**Reduce block sizes**::

    # Before
    @triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256}, ...)

    # After
    @triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, ...)

**Reduce num_stages**::

    # Before
    num_stages = 5

    # After
    num_stages = 3  # or 2

**Adjust auto-tune configs**::

    configs = [
        triton.Config({...}, num_stages=2, num_warps=4),  # Less SRAM
        # Remove configs with large blocks
    ]

Correctness Issues
------------------

Wrong Results
~~~~~~~~~~~~~

**Symptoms**: Output doesn't match PyTorch or expected values

**Debug Steps**:

1. **Check masking**::

       mask = offsets < n_elements
       x = tl.load(x_ptr + offsets, mask=mask)  # Don't forget mask!

2. **Verify pointer arithmetic**::

       # Check strides are correct
       assert a.is_contiguous()

       # Print debug info
       print(f"Stride: {a.stride()}")

3. **Use float32 for accumulation**::

       # Bad: FP16 accumulation loses precision
       acc = tl.zeros([M, N], dtype=tl.float16)

       # Good: FP32 accumulation
       acc = tl.zeros([M, N], dtype=tl.float32)
       result = acc.to(tl.float16)  # Cast at end

4. **Check numerical stability**::

       # Softmax: always subtract max
       x_max = tl.max(x, axis=0)
       x_normalized = x - x_max  # Prevents overflow

5. **Verify boundary conditions**::

       # Test with non-power-of-2 sizes
       x = torch.randn(1001, device='cuda')  # Not 1024!

NaN or Inf Values
~~~~~~~~~~~~~~~~~

**Common causes**:

1. **Division by zero**::

       # Add epsilon
       result = x / (y + 1e-8)

2. **Overflow in exp**::

       # Subtract max before exp
       x = x - tl.max(x)
       result = tl.exp(x)

3. **Log of negative/zero**::

       # Clamp before log
       result = tl.log(tl.maximum(x, 1e-10))

4. **Uninitialized memory**::

       # Always initialize
       acc = tl.zeros([M, N], dtype=tl.float32)  # Not: acc = tl.empty(...)

Performance Issues
------------------

Slower Than PyTorch
~~~~~~~~~~~~~~~~~~~

**Diagnosis**:

1. **Profile both**::

       # PyTorch
       import torch.utils.benchmark as benchmark
       t = benchmark.Timer(stmt='torch.matmul(a, b)', globals={'a': a, 'b': b})
       print(t.timeit(100))

       # Triton
       ms = triton.testing.do_bench(lambda: triton_matmul(a, b))

2. **Check if PyTorch uses vendor libs**::

       # PyTorch often uses cuBLAS, cuDNN
       # These are extremely optimized
       # Matching them is success!

**Common reasons**:

1. **Not using Tensor Cores**

   * Ensure FP16/BF16 inputs
   * Use ``tl.dot()`` for matmul
   * Check block sizes are multiples of 16

2. **Suboptimal configuration**

   * Need auto-tuning
   * Try different block sizes
   * Adjust num_warps and num_stages

3. **Non-contiguous tensors**::

       # Check contiguity
       assert a.is_contiguous()

       # Make contiguous if needed
       a = a.contiguous()

4. **Missing optimizations**

   * No kernel fusion
   * Not using SRAM effectively
   * Poor memory access patterns

Low GPU Utilization
~~~~~~~~~~~~~~~~~~~

**Check with**::

    nvidia-smi -l 1  # Monitor GPU utilization

**If low (<50%)**:

1. **Increase batch size**: More parallel work
2. **Check occupancy**: May be too low
3. **Pipeline CPU-GPU**: Overlap data transfer and compute
4. **Profile**: Use ``nsys`` to find bottlenecks

**If high (>90%) but slow**:

* Memory-bound: Optimize memory access
* Compute-bound: Use Tensor Cores, increase arithmetic intensity

Compilation Issues
------------------

Compilation Errors
~~~~~~~~~~~~~~~~~~

**Error**: ``TypeError: unsupported operand type(s)``

**Cause**: Type mismatch in Triton

**Solution**::

    # Explicit casting
    x = x.to(tl.float32)
    y = y.to(tl.float32)
    result = x + y

**Error**: ``constexpr`` parameter not constant

**Solution**::

    # Must be compile-time constant
    BLOCK_SIZE: tl.constexpr = 128  # Not a variable!

Slow Compilation
~~~~~~~~~~~~~~~~

**First compilation is slow** (minutes):

* Normal! Triton JIT-compiles and auto-tunes
* Subsequent runs use cached version
* Use ``TRITON_CACHE_DIR`` to persist cache

**Every run is slow**:

* Check if cache is working::

      import os
      print(os.environ.get('TRITON_CACHE_DIR'))

* Set cache directory::

      export TRITON_CACHE_DIR=/path/to/cache

Platform-Specific Issues
-------------------------

NVIDIA-Specific
~~~~~~~~~~~~~~~

**Compute capability too low**::

    RuntimeError: Triton requires compute capability >= 7.0

**Solution**: Upgrade GPU (Volta or newer required)

**Driver version mismatch**::

    CUDA driver version is insufficient for CUDA runtime version

**Solution**::

    # Check versions
    nvidia-smi  # Driver version
    python -c "import torch; print(torch.version.cuda)"  # CUDA version

    # Update driver if needed

AMD-Specific
~~~~~~~~~~~~

**ROCm not found**::

    ModuleNotFoundError: No module named 'triton.backends.amd'

**Solution**: Install ROCm-enabled Triton::

    pip install triton --index-url https://download.pytorch.org/whl/rocm5.6

**Kernel launch failures**:

* Check ``HSA_OVERRIDE_GFX_VERSION`` for older GPUs
* Verify ROCm version matches GPU architecture

Multi-GPU Issues
----------------

Wrong GPU Selected
~~~~~~~~~~~~~~~~~~

**Specify GPU**::

    # Set before any CUDA operations
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0

    # Or in Python
    torch.cuda.set_device(0)

**In kernel**::

    DEVICE = torch.device(f'cuda:{gpu_id}')
    x = torch.randn(1000, device=DEVICE)

Debugging Techniques
--------------------

Print Debugging
~~~~~~~~~~~~~~~

**In kernel** (limited)::

    @triton.jit
    def kernel(...):
        pid = tl.program_id(0)

        # Only print from first program
        if pid == 0:
            tl.device_print("pid:", pid)

**Outside kernel**::

    # Print shapes, dtypes
    print(f"Shape: {x.shape}, dtype: {x.dtype}, device: {x.device}")

    # Check values
    print(f"Min: {x.min()}, Max: {x.max()}, Mean: {x.mean()}")

    # Look for NaN/Inf
    print(f"Has NaN: {torch.isnan(x).any()}")
    print(f"Has Inf: {torch.isinf(x).any()}")

Profiling
~~~~~~~~~

**NVIDIA**::

    # Timeline profiling
    nsys profile -o output python script.py

    # Detailed metrics
    ncu --set full -o output python script.py

    # Specific metrics
    ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed python script.py

**AMD**::

    rocprof --stats python script.py

Assertions
~~~~~~~~~~

**Add runtime checks**::

    @triton.jit
    def kernel(...):
        # Check bounds
        tl.static_assert(BLOCK_M <= 256, "BLOCK_M too large")

        # Runtime assertion
        tl.device_assert(offset < n_elements, "Out of bounds")

Unit Testing
~~~~~~~~~~~~

**Test correctness**::

    def test_my_kernel():
        x = torch.randn(1000, device='cuda')

        # Triton result
        y_triton = my_kernel(x)

        # Reference (PyTorch)
        y_torch = reference_implementation(x)

        # Check
        torch.testing.assert_close(y_triton, y_torch, rtol=1e-3, atol=1e-3)

Common Error Messages
---------------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Error
     - Solution
   * - ``CUDA out of memory``
     - Reduce batch size, use gradient checkpointing
   * - ``out of resource: shared memory``
     - Reduce BLOCK_SIZE, num_stages
   * - ``out of resource: registers``
     - Reduce local variables, smaller blocks
   * - ``constexpr parameter must be compile-time constant``
     - Use ``tl.constexpr`` type annotation
   * - ``Tensor must be contiguous``
     - Call ``.contiguous()`` on input
   * - ``Type mismatch``
     - Explicit ``to()`` casting
   * - ``AttributeError: 'Tensor' object has no attribute 'stride'``
     - Pass tensor, not pointer
   * - ``RuntimeError: unspecified launch failure``
     - Out of bounds access, check masking

Getting Help
------------

When Stuck
~~~~~~~~~~

1. **Search existing issues**: `Triton GitHub Issues <https://github.com/openai/triton/issues>`_

2. **Minimal reproducible example**::

       import torch
       import triton

       @triton.jit
       def broken_kernel(...):
           # Simplified version that shows the issue
           ...

3. **Provide details**:

   * Error message (full stack trace)
   * Triton version: ``import triton; print(triton.__version__)``
   * PyTorch version: ``import torch; print(torch.__version__)``
   * GPU model: ``nvidia-smi``
   * Minimal code to reproduce

4. **Ask in right place**:

   * `Triton Discussions <https://github.com/openai/triton/discussions>`_ - General questions
   * `Triton Issues <https://github.com/openai/triton/issues>`_ - Bugs
   * `PyTorch Forums <https://discuss.pytorch.org/>`_ - PyTorch integration

Best Practices for Avoiding Issues
-----------------------------------

1. **Start simple**: Get basic version working before optimizing
2. **Test incrementally**: Add features one at a time
3. **Verify correctness**: Always compare with PyTorch
4. **Profile early**: Understand bottlenecks before optimizing
5. **Use auto-tuning**: Don't guess optimal configurations
6. **Check edge cases**: Non-power-of-2 sizes, empty tensors
7. **Handle boundaries**: Always use masking for safety
8. **Maintain precision**: Use float32 for accumulation

Prevention Checklist
--------------------

Before deploying:

[ ] Tested with various input sizes
[ ] Compared output with PyTorch
[ ] Profiled performance
[ ] Checked for NaN/Inf
[ ] Verified memory usage is reasonable
[ ] Tested edge cases (size=1, size=prime number, etc.)
[ ] Added assertions for debug builds
[ ] Documented any limitations

Summary
-------

Most issues fall into three categories:

1. **Memory**: OOM, shared memory limits -> Reduce sizes
2. **Correctness**: Wrong results, NaN -> Check masking, precision, stability
3. **Performance**: Slow -> Profile, auto-tune, optimize memory access

When in doubt:

* Profile to find the real bottleneck
* Compare with PyTorch to verify correctness
* Start simple and add complexity incrementally

Still stuck? See :doc:`references` for more resources.
