# Using External Functions (libdevice) in Triton

## Overview
This tutorial shows how to call external library functions from Triton kernels. Specifically, we'll use CUDA's `libdevice` (or AMD's device libraries) to access optimized mathematical functions that aren't available in standard Triton.

## What You'll Learn
- How to use **external library functions** in Triton
- What **libdevice** is and why it's useful
- How to specify **library paths** for compilation
- The difference between Triton intrinsics and external functions
- When to use external functions vs Triton built-ins

## What is libdevice?

### NVIDIA's libdevice

**libdevice** is a library of device-side mathematical functions for CUDA:
- Provided by NVIDIA as part of CUDA toolkit
- Highly optimized for GPU architectures
- Covers special functions not in standard CUDA

**File format**: `.bc` (LLVM bitcode)
- `libdevice.10.bc`: Version for Compute Capability 10+ (Hopper, Blackwell)
- Contains pre-compiled device functions

### AMD's Device Libraries

AMD provides similar libraries for ROCm:
- **ocml.bc**: OpenCL math library (sin, cos, exp, etc.)
- **ockl.bc**: OpenCL kernel library (atomics, barriers, etc.)

### What Functions Are Available?

**Examples from libdevice**:
```c
double __nv_asin(double)      // Arc sine
float  __nv_asinf(float)       // Arc sine (single precision)
double __nv_j0(double)         // Bessel function J0
float  __nv_erfcinvf(float)    // Inverse complementary error function
double __nv_tgamma(double)     // Gamma function
```

**Full list**: See [CUDA libdevice Users Guide](https://docs.nvidia.com/cuda/libdevice-users-guide/)

## Why Use External Functions?

### Triton Built-in Math

Triton provides common functions:
```python
tl.exp(x)    # Exponential
tl.log(x)    # Natural logarithm
tl.sin(x)    # Sine
tl.cos(x)    # Cosine
tl.sqrt(x)   # Square root
tl.abs(x)    # Absolute value
```

These are fast and convenient!

### When You Need More

For specialized functions:
```python
asin(x)      # Arc sine - NOT in Triton intrinsics!
bessel_j0(x) # Bessel function - NOT available!
erf(x)       # Error function - need libdevice
gamma(x)     # Gamma function - need libdevice
```

**Options**:
1. Implement yourself (complex, error-prone)
2. Use libdevice (optimized, tested, easy)

## Using libdevice in Triton

### The Simple Way (Default Path)

Triton automatically finds libdevice:

```python
from triton.language.extra import libdevice

@triton.jit
def asin_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = libdevice.asin(x)  # Call libdevice function!
    tl.store(y_ptr + offsets, y, mask=mask)
```

**Triton handles**:
- Finding libdevice.bc in the installation
- Linking it during compilation
- Type matching (float vs double)

### The Custom Path Way

Specify library path explicitly:

```python
import triton
from pathlib import Path

# Find libdevice in Triton installation
triton_dir = Path(triton.__file__).parent
libdir = triton_dir / 'backends/nvidia/lib'
extern_libs = {'libdevice': str(libdir / 'libdevice.10.bc')}

# Pass to kernel
asin_kernel[grid](x, y, n_elements, BLOCK_SIZE=1024, extern_libs=extern_libs)
```

**When to use custom paths**:
- Testing different library versions
- Using custom-built libraries
- Debugging linking issues

## How It Works Under the Hood

### Compilation Process

1. **Triton → LLVM IR**: Your kernel is compiled to LLVM intermediate representation

2. **Link external libraries**: LLVM linker merges your code with libdevice.bc

3. **Optimize**: LLVM optimizes the combined code

4. **Generate PTX** (NVIDIA) or **AMDGCN** (AMD): Backend-specific assembly

5. **JIT to machine code**: Final GPU binary

### Type Dispatch

Triton's `libdevice.asin` is a wrapper:

```python
def asin(x):
    if x.dtype == tl.float32:
        return __nv_asinf(x)  # Single precision
    elif x.dtype == tl.float64:
        return __nv_asin(x)   # Double precision
    # ... etc for other types
```

**Benefit**: You write `libdevice.asin(x)`, Triton picks the right implementation!

### Calling Convention

Libdevice functions use standard LLVM calling conventions:
- Arguments in registers or stack
- Return value in register
- Triton generates correct call sequences automatically

## Available libdevice Functions in Triton

### Triton's libdevice Wrapper

Located in `triton/language/extra/libdevice.py`:

```python
from triton.language.extra import libdevice

# Trigonometric
libdevice.sin(x)
libdevice.cos(x)
libdevice.tan(x)
libdevice.asin(x)   # Arc sine (our example!)
libdevice.acos(x)   # Arc cosine
libdevice.atan(x)   # Arc tangent

# Hyperbolic
libdevice.sinh(x)
libdevice.cosh(x)
libdevice.tanh(x)

# Exponential/Logarithmic
libdevice.exp(x)
libdevice.log(x)
libdevice.pow(x, y)

# Special functions
libdevice.erf(x)     # Error function
libdevice.erfc(x)    # Complementary error function
libdevice.j0(x)      # Bessel function J0
libdevice.j1(x)      # Bessel function J1
libdevice.y0(x)      # Bessel function Y0

# And many more!
```

**Tip**: Look at `triton/language/extra/libdevice.py` in the Triton source for full list.

## Example: Arc Sine (asin)

### The Math

Arc sine is the inverse of sine:
```
If sin(y) = x, then asin(x) = y
Domain: [-1, 1]
Range: [-π/2, π/2]
```

### Why Use libdevice?

**Implementing asin yourself**:
```python
def asin_approx(x):
    # Use Taylor series or other approximation
    # Need to handle edge cases (x near ±1)
    # Accuracy vs performance trade-off
    # Lots of potential bugs!
```

**Using libdevice**:
```python
y = libdevice.asin(x)  # Done! Optimized, accurate, tested
```

### Full Example

```python
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def asin_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask)

    # Apply arc sine using libdevice
    y = libdevice.asin(x)

    # Store result
    tl.store(y_ptr + offsets, y, mask=mask)

def asin_triton(x):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    asin_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output

# Test
x = torch.rand(100000, device='cuda')  # Random values in [0, 1]
y_torch = torch.asin(x)
y_triton = asin_triton(x)

# Should match!
max_diff = torch.max(torch.abs(y_torch - y_triton))
print(f'Max difference: {max_diff}')  # Should be ~1e-7 (float precision)
```

## Performance Considerations

### Libdevice Performance

**Generally very good**:
- Hand-optimized by NVIDIA/AMD
- Uses GPU-specific instructions
- Often as fast as or faster than DIY implementations

**Example timings** (1M elements):
- `libdevice.asin`: ~0.05 ms
- Custom approximation: ~0.03-0.10 ms (depends on accuracy)
- Benefit: Guaranteed correctness + good performance

### When to Use vs Triton Intrinsics

**Use Triton intrinsics when available**:
```python
tl.exp(x)   # Prefer this over libdevice.exp
tl.sin(x)   # Prefer this over libdevice.sin
```

Triton intrinsics may have special optimizations.

**Use libdevice for**:
- Functions not in Triton (asin, bessel, erf, etc.)
- When you need specific precision guarantees
- Complex mathematical operations

## Linking Multiple Libraries (AMD Example)

For AMD GPUs, need both ocml and ockl:

```python
from pathlib import Path
import triton

if is_hip():
    triton_dir = Path(triton.__file__).parent
    libdir = triton_dir / 'backends/amd/lib'

    extern_libs = {
        'ocml': str(libdir / 'ocml.bc'),
        'ockl': str(libdir / 'ockl.bc')
    }

    my_kernel[grid](..., extern_libs=extern_libs)
```

**Why multiple libraries?**
- Different functions in different libraries
- OCML: Math functions
- OCKL: Kernel utilities (barriers, atomics, etc.)

## Debugging Linking Issues

### Common Errors

**1. Library Not Found**
```
FileNotFoundError: [Errno 2] No such file or directory: '/path/to/libdevice.10.bc'
```

**Solution**: Check library path
```python
libdir = Path(triton.__file__).parent / 'backends/nvidia/lib'
print(libdir.exists())
print(list(libdir.glob('*.bc')))
```

**2. Undefined Symbol**
```
Error: undefined reference to '__nv_asin'
```

**Solution**: Make sure library is in `extern_libs` dict

**3. Type Mismatch**
```
Error: cannot convert 'float' to 'double'
```

**Solution**: Ensure input types match function expectations
```python
x = x.to(tl.float32)  # Explicitly cast if needed
```

### Verifying Libraries

```python
from pathlib import Path
import triton

triton_dir = Path(triton.__file__).parent

# NVIDIA
nvidia_lib = triton_dir / 'backends/nvidia/lib/libdevice.10.bc'
print(f'NVIDIA libdevice: {nvidia_lib.exists()}')

# AMD
amd_lib = triton_dir / 'backends/amd/lib/ocml.bc'
print(f'AMD ocml: {amd_lib.exists()}')
```

## Advanced Usage

### Calling Custom External Functions

You can link your own LLVM bitcode:

**1. Write CUDA device function**:
```cuda
// my_func.cu
extern "C" __device__ float my_special_func(float x) {
    return x * x + 2.0f * x + 1.0f;
}
```

**2. Compile to bitcode**:
```bash
clang++ --cuda-device-only -emit-llvm -c my_func.cu -o my_func.bc
```

**3. Link in Triton**:
```python
extern_libs = {'my_lib': '/path/to/my_func.bc'}
my_kernel[grid](..., extern_libs=extern_libs)
```

**4. Declare in Triton**:
```python
@triton.jit
def my_kernel(...):
    # Declare external function
    my_special_func = tl.extern_func('my_special_func', tl.float32, [tl.float32])

    # Use it
    y = my_special_func(x)
```

### Mixing Multiple External Libraries

```python
extern_libs = {
    'libdevice': '/path/to/libdevice.10.bc',
    'my_lib': '/path/to/my_func.bc',
    'another_lib': '/path/to/another.bc'
}

my_kernel[grid](..., extern_libs=extern_libs)
```

All libraries are linked together during compilation.

## Portability Considerations

### NVIDIA vs AMD

```python
def get_extern_libs():
    triton_dir = Path(triton.__file__).parent

    if is_cuda():
        return {
            'libdevice': str(triton_dir / 'backends/nvidia/lib/libdevice.10.bc')
        }
    elif is_hip():
        libdir = triton_dir / 'backends/amd/lib'
        return {
            'ocml': str(libdir / 'ocml.bc'),
            'ockl': str(libdir / 'ockl.bc')
        }
    else:
        return {}

extern_libs = get_extern_libs()
kernel[grid](..., extern_libs=extern_libs)
```

### Function Name Differences

Some functions have different names:
- NVIDIA: `__nv_sinf`
- AMD: `__ocml_sin_f32`

**Triton's libdevice wrapper handles this automatically!**

## Comparison to CUDA

### In CUDA

```cuda
#include <math_functions.h>

__global__ void asin_kernel(float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = asinf(x[i]);  // Automatically linked
    }
}
```

- Include header
- Use function
- nvcc handles linking

### In Triton

```python
from triton.language.extra import libdevice

@triton.jit
def asin_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # ...
    y = libdevice.asin(x)  # Import wrapper, use function
```

**Very similar experience!**

## Key Takeaways

1. **libdevice provides optimized math functions**: Beyond Triton's built-ins
2. **Easy to use**: `from triton.language.extra import libdevice`
3. **Automatic type dispatch**: Triton picks float vs double version
4. **Custom paths available**: For advanced use cases
5. **Good performance**: NVIDIA/AMD optimized implementations
6. **Portability**: Triton handles differences between backends
7. **Extensible**: Can link your own LLVM bitcode
8. **Essential for complex math**: Bessel functions, error functions, etc.

External functions let you leverage decades of numerical optimization work while writing simple Triton code!
