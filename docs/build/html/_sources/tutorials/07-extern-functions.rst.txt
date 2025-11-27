Using External Functions (libdevice) in Triton
==============================================

Overview
--------
This tutorial shows how to call external library functions from Triton kernels. Specifically, we'll use CUDA's ``libdevice`` (or AMD's device libraries) to access optimized mathematical functions that aren't available in standard Triton.

What You'll Learn
-----------------
- How to use **external library functions** in Triton
- What **libdevice** is and why it's useful
- How to specify **library paths** for compilation
- The difference between Triton intrinsics and external functions
- When to use external functions vs Triton built-ins

What is libdevice?
------------------

NVIDIA's libdevice
~~~~~~~~~~~~~~~~~~

**libdevice** is a library of device-side mathematical functions for CUDA:
- Provided by NVIDIA as part of CUDA toolkit
- Highly optimized for GPU architectures
- Covers special functions not in standard CUDA

**File format**: ``.bc`` (LLVM bitcode)
- ``libdevice.10.bc``: Version for Compute Capability 10+ (Hopper, Blackwell)
- Contains pre-compiled device functions

AMD's Device Libraries
~~~~~~~~~~~~~~~~~~~~~~

AMD provides similar libraries for ROCm:
- **ocml.bc**: OpenCL math library (sin, cos, exp, etc.)
- **ockl.bc**: OpenCL kernel library (atomics, barriers, etc.)

What Functions Are Available?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Examples from libdevice**:
.. code-block:: c

double _*nv*asin(double)      // Arc sine
float  _*nv*asinf(float)       // Arc sine (single precision)
double _*nv*j0(double)         // Bessel function J0
float  _*nv*erfcinvf(float)    // Inverse complementary error function
double _*nv*tgamma(double)     // Gamma function
::


**Full list**: See `CUDA libdevice Users Guide <https://docs.nvidia.com/cuda/libdevice-users-guide/>`_

Why Use External Functions?
---------------------------

Triton Built-in Math
~~~~~~~~~~~~~~~~~~~~

Triton provides common functions:
.. code-block:: python

tl.exp(x)    # Exponential
tl.log(x)    # Natural logarithm
tl.sin(x)    # Sine
tl.cos(x)    # Cosine
tl.sqrt(x)   # Square root
tl.abs(x)    # Absolute value
::


These are fast and convenient!

When You Need More
~~~~~~~~~~~~~~~~~~

For specialized functions:
.. code-block:: python

asin(x)      # Arc sine - NOT in Triton intrinsics!
bessel*j0(x) # Bessel function - NOT available!
erf(x)       # Error function - need libdevice
gamma(x)     # Gamma function - need libdevice
::


**Options**:
1. Implement yourself (complex, error-prone)
2. Use libdevice (optimized, tested, easy)

Using libdevice in Triton
-------------------------

The Simple Way (Default Path)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Triton automatically finds libdevice:

.. code-block:: python

from triton.language.extra import libdevice

@triton.jit
def asin*kernel(x*ptr, y*ptr, n*elements, BLOCK*SIZE: tl.constexpr):
    pid = tl.program*id(axis=0)
    offsets = pid * BLOCK*SIZE + tl.arange(0, BLOCK*SIZE)
    mask = offsets < n*elements

    x = tl.load(x*ptr + offsets, mask=mask)
    y = libdevice.asin(x)  # Call libdevice function!
    tl.store(y*ptr + offsets, y, mask=mask)
::


**Triton handles**:
- Finding libdevice.bc in the installation
- Linking it during compilation
- Type matching (float vs double)

The Custom Path Way
~~~~~~~~~~~~~~~~~~~

Specify library path explicitly:

.. code-block:: python

import triton
from pathlib import Path

Find libdevice in Triton installation
=====================================
triton*dir = Path(triton.**file**).parent
libdir = triton*dir / 'backends/nvidia/lib'
extern*libs = {'libdevice': str(libdir / 'libdevice.10.bc')}

Pass to kernel
==============
asin*kernel`grid <x, y, n*elements, BLOCK*SIZE=1024, extern*libs=extern*libs>`_
::


**When to use custom paths**:
- Testing different library versions
- Using custom-built libraries
- Debugging linking issues

How It Works Under the Hood
---------------------------

Compilation Process
~~~~~~~~~~~~~~~~~~~

1. **Triton → LLVM IR**: Your kernel is compiled to LLVM intermediate representation

2. **Link external libraries**: LLVM linker merges your code with libdevice.bc

3. **Optimize**: LLVM optimizes the combined code

4. **Generate PTX** (NVIDIA) or **AMDGCN** (AMD): Backend-specific assembly

5. **JIT to machine code**: Final GPU binary

Type Dispatch
~~~~~~~~~~~~~

Triton's ``libdevice.asin`` is a wrapper:

.. code-block:: python

def asin(x):
    if x.dtype == tl.float32:
        return _*nv*asinf(x)  # Single precision
    elif x.dtype == tl.float64:
        return _*nv*asin(x)   # Double precision
    # ... etc for other types
::


**Benefit**: You write ``libdevice.asin(x)``, Triton picks the right implementation!

Calling Convention
~~~~~~~~~~~~~~~~~~

Libdevice functions use standard LLVM calling conventions:
- Arguments in registers or stack
- Return value in register
- Triton generates correct call sequences automatically

Available libdevice Functions in Triton
---------------------------------------

Triton's libdevice Wrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~

Located in ``triton/language/extra/libdevice.py``:

.. code-block:: python

from triton.language.extra import libdevice

Trigonometric
=============
libdevice.sin(x)
libdevice.cos(x)
libdevice.tan(x)
libdevice.asin(x)   # Arc sine (our example!)
libdevice.acos(x)   # Arc cosine
libdevice.atan(x)   # Arc tangent

Hyperbolic
==========
libdevice.sinh(x)
libdevice.cosh(x)
libdevice.tanh(x)

Exponential/Logarithmic
=======================
libdevice.exp(x)
libdevice.log(x)
libdevice.pow(x, y)

Special functions
=================
libdevice.erf(x)     # Error function
libdevice.erfc(x)    # Complementary error function
libdevice.j0(x)      # Bessel function J0
libdevice.j1(x)      # Bessel function J1
libdevice.y0(x)      # Bessel function Y0

And many more!
==============
::


**Tip**: Look at ``triton/language/extra/libdevice.py`` in the Triton source for full list.

Example: Arc Sine (asin)
------------------------

The Math
~~~~~~~~

Arc sine is the inverse of sine:
::

If sin(y) = x, then asin(x) = y
Domain: [-1, 1]
Range: [-π/2, π/2]
::


Why Use libdevice?
~~~~~~~~~~~~~~~~~~

**Implementing asin yourself**:
.. code-block:: python

def asin*approx(x):
    # Use Taylor series or other approximation
    # Need to handle edge cases (x near ±1)
    # Accuracy vs performance trade-off
    # Lots of potential bugs!
::


**Using libdevice**:
.. code-block:: python

y = libdevice.asin(x)  # Done! Optimized, accurate, tested
::


Full Example
~~~~~~~~~~~~

.. code-block:: python

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def asin*kernel(x*ptr, y*ptr, n*elements, BLOCK*SIZE: tl.constexpr):
    pid = tl.program*id(axis=0)
    block*start = pid * BLOCK*SIZE
    offsets = block*start + tl.arange(0, BLOCK*SIZE)
    mask = offsets < n*elements

    # Load input
    x = tl.load(x*ptr + offsets, mask=mask)

    # Apply arc sine using libdevice
    y = libdevice.asin(x)

    # Store result
    tl.store(y*ptr + offsets, y, mask=mask)

def asin*triton(x):
    output = torch.empty*like(x)
    n*elements = x.numel()
    grid = lambda meta: (triton.cdiv(n*elements, meta['BLOCK*SIZE']),)
    asin*kernel`grid <x, output, n*elements, BLOCK*SIZE=1024>`_
    return output

Test
====
x = torch.rand(100000, device='cuda')  # Random values in [0, 1]
y*torch = torch.asin(x)
y*triton = asin*triton(x)

Should match!
=============
max*diff = torch.max(torch.abs(y*torch - y*triton))
print(f'Max difference: {max*diff}')  # Should be ~1e-7 (float precision)
::


Performance Considerations
--------------------------

Libdevice Performance
~~~~~~~~~~~~~~~~~~~~~

**Generally very good**:
- Hand-optimized by NVIDIA/AMD
- Uses GPU-specific instructions
- Often as fast as or faster than DIY implementations

**Example timings** (1M elements):
- ``libdevice.asin``: ~0.05 ms
- Custom approximation: ~0.03-0.10 ms (depends on accuracy)
- Benefit: Guaranteed correctness + good performance

When to Use vs Triton Intrinsics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use Triton intrinsics when available**:
.. code-block:: python

tl.exp(x)   # Prefer this over libdevice.exp
tl.sin(x)   # Prefer this over libdevice.sin
::


Triton intrinsics may have special optimizations.

**Use libdevice for**:
- Functions not in Triton (asin, bessel, erf, etc.)
- When you need specific precision guarantees
- Complex mathematical operations

Linking Multiple Libraries (AMD Example)
----------------------------------------

For AMD GPUs, need both ocml and ockl:

.. code-block:: python

from pathlib import Path
import triton

if is*hip():
    triton*dir = Path(triton.**file**).parent
    libdir = triton*dir / 'backends/amd/lib'

    extern*libs = {
        'ocml': str(libdir / 'ocml.bc'),
        'ockl': str(libdir / 'ockl.bc')
    }

    my*kernel`grid <..., extern*libs=extern*libs>`_
::


**Why multiple libraries?**
- Different functions in different libraries
- OCML: Math functions
- OCKL: Kernel utilities (barriers, atomics, etc.)

Debugging Linking Issues
------------------------

Common Errors
~~~~~~~~~~~~~

**1. Library Not Found**
::

FileNotFoundError: [Errno 2] No such file or directory: '/path/to/libdevice.10.bc'
::


**Solution**: Check library path
.. code-block:: python

libdir = Path(triton._*file**).parent / 'backends/nvidia/lib'
print(libdir.exists())
print(list(libdir.glob('*.bc')))
::


**2. Undefined Symbol**
::

Error: undefined reference to '**nv*asin'
::


**Solution**: Make sure library is in ``extern*libs`` dict

**3. Type Mismatch**
::

Error: cannot convert 'float' to 'double'
::


**Solution**: Ensure input types match function expectations
.. code-block:: python

x = x.to(tl.float32)  # Explicitly cast if needed
::


Verifying Libraries
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

from pathlib import Path
import triton

triton*dir = Path(triton._*file**).parent

NVIDIA
======
nvidia*lib = triton*dir / 'backends/nvidia/lib/libdevice.10.bc'
print(f'NVIDIA libdevice: {nvidia*lib.exists()}')

AMD
===
amd*lib = triton*dir / 'backends/amd/lib/ocml.bc'
print(f'AMD ocml: {amd*lib.exists()}')
::


Advanced Usage
--------------

Calling Custom External Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can link your own LLVM bitcode:

**1. Write CUDA device function**:
.. code-block:: cuda

// my*func.cu
extern "C" _*device** float my*special*func(float x) {
    return x * x + 2.0f * x + 1.0f;
}
::


**2. Compile to bitcode**:
.. code-block:: bash

clang++ --cuda-device-only -emit-llvm -c my*func.cu -o my*func.bc
::


**3. Link in Triton**:
.. code-block:: python

extern*libs = {'my*lib': '/path/to/my*func.bc'}
my*kernel`grid <..., extern*libs=extern*libs>`_
::


**4. Declare in Triton**:
.. code-block:: python

@triton.jit
def my*kernel(...):
    # Declare external function
    my*special*func = tl.extern*func('my*special*func', tl.float32, [tl.float32])

    # Use it
    y = my*special*func(x)
::


Mixing Multiple External Libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

extern*libs = {
    'libdevice': '/path/to/libdevice.10.bc',
    'my*lib': '/path/to/my*func.bc',
    'another*lib': '/path/to/another.bc'
}

my*kernel`grid <..., extern*libs=extern*libs>`_
::


All libraries are linked together during compilation.

Portability Considerations
--------------------------

NVIDIA vs AMD
~~~~~~~~~~~~~

.. code-block:: python

def get*extern*libs():
    triton*dir = Path(triton.**file**).parent

    if is*cuda():
        return {
            'libdevice': str(triton*dir / 'backends/nvidia/lib/libdevice.10.bc')
        }
    elif is*hip():
        libdir = triton*dir / 'backends/amd/lib'
        return {
            'ocml': str(libdir / 'ocml.bc'),
            'ockl': str(libdir / 'ockl.bc')
        }
    else:
        return {}

extern*libs = get*extern*libs()
kernel`grid <..., extern*libs=extern*libs>`_
::


Function Name Differences
~~~~~~~~~~~~~~~~~~~~~~~~~

Some functions have different names:
- NVIDIA: ``_*nv*sinf``
- AMD: ``_*ocml*sin*f32``

**Triton's libdevice wrapper handles this automatically!**

Comparison to CUDA
------------------

In CUDA
~~~~~~~

.. code-block:: cuda

#include <math*functions.h>

_*global** void asin*kernel(float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = asinf(x[i]);  // Automatically linked
    }
}
::


- Include header
- Use function
- nvcc handles linking

In Triton
~~~~~~~~~

.. code-block:: python

from triton.language.extra import libdevice

@triton.jit
def asin*kernel(x*ptr, y*ptr, n*elements, BLOCK_SIZE: tl.constexpr):
    # ...
    y = libdevice.asin(x)  # Import wrapper, use function
::


**Very similar experience!**

Key Takeaways
-------------

1. **libdevice provides optimized math functions**: Beyond Triton's built-ins
2. **Easy to use**: ``from triton.language.extra import libdevice``
3. **Automatic type dispatch**: Triton picks float vs double version
4. **Custom paths available**: For advanced use cases
5. **Good performance**: NVIDIA/AMD optimized implementations
6. **Portability**: Triton handles differences between backends
7. **Extensible**: Can link your own LLVM bitcode
8. **Essential for complex math**: Bessel functions, error functions, etc.

External functions let you leverage decades of numerical optimization work while writing simple Triton code!
