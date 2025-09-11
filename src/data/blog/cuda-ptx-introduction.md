---
title: A Gentle Introduction to CUDA PTX
description: A gentle introduction to the PTX ISA. This post explains the entire CUDA compilation pipeline from C++ to SASS, provides a PTX playground and fully explains a hand-written PTX kernel.
pubDatetime: 2025-09-11T19:42:16Z
modDatetime:
slug: cuda-ptx-introduction
featured: true
tags:
  - PTX
  - CUDA
  - Assembly
---

## Introduction

As a CUDA developer, you might not interact with Parallel Thread Execution (PTX) every day, but it is the fundamental layer between your CUDA code and the hardware. Understanding it is essential for deep performance analysis and for accessing the latest hardware features, sometimes long before they are exposed in C++. For example, the [`wgmma`](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions) instructions, which perform warpgroup-level matrix operations and are used in some of the most performant GEMM kernels, are available only through PTX instructions.

This post serves as a gentle introduction to PTX and its place in the CUDA ecosystem. We will set up a simple playground environment and walk through a complete kernel in PTX. My goal is not only to give you the foundation to use PTX but to also share my mental model of how PTX fits into the CUDA landscape.

## PTX and the CUDA ecosystem

Every processor has an instruction set architecture (ISA), which is the specific set of commands the hardware can execute. NVIDIA GPUs are no different, their native, hardware-specific ISA is called SASS (streaming assembly). However, the SASS for one GPU generation can be incompatible with another, meaning a program compiled for an older GPU might not run on a newer one. In other words, there is no forward compatibility. This is one of the problems that PTX solves. PTX is an ISA for a virtual machine: an abstract GPU that represents the common features of all NVIDIA hardware. When you compile your CUDA code, a tool called [`ptxas`](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#ptxas-options) translates your hardware-agnostic PTX into the specific SASS for your target GPU. This two-stage design is a common pattern in modern compilers. The [LLVM (Low Level Virtual Machine) project](https://llvm.org/) is a well-known example of this architecture.

We can utilize the PTX forward compatibility by using just-in-time (JIT) compilation. You can choose to embed the PTX code directly into your final executable (I will cover this [later](#appendix-a-controlling-the-fatbin-with-nvcc) in the post). When your application runs on a new GPU for which it doesn't have pre-compiled SASS, the NVIDIA driver on the system acts as a JIT compiler. It's important to note that this provides forward compatibility only. For example, PTX generated for `compute_70` can run on any future GPU (8.x, 9.x, etc.), but it cannot be run on an older 6.x GPU. This is different from the SASS binary itself, which has much stricter rules and is generally only compatible with GPUs of the same major version number. Tools like [Triton](https://github.com/triton-lang/triton) rely on this. They generate PTX and leave the final, hardware-specific compilation to the driver. By default, `nvcc` includes both PTX and SASS in your executable, giving you both immediate performance and future compatibility.

## The PTX playground

The best way to learn PTX is to see it in action. To do that, we need a simple environment that lets us write PTX code and see it run. I have created precisely that in [this repository](https://github.com/philipfabianek/ptx-playground).

The repository contains two main files:

1.  [`add_kernel.ptx`](https://github.com/philipfabianek/ptx-playground/blob/main/add_kernel.ptx): This is the text file that contains the raw PTX instructions for our kernel.
2.  [`main.cu`](https://github.com/philipfabianek/ptx-playground/blob/main/main.cu): This is a C++ program that runs on the host to load, run, and verify the result of our PTX kernel.

Our C++ host code needs a way to load and run the .ptx file at runtime. This requires the [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/index.html), a lower-level interface than the more common Runtime API. This is why the code in main.cu uses functions like `cuLaunchKernel` instead of the familiar `<<<...>>>` syntax. While a full explanation of the Driver API is outside the scope of this post, the `main.cu` file handles all the necessary setup for you, allowing us to focus entirely on the PTX code.

To compile the program, use the following command:

```bash
nvcc main.cu -o ptx_runner -lcuda
```

_(notice the -lcuda which ensures the CUDA Driver API library is included)_

Then, run the executable:

```bash
./ptx_runner
```

If everything works correctly, you should see a success message as the output.

## Kernel walkthrough

Now that we have a working environment, we can dive into the PTX kernel itself. Our goal is to implement a classic kernel that performs an element-wise addition of two vectors, `a` and `b`, and stores the result in a third vector, `c`. Before we look at the assembly, let's first look at the equivalent logic in standard CUDA C++:

```cpp
__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		c[idx] = a[idx] + b[idx];
	}
}
```

Here is the complete `add_kernel.ptx` file that accomplishes this exact same task. Don't worry if you don't understand what is happening in this file, by the end of this post that will change.

```cpp
.version 7.0
.target sm_70
.address_size 64

.visible .entry add_kernel (
    // Kernel parameters passed from the host
    .param .u64 in_a,
    .param .u64 in_b,
    .param .u64 out_c,
    .param .u32 n
)
{
    // Setup registers
    .reg .b64   %rd<8>;                  // %rd for 64-bit addresses and pointers
    .reg .b32   %r<2>;                   // %r for signed 32-bit integers
    .reg .u32   %u<5>;                   // %u for unsigned 32-bit integers
    .reg .f32   %f<4>;                   // %f for 32-bit floating point values
    .reg .pred  %p<2>;                   // %p for predicates (booleans)

    // Load kernel parameters into registers
    ld.param.u64    %rd1, [in_a];        // %rd1 = base pointer for 'a'
    ld.param.u64    %rd2, [in_b];        // %rd2 = base pointer for 'b'
    ld.param.u64    %rd3, [out_c];       // %rd3 = base pointer for 'c'
    ld.param.u32    %u1, [n];            // %u1 = size 'n'

    // Move special register values to general registers to use them
    mov.u32         %u2, %ctaid.x;       // %u2 = blockIdx.x
    mov.u32         %u3, %ntid.x;        // %u3 = blockDim.x
    mov.u32         %u4, %tid.x;         // %u4 = threadIdx.x

    // Calculate the global thread ID by
    // idx = blockIdx.x * blockDim.x + threadIdx.x
    mad.lo.s32      %r1, %u2, %u3, %u4;  // %r1 = idx

    // Terminate if idx >= n
    setp.ge.s32     %p1, %r1, %u1;       // Set predicate %p1 if idx >= n
    @%p1 bra        DONE;                // Branch to DONE if %p1 is true

    // Compute how many bytes to offset for the current index,
    // use 4 because we're dealing with 32-bit (4-byte) floats
    mul.wide.s32    %rd4, %r1, 4;        // %rd4 = byte offset

    // Calculate addresses for a[idx], b[idx], and c[idx]
    // by using the base address + offset
    add.s64         %rd5, %rd1, %rd4;    // %rd5 = address of a[idx]
    add.s64         %rd6, %rd2, %rd4;    // %rd6 = address of b[idx]
    add.s64         %rd7, %rd3, %rd4;    // %rd7 = address of c[idx]

    // Load vector values at the target position
    ld.global.f32   %f1, [%rd5];         // %f1 = a[idx]
    ld.global.f32   %f2, [%rd6];         // %f2 = b[idx]

    // Perform the addition and store the result
    add.f32         %f3, %f1, %f2;       // %f3 = a[idx] + b[idx]
    st.global.f32   [%rd7], %f3;         // c[idx] = %f3

DONE:
    ret;
}
```

_(You can also compare this with how `nvcc` compiles the C++ version of our kernel into PTX on the [Compiler Explorer (Godbolt)](https://godbolt.org/z/a56drqdYT))_

### Preamble and signature

The first thing to notice is that the .ptx file contains only the code for our GPU kernel. The host code you saw in `main.cu` is compiled separately by `nvcc` and then linked together with the GPU binary at the end.

Let's look at the first part of the file, which defines the kernel's properties and the arguments it expects:

```cpp
.version 7.0
.target sm_70
.address_size 64

.visible .entry add_kernel (
	// Kernel parameters passed from the host
	.param .u64 in_a,
	.param .u64 in_b,
	.param .u64 out_c,
	.param .u32 n
)
```

The file begins with a preamble that sets up the compilation environment using several directives. A directive is a command for the PTX assembler that starts with a dot (.). The `.version` directive specifies the PTX language version we are using. The `.target` directive declares the minimum GPU architecture required, which in our case is `sm_70`. Thanks to PTX's forward compatibility, this means our kernel will run correctly on any NVIDIA GPU from the Volta generation (`sm_70`) or newer. Finally, `.address_size` confirms that we are working with 64-bit pointers.

Following this is the kernel's signature. The `.visible` and `.entry` keywords declare that this is a kernel that can be seen and launched by the host. The name of our kernel, `add_kernel`, is what we reference from our C++ code. When `nvcc` compiles a C++ kernel, it normally generates a "mangled" name that encodes the function's arguments, resulting in a long, unreadable string like `_Z10add_kernelPKfS0_Pfi`. Since we are writing the PTX by hand, we can choose a simple name for clarity.

Inside the parentheses are the kernel's parameters, declared with the `.param` directive. These must match the order and type of the arguments we pass from the host. We use `.u64` for the pointers, as they are 64-bit addresses, and `.u32` for the size of the vectors.

### Register declarations

After the kernel signature, we define the virtual registers that will serve as our working variables.

Registers are a small amount of extremely fast, on-chip storage. For a processor to perform any computation, such as adding two numbers, the data must first be loaded from main memory into these registers. After that, the processor does the work and stores the results from the registers back to memory.

Our kernel declares a pool of registers using the `.reg` directive:

```cpp
// Setup registers
.reg .b64   %rd<8>;      // %rd for 64-bit addresses and pointers
.reg .b32   %r<2>;       // %r for signed 32-bit integers
.reg .u32   %u<5>;       // %u for unsigned 32-bit integers
.reg .f32   %f<4>;       // %f for 32-bit floating point values
.reg .pred  %p<2>;       // %p for predicates (booleans)
```

The prefixes like `%rd`, `%r`, and `%u` are not rules but a common convention that makes the code much easier to read by signaling the intended use of a register.

The `<N>` syntax declares a count of N registers, which are indexed from 0 to N-1. For example, `.reg .pred %p<2>` declares two predicate registers: `%p0` and `%p1`.

You might notice that we have declared more registers than we actually use, and we don't always start our indexing at 0. This is perfectly fine. The final executable binary will only allocate physical hardware registers for the virtual registers that are part of the program's logic.

### Data movement instructions

After the register declarations, we get to the main body of our kernel, which consists of a sequence of instructions. In PTX, an instruction is an operation that the GPU will execute, like loading data or performing addition. They all follow a similar pattern:

```cpp
opcode{.modifier} destination_operand, source_operand_A, source_operand_B, ...;
```

The `opcode` is the name of the operation (e.g. `ld` for load). This is often followed by one or more `.modifiers` that specify the data type and other options. The destination operand is almost always a register where the result will be stored, and it is followed by one or more source operands. Keep in mind the output is always before the input in these instructions.

Let's start by looking at the core data movement instructions in our kernel: `ld`, `st`, and `mov`.

#### [`ld`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld) (Load): Reading from memory

The `ld` instruction is used to load data from a memory location into a register. If you look at the official PTX documentation, the full syntax for `ld` is quite complex, with many optional modifiers for controlling caching, memory synchronization, and more:

```cpp
ld{.weak}{.ss}{...}.type d, [a]{...};
```

While this looks intimidating, the vast majority of these modifiers are for advanced usage. For our purposes, we can simplify this down to a form that we will actually be using:

```cpp
ld{.ss}.type d, [a];
```

Here's how you can read this:

- `ld`: The opcode.
- `{.ss}`: The curly braces `{}` mean this part is an optional modifier. The `.ss` stands for "state space" and is where you would put `.global`, `.shared`, or `.param` to tell the instruction where to load from.
- `.type`: This is a mandatory modifier where you specify the data type, like `.f32` or `.u64`.
- `d`: The destination operand, which must be a register.
- `[a]`: The source operand. The square brackets `[]` mean this is a memory address. The `a` inside is a register that holds the address. This syntax means "dereference the pointer in register `a`."

Now, let's look at the `ld` instructions from our kernel. The first ones load the kernel parameters passed from the host into our general-purpose registers:

```cpp
// Load kernel parameters into registers
ld.param.u64    %rd1, [in_a];        // %rd1 = base pointer for 'a'
ld.param.u64    %rd2, [in_b];        // %rd2 = base pointer for 'b'
ld.param.u64    %rd3, [out_c];       // %rd3 = base pointer for 'c'
ld.param.u32    %u1, [n];            // %u1 = size 'n'
```

Here you can see the pattern in action. For the first instruction, we are loading from the `.param` state space, the data type is a `.u64`, the destination is the register `%rd1`, and the source address is the one associated with the parameter `in_a`.

#### [`st`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st) (Store): Writing to memory

The `st` instruction is the mirror image of `ld`. It is used to store data from a register to a memory location. Its simplified syntax is very similar, but the order of the operands is reversed:

```cpp
st{.ss}.type [a], b;
```

Here, the memory address `[a]` is the destination, and the register `b` is the source. Our kernel uses `st` at the very end to write the final result back to global memory:

```cpp
// Perform the addition and store the result
add.f32         %f3, %f1, %f2;       // %f3 = a[idx] + b[idx]
st.global.f32   [%rd7], %f3;         // c[idx] = %f3
```

This instruction stores the 32-bit float (`.f32`) value from the source register `%f3` into the destination address `[%rd7]` in global (`.global`) memory.

#### [`mov`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mov) (Move): Working with registers

Finally, the `mov` instruction is used to move data between registers or to get the address of a variable. It has a much simpler syntax:

```cpp
mov.type d, a;
```

Our kernel uses mov to copy the values from special registers (see below) into the general-purpose registers that our other instructions can use.

```cpp
// Move special register values to general registers to use them
mov.u32         %u2, %ctaid.x;       // %u2 = blockIdx.x
mov.u32         %u3, %ntid.x;        // %u3 = blockDim.x
mov.u32         %u4, %tid.x;         // %u4 = threadIdx.x
```

[Special registers](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers) (in this example `%ctaid.x`, `%ntid.x`, and `%tid.x`) are predefined, read-only registers that the GPU uses to give our kernel information about its environment. They live in a special state space called `.sreg`. As a programmer, you don't declare them, you simply read from them to get essential values like the thread's ID or the dimensions of the grid.

In PTX, these special registers have direct analogs to the built-in variables you would use in a CUDA C++ kernel:

| PTX Special Register | CUDA C++ Built-in Variable |
| -------------------- | -------------------------- |
| %tid.x               | threadIdx.x                |
| %ntid.x              | blockDim.x                 |
| %ctaid.x             | blockIdx.x                 |
| %nctaid.x            | gridDim.x                  |

Most arithmetic instructions cannot use these special registers directly. This is why we first use `mov` to copy these values into our general-purpose registers (`%u2`, `%u3`, and `%u4`), where we can then use them for our address calculations.

### Computation and control flow

Now that we have loaded our parameters and identified our thread, we can move on to the core logic of the kernel. This involves calculating the thread's index, checking if it's within the bounds of our vectors, and finally, performing the addition.

The first step is to calculate the global index for the current thread, which corresponds to the following C++ line:

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

PTX has a dedicated instruction for this common pattern:

```cpp
// Calculate the global thread ID by
// idx = blockIdx.x * blockDim.x + threadIdx.x
mad.lo.s32      %r1, %u2, %u3, %u4;  // %r1 = idx
```

The [`mad`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-mad) instruction performs a multiply-add operation. It multiplies the first two source operands (`%u2` and `%u3`) and adds the third (`%u4`), storing the result in the destination (`%r1`). When two 32-bit numbers are multiplied, the full result can be up to 64 bits long. Since we know our thread index will not exceed the 32-bit limit, the `.lo` modifier tells the instruction to take only the lower 32 bits of the full multiplication result before performing the addition. The `.s32` modifier tells the instruction to treat the operands as 32-bit signed integers.

Next, we need to implement the boundary check from our C++ code:

```cpp
if (idx < n)
```

In assembly, this is typically done by checking for the opposite condition and skipping the work if it's met:

```cpp
// Terminate if idx >= n
setp.ge.s32     %p1, %r1, %u1;       // Set predicate %p1 if idx >= n
@%p1 bra        DONE;                // Branch to DONE if %p1 is true
```

First, the [`setp`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp) (set predicate) instruction performs a comparison. Here, it compares `%r1` (the index) with `%u1` (the vector size). The `.ge` modifier means "greater than or equal to." If `idx >= n`, the 1-bit predicate register `%p1` is set to `true`.

The next instruction, [`bra`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-bra) (branch), is a jump to a label. The `@%p1` at the beginning is a predicate guard, meaning the instruction only executes if `%p1` is true. Together, these two lines mean: "If idx is greater than or equal to n, jump to the `DONE` label at the end of the kernel." This is how we skip the main logic for out-of-bounds threads.

Before we can load our data, we need to calculate the exact memory address for `a[idx]`, `b[idx]`, and `c[idx]`. This requires converting our logical index into a physical byte offset:

```cpp
// Compute how many bytes to offset for the current index,
// use 4 because we're dealing with 32-bit (4-byte) floats
mul.wide.s32    %rd4, %r1, 4;        // %rd4 = byte offset
```

This instruction multiplies our index in `%r1` by 4, because each float takes up 4 bytes in memory. We use [`mul.wide.s32`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-mul), which takes two 32-bit inputs but produces a full 64-bit result, preventing any potential overflow. The result, our byte offset, is stored in `%rd4`.

Then, we simply use the integer [add](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-add) instruction to add this 64-bit offset to our 64-bit base pointers (`%rd1`, `%rd2`, and `%rd3`) to get the final addresses for the elements we need to access:

```cpp
// Calculate addresses for a[idx], b[idx], and c[idx]
// by using the base address + offset
add.s64         %rd5, %rd1, %rd4;    // %rd5 = address of a[idx]
add.s64         %rd6, %rd2, %rd4;    // %rd6 = address of b[idx]
add.s64         %rd7, %rd3, %rd4;    // %rd7 = address of c[idx]
```

Finally, with all our addresses calculated and data loaded, we can perform the core task of the kernel:

```cpp
// Load vector values at the target position
ld.global.f32   %f1, [%rd5];         // %f1 = a[idx]
ld.global.f32   %f2, [%rd6];         // %f2 = b[idx]

// Perform the addition and store the result
add.f32         %f3, %f1, %f2;       // %f3 = a[idx] + b[idx]
```

This is the simplest part. The float [`add.f32`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-add) instruction takes the two float values we loaded into registers `%f1` and `%f2` and adds them, storing the final result in `%f3`.

The last two lines of our kernel handle the exit path:

```cpp
DONE:
    ret;
```

`DONE` is a label. It is not an instruction, but a bookmark in the code. Its only purpose is to serve as a target for the `bra` (branch) instruction we saw earlier. Threads that are out-of-bounds will jump directly to this line.

The [`ret`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-ret) instruction is the final command. It returns from the kernel, ending the execution for the current thread. All threads, whether they performed the addition or branched to `DONE`, will eventually execute this instruction.

This completes the walkthrough of all the instructions in our kernel.

## Conclusion

Congratulations! If you reached this point, you should now have a solid foundation and a practical mental model of how PTX works, from the virtual machine concepts down to the individual instructions.

While the workflow we used (using a full .ptx file as a kernel) is the best way to learn the fundamentals, in practice, the most common way you will use PTX is through inline assembly directly inside your CUDA `__global__` functions. You can learn more about inline PTX [here](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html). This technique is used for specific optimizations, allowing you to inject a few specific PTX instructions to perform a task that C++ cannot express. This is exactly how you would use instructions like the `wgmma` operations we mentioned in the introduction.

You should now possess the knowledge to navigate the dense [official PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html). It is best to use it as a reference manual. When you need to know the exact syntax of a specific instruction or explore a new hardware feature, you can read the documentation and understand the instructions.

## Appendix A: Controlling the `fatbin` with `nvcc`

To have precise control over what gets embedded in your final executable (called `fatbin`), you need to understand the terminology used by `nvcc`. The target for the PTX virtual machine is called a compute capability, specified with flags like `compute_70`. The target for the final, hardware-specific SASS is the streaming multiprocessor (SM) architecture, specified with `sm_70`.

The `-arch` flag used with `sm_XX` (e.g. `-arch sm_86`) results in both PTX and SASS for that specific architecture being included. If you use `-arch compute_86`, then only PTX will be included. If you want to specify exactly what goes into the binary, you can use the more powerful `-gencode` flag. For instance, you could ship an application with pre-compiled SASS (and no PTX) for both Ampere (`sm_86`) and Hopper (`sm_90`) by using this command:

```bash
nvcc program.cu \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_90,code=sm_90
```

By including multiple SASS files you can prevent JIT compilation which can for example eliminate startup latency.

To see what's inside your final executable, you can use the [`cuobjdump`](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#cuobjdump) utility. Running `cuobjdump -ptx <executable>` will extract and print the embedded PTX, while `cuobjdump -sass <executable>` will disassemble and show you the native SASS machine code for each architecture it contains. While you can inspect these to make performance adjustments, more commonly you generate a report using [Nsight Compute (NCU)](https://developer.nvidia.com/nsight-compute) and inspect the PTX and SASS there. The advantage is that NCU can link specific lines from your kernel code to their compiled PTX and SASS representations, making performance analysis easier. For interactive exploration, a fantastic online tool is the previously mentioned [Compiler Explorer (Godbolt)](https://godbolt.org/), which can show you the PTX or SASS generated by `nvcc` in real-time as you type CUDA C++ code.

## Appendix B: The full compilation pipeline and NVVM IR

As we've seen, CUDA C++ is compiled to PTX, and this PTX is then assembled into SASS machine code for a specific GPU. However, there is one more step in this process: the [NVVM IR](https://docs.nvidia.com/cuda/nvvm-ir-spec/). This is another internal representation that sits between your C++ code and the final PTX. NVVM IR is NVIDIA's specialized version of the popular LLVM IR, and it's converted to PTX using an NVIDIA library called `libnvvm`.

So what is the purpose of this extra layer? By building their compiler on LLVM, NVIDIA makes it much easier for other programming languages to target their GPUs. If someone wants to write a compiler for a new language that runs on NVIDIA hardware, they don't need to become experts in generating PTX. Instead, they can target the much higher-level NVVM IR, allowing them to utilize the entire mature LLVM infrastructure. This is exactly how tools like [Triton](https://github.com/triton-lang/triton) and the [Rust GPU compiler](https://github.com/rust-gpu/rust-gpu) work.

This translation from CUDA C++ to NVVM IR is handled by a Clang-based C++ frontend used by `nvcc`. However, `nvcc` treats this NVVM IR as a purely internal representation. This makes PTX the first stage in the compilation pipeline where we can reliably inspect the output, making it an essential tool for performance analysis.
