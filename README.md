# CUDA CNN Kernels

**FP32 matrix multiply and 2D convolution written from scratch in CUDA C++, optimized step by step and benchmarked against cuBLAS and cuDNN on the layer shapes of a ResNet-34 U-Net.**

![CI](https://github.com/Dhruvsanghavi07/cuda-cnn-kernels/actions/workflows/ci.yml/badge.svg)

Convolutions and the matrix multiplies behind them are most of the inference time of a segmentation network. This project rebuilds them from the naive version up, to show where GPU time actually goes: global-memory bandwidth, then shared-memory bandwidth, then register use.

## Kernels

| Kernel | Idea | What it fixes |
|---|---|---|
| `gemm` naive | One thread per output; reads A and B straight from global memory | Baseline: two global loads per multiply-add, so it is memory-bound |
| `gemm` shared-memory tiled | 32×32 tiles of A and B staged in shared memory, reused 32 times | Cuts global-memory traffic ~32×; padded shared array avoids bank conflicts |
| `gemm` register-blocked | 64×64 block tile, 8-deep K slices; each thread keeps a 4×4 patch of C in registers | 16 multiply-adds per 8 shared-memory reads; A stored transposed so reads are contiguous |
| `conv2d` direct | One thread per output pixel, loops over C×R×S | Simple and memory-light; re-reads inputs many times |
| `conv2d` im2col + GEMM | Lowers patches to a `[C·R·S × OH·OW]` matrix, then one register-blocked GEMM | Turns convolution into the fastest kernel above; bias and ReLU fused into the GEMM epilogue |

All kernels handle any size (edges are zero-filled in shared memory), and every one is checked against a CPU reference that accumulates in double precision. Built for sm_75 (clang 18 to PTX, then NVIDIA ptxas 12.9), no kernel spills to local memory; the register-blocked GEMM uses 58 registers per thread and 4 KB of shared memory per block. CI prints nvcc's numbers on every push.

## Results

Run `notebooks/colab_benchmark.ipynb` on a free Colab T4 (or any NVIDIA GPU). It builds the project, runs the correctness tests, and writes a results file to `results/` with:

- GEMM GFLOP/s at 256–4096 for each kernel, and the percentage of cuBLAS it reaches
- time per convolution layer for direct vs im2col + GEMM, for seven ResNet-34 U-Net layers (7×7 stem through 512-channel 8×8)
- the same layers in PyTorch with cuDNN, as the reference to compare against

<!-- Paste the tables from results/bench_<gpu>.md here after running the notebook. -->

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # add -DCMAKE_CUDA_ARCHITECTURES=native for your GPU
cmake --build build -j
./build/test_reference      # CPU reference tests: work on any machine
./build/test_kernels        # every GPU kernel vs the reference on 12 GEMM and 9 conv shapes
./build/bench --out results/bench_mygpu.md
```

Without a CUDA compiler, CMake builds only the CPU reference and its tests (`-DWITH_CUDA=OFF` forces this). CI runs the CPU tests and compiles the CUDA code in NVIDIA's CUDA 12.6 container, printing register and spill counts per kernel; hosted runners have no GPU, so GPU tests run in Colab.

## Layout

```
include/reference.h    ConvShape + CPU reference GEMM, conv2d, im2col
include/kernels.h      GPU API
src/gemm.cu            naive, shared-memory tiled and register-blocked GEMM, fused bias + ReLU
src/conv.cu            direct convolution, im2col kernel, im2col + GEMM convolution
tests/                 CPU reference tests (hand-computed values) and GPU-vs-reference tests
bench/bench.cu         CUDA-event timing, cuBLAS comparison, Markdown output
notebooks/             Colab notebook: build, test, benchmark, cuDNN comparison
```

## Testing approach

- The CPU reference is checked against hand-computed values (2×3 · 3×2 GEMM; 3×3 input with a 2×2 kernel, with and without padding) and checks itself: im2col + GEMM must equal direct convolution for odd sizes, stride 2, a 7×7 stem and 1×1 kernels.
- GPU tests fill output buffers with NaN before each launch, so an output element a kernel forgets to write fails the test instead of passing on leftover memory.
- Shapes include sizes that are not multiples of any tile (65×63×66, 1×257×129, 33×1000×17) to exercise every boundary check.

## Next steps

- Vectorized `float4` loads and double-buffered shared memory to overlap loads with math
- Implicit GEMM convolution (no im2col buffer), and Tensor Core (WMMA) versions in FP16/TF32
- Profile with Nsight Compute and report achieved memory bandwidth and occupancy per kernel

## License

MIT
