"""Lecture 004: 用同一组例子理解 tiling 和分块 attention。

无需第三方依赖：python3 tiling_demo.py --cpu
需要 PyTorch、NVIDIA GPU、CUDA Toolkit：python3 tiling_demo.py
配套讲解：STUDY_GUIDE.zh-CN.md。这里的 CUDA kernel 是教学实现。
"""

import argparse
import math
import random


def reference_matmul(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)]
            for row in a]


def tiled_matmul_cpu(a, b, tile, trace=False):
    """按 CUDA 的阶段顺序模拟整个 block；统计有效输入元素的读取次数。"""
    m, k, n = len(a), len(b), len(b[0])
    out = [[0 for _ in range(n)] for _ in range(m)]
    loads = 0
    for row0 in range(0, m, tile):
        for col0 in range(0, n, tile):
            acc = [[0 for _ in range(tile)] for _ in range(tile)]
            for k0 in range(0, k, tile):
                at = [[0 for _ in range(tile)] for _ in range(tile)]
                bt = [[0 for _ in range(tile)] for _ in range(tile)]
                # 合作搬运：先填好整块，对应第一处 __syncthreads() 前的工作。
                for ty in range(tile):
                    for tx in range(tile):
                        if row0 + ty < m and k0 + tx < k:
                            at[ty][tx] = a[row0 + ty][k0 + tx]
                            loads += 1
                        if k0 + ty < k and col0 + tx < n:
                            bt[ty][tx] = b[k0 + ty][col0 + tx]
                            loads += 1
                delta = reference_matmul(at, bt)
                for ty in range(tile):
                    for tx in range(tile):
                        acc[ty][tx] += delta[ty][tx]
                if trace and row0 == 0 and col0 == 0:
                    print(f"k={k0}..{min(k0 + tile, k) - 1}: "
                          f"本轮贡献={delta}, 累计={acc}")
                # 整块计算完才进入下一轮，对应第二处 __syncthreads()。
            for ty in range(tile):
                for tx in range(tile):
                    if row0 + ty < m and col0 + tx < n:
                        out[row0 + ty][col0 + tx] = acc[ty][tx]
    return out, loads


def online_attention_row(scores, values, tile=2):
    """只演示一行、标量 V 的稳定分块 softmax 加权和，不是完整 FlashAttention。"""
    running_max, denominator, numerator = -math.inf, 0.0, 0.0
    for start in range(0, len(scores), tile):
        block = scores[start:start + tile]
        new_max = max(running_max, max(block))
        scale = math.exp(running_max - new_max)
        weights = [math.exp(s - new_max) for s in block]
        denominator = scale * denominator + sum(weights)
        numerator = scale * numerator + sum(
            w * v for w, v in zip(weights, values[start:start + tile]))
        running_max = new_max
    return numerator / denominator


def run_cpu():
    a = [[1, 2, 3, 4], [5, 6, 7, 8],
         [9, 10, 11, 12], [13, 14, 15, 16]]
    b = [[1, 0, 2, 1], [0, 1, 3, 2], [4, 0, 1, 0], [0, 5, 0, 1]]
    out, loads = tiled_matmul_cpu(a, b, tile=2, trace=True)
    assert out == reference_matmul(a, b)
    assert [row[:2] for row in out[:2]] == [[13, 22], [33, 46]]
    assert loads == 64
    print(f"整个 4x4 乘法：朴素逻辑输入读取={2 * 4**3}，tiling={loads}")

    rng = random.Random(0)
    for m, n, k in [(1, 1, 1), (3, 5, 7), (5, 3, 2), (4, 4, 4)]:
        a = [[rng.randrange(-3, 4) for _ in range(k)] for _ in range(m)]
        b = [[rng.randrange(-3, 4) for _ in range(n)] for _ in range(k)]
        for tile in (2, 3, 4):
            actual, _ = tiled_matmul_cpu(a, b, tile)
            assert actual == reference_matmul(a, b), (m, n, k, tile)
    print("CPU 分块模拟：矩形、边界补零、不同 tile 的结果均正确。")

    scores = [math.log(x) for x in (1, 2, 3, 4)]
    values = [10, 20, 30, 40]
    result = online_attention_row(scores, values)
    assert math.isclose(result, 30.0)
    print(f"分块 attention：{result:.1f}（完整 softmax 加权和也是 30.0）")
    for scores in ([1000, 999, -1000, 1001], [-1000, -1001, -999, -1002]):
        weights = [math.exp(x - max(scores)) for x in scores]
        expected = sum(w * v for w, v in zip(weights, values)) / sum(weights)
        for tile in (1, 2, 3, 4):
            assert math.isclose(online_attention_row(scores, values, tile), expected)
    print("分块 softmax：大幅正负分数、不同 tile 的数值检查通过。")


CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

// 朴素版：一个线程算一个输出，每次乘加都从 global memory 指针取数。
__global__ void naive_kernel(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
        acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

template<int T>
__global__ void tiled_kernel(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];
    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * T + ty;
    int col = blockIdx.x * T + tx;
    float acc = 0.0f;  // 每个线程私有；跨所有 K tile 累加。

    // 即使自己的输出越界，也要参与合作搬运和同步，不能在这里提前 return。
    for (int k0 = 0; k0 < K; k0 += T) {
        As[ty][tx] = (row < M && k0 + tx < K)
            ? A[row * K + k0 + tx] : 0.0f;
        Bs[ty][tx] = (k0 + ty < K && col < N)
            ? B[(k0 + ty) * N + col] : 0.0f;
        __syncthreads();  // 等所有线程搬完，才能读取别人搬入的元素。
        for (int t = 0; t < T; ++t)
            acc += As[ty][t] * Bs[t][tx];
        __syncthreads();  // 等所有线程用完，才能覆盖成下一轮 tile。
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B, int64_t tile) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(A.device() == B.device(), "inputs must be on the same device");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "inputs must be matrices");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 &&
                B.scalar_type() == torch::kFloat32, "inputs must be float32");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(A.size(1) == B.size(0), "inner dimensions must match");
    TORCH_CHECK(tile == 0 || tile == 16 || tile == 32, "tile must be 0, 16 or 32");
    int M = A.size(0), N = B.size(1), K = A.size(1);
    TORCH_CHECK(M > 0 && N > 0 && K > 0, "dimensions must be positive");
    c10::cuda::CUDAGuard guard(A.device());
    auto C = torch::empty({M, N}, A.options());
    auto stream = c10::cuda::getCurrentCUDAStream(A.get_device());
    int T = tile == 0 ? 16 : static_cast<int>(tile);
    dim3 threads(T, T), blocks((N + T - 1) / T, (M + T - 1) / T);
    if (tile == 0)
        naive_kernel<<<blocks, threads, 0, stream.stream()>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    else if (tile == 16)
        tiled_kernel<16><<<blocks, threads, 0, stream.stream()>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    else
        tiled_kernel<32><<<blocks, threads, 0, stream.stream()>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
}
"""


def run_gpu(size, repeats):
    import torch
    from torch.utils.cpp_extension import load_inline

    if not torch.cuda.is_available():
        raise SystemExit("没有可用的 CUDA GPU；可以先运行 --cpu 理解计算过程。")
    # 与前一讲一致：声明给 .cpp 绑定代码看，定义位于上面的 CUDA_SOURCE 中。
    module = load_inline(
        name="lecture004_tiling_demo",
        cpp_sources="torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B, int64_t tile);",
        cuda_sources=CUDA_SOURCE,
        functions=["matmul_cuda"],
        extra_cuda_cflags=["-O2"],
        with_cuda=True,
    )
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    # 先检查可整除和不可整除的矩形，再计时。数值接近，不要求浮点结果逐位相同。
    for m, n, k in [(32, 32, 32), (31, 29, 37), (1, 7, 3)]:
        a = torch.rand(m, k, device="cuda") * 2 - 1
        b = torch.rand(k, n, device="cuda") * 2 - 1
        expected = a @ b
        for tile in (0, 16, 32):
            actual = module.matmul_cuda(a, b, tile)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    print("GPU 正确性检查通过。GPU:", torch.cuda.get_device_name())

    a = torch.rand(size, size, device="cuda")
    b = torch.rand(size, size, device="cuda")
    timings = {}
    for name, tile in [("naive", 0), ("tile=16", 16), ("tile=32", 32)]:
        for _ in range(5):
            module.matmul_cuda(a, b, tile)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            module.matmul_cuda(a, b, tile)
        end.record()
        end.synchronize()
        timings[name] = start.elapsed_time(end) * 1000 / repeats
    for name, us in timings.items():
        print(f"{name:8s}: {us:9.2f} us, 相对 naive {timings['naive'] / us:.2f}x")
    print("这是本机测量；tile 越大不一定越快。编译与预热不计入以上时间。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="不依赖 PyTorch 的手算演示与检查")
    parser.add_argument("--size", type=int, default=512, help="GPU 方阵边长，默认 512")
    parser.add_argument("--repeats", type=int, default=50, help="GPU 重复次数，默认 50")
    args = parser.parse_args()
    if args.size <= 0 or args.repeats <= 0:
        parser.error("--size 和 --repeats 必须为正整数")
    if args.cpu:
        run_cpu()
    else:
        run_gpu(args.size, args.repeats)


if __name__ == "__main__":
    main()
