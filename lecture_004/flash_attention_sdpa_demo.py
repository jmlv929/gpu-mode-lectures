"""相同的四个 key：显式 attention 对照 PyTorch SDPA。

运行：python3 lecture_004/flash_attention_sdpa_demo.py
依赖 PyTorch；有 CUDA 时强制使用 FLASH_ATTENTION 后端，CPU 时只核对数学结果。
GPU 内部分块由后端决定，不保证按教学示例每块两个 key 执行。
"""

import math


def main():
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # SDPA 的维度：[batch, head, query/key 个数, head_dim]。
    # head_dim 取 64，让输入符合常见 CUDA FlashAttention 后端的尺寸要求。
    # 后 62 维全为零，不影响手算；缩放因子是 1/sqrt(64) = 1/8。
    q = torch.zeros((1, 1, 1, 64), device=device, dtype=dtype)
    k = torch.zeros((1, 1, 4, 64), device=device, dtype=dtype)
    v = torch.zeros_like(k)
    q[0, 0, 0, 0] = 8
    k[0, 0, :, 0] = torch.tensor(
        [math.log(x) for x in (1, 2, 3, 4)], device=device, dtype=dtype
    )
    v[0, 0, :, :2] = torch.tensor(
        [[10, 0], [0, 20], [30, 0], [0, 40]], device=device, dtype=dtype
    )

    # 参考计算会保存完整 scores 和 probabilities，仅用于这个小例子的核对。
    scores = q.float() @ k.float().transpose(-2, -1) / math.sqrt(64)
    probabilities = scores.softmax(dim=-1)
    reference = probabilities @ v.float()

    # 这里只有一个 query，而且允许它看到全部四个 key。
    # 不要设置 is_causal=True：非方阵的因果对齐与“最后一个 token”不是一回事。
    backend = SDPBackend.FLASH_ATTENTION if device == "cuda" else SDPBackend.MATH
    with sdpa_kernel(backends=[backend]):
        output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        )

    torch.testing.assert_close(output.float(), reference, atol=0.02, rtol=0.002)
    torch.testing.assert_close(
        output[0, 0, 0, :2].float(),
        torch.tensor([10.0, 20.0], device=device),
        atol=0.02,
        rtol=0.002,
    )
    print(f"device={device}, backend={backend}")
    print("softmax 权重 ≈", probabilities[0, 0, 0].tolist())
    print("输出的前两维 ≈", output[0, 0, 0, :2].tolist())
    print("与显式 attention、手算 [10, 20] 核对通过。")
    if device == "cpu":
        print("本次运行的是 CPU 数学后端，没有执行 CUDA FlashAttention。")


if __name__ == "__main__":
    main()
