"""从 Q、K、V 到输出：普通 attention 与 FlashAttention 核心算法的 CPU 对照。

运行：python3 lecture_004/flash_attention_demo.py
仅使用 Python 标准库；模拟单个 head 的无 mask 前向计算，不是 GPU 加速实现。
"""

import math
import random


def matmul(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)]
            for row in a]


def transpose(a):
    return [list(col) for col in zip(*a)]


def attention_reference(q, k, v):
    """直观实现：显式保存整张 scores 和 probabilities。"""
    scale = 1 / math.sqrt(len(q[0]))
    scores = [[s * scale for s in row] for row in matmul(q, transpose(k))]
    probabilities = []
    for row in scores:
        maximum = max(row)
        weights = [math.exp(s - maximum) for s in row]
        denominator = sum(weights)
        probabilities.append([w / denominator for w in weights])
    return matmul(probabilities, v), scores, probabilities


def flash_attention_cpu(q, k, v, q_tile=2, kv_tile=2, trace=False):
    """分块模拟：每行维护 m、l、u，无需保存完整 scores 或 probabilities。

    q_tile 个 query 共用一次 K、V 块加载；每行的统计量独立更新。
    与 GPU kernel 的区别是这里由 Python 循环执行，不模拟并行与同步。
    """
    d, dv = len(q[0]), len(v[0])
    scale = 1 / math.sqrt(d)
    output = []
    for q0 in range(0, len(q), q_tile):
        qb = q[q0:q0 + q_tile]
        m = [-math.inf] * len(qb)
        l = [0.0] * len(qb)
        u = [[0.0] * dv for _ in qb]

        for k0 in range(0, len(k), kv_tile):
            kb, vb = k[k0:k0 + kv_tile], v[k0:k0 + kv_tile]
            # 第一处小矩阵乘法：只生成当前 query 块与 key 块之间的得分。
            sb = [[s * scale for s in row] for row in matmul(qb, transpose(kb))]
            for r, scores in enumerate(sb):
                new_m = max(m[r], max(scores))
                alpha = math.exp(m[r] - new_m)
                weights = [math.exp(s - new_m) for s in scores]
                l[r] = alpha * l[r] + sum(weights)
                # 第二处小矩阵乘法 weights @ V_block，结果立刻并入 u。
                contribution = matmul([weights], vb)[0]
                u[r] = [alpha * old + add for old, add in zip(u[r], contribution)]
                m[r] = new_m
                if trace and q0 == 0 and r == 0:
                    print(f"key {k0}..{k0 + len(kb) - 1}: "
                          f"m={m[r]:.6f}, alpha={alpha:.6f}, "
                          f"l={l[r]:.6f}, u={[round(x, 6) for x in u[r]]}")
            # 进入下一轮后，上一个 sb 和 weights 就不再需要。
        output.extend([[x / denominator for x in row]
                       for row, denominator in zip(u, l)])
    return output


def assert_close(a, b):
    assert len(a) == len(b)
    for actual, expected in zip(a, b):
        assert len(actual) == len(expected)
        for x, y in zip(actual, expected):
            assert math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-10), (x, y)


def main():
    # 刻意构造的数字，便于把指数权重算成 1、2、3、4。
    # q @ k.T / sqrt(2) = [ln(1), ln(2), ln(3), ln(4)]。
    q = [[math.sqrt(2), 0]]
    k = [[math.log(x), 0] for x in (1, 2, 3, 4)]
    v = [[10, 0], [0, 20], [30, 0], [0, 40]]
    expected, scores, probabilities = attention_reference(q, k, v)
    print("scores:", [round(x, 6) for x in scores[0]])
    print("完整 softmax 权重:", [round(x, 6) for x in probabilities[0]])
    print("普通 attention 输出:", [[round(x, 6) for x in row] for row in expected])
    actual = flash_attention_cpu(q, k, v, kv_tile=2, trace=True)
    print("分块 attention 输出:", [[round(x, 6) for x in row] for row in actual])
    assert_close(expected, [[10.0, 20.0]])
    assert_close(actual, expected)

    # 验证多个 query、不同 Q/K/V 长度和向量维度，以及不能整除的块边界。
    rng = random.Random(0)
    for lq, lk, d, dv in [(1, 4, 2, 2), (5, 7, 3, 2), (4, 4, 4, 3)]:
        q = [[rng.uniform(-2, 2) for _ in range(d)] for _ in range(lq)]
        k = [[rng.uniform(-2, 2) for _ in range(d)] for _ in range(lk)]
        v = [[rng.uniform(-2, 2) for _ in range(dv)] for _ in range(lk)]
        expected, _, _ = attention_reference(q, k, v)
        for q_tile in (1, 2, 3):
            for kv_tile in (1, 2, 3, 4):
                assert_close(flash_attention_cpu(q, k, v, q_tile, kv_tile), expected)
    print("多 query、向量 V、矩形尺寸和不同分块大小的结果检查通过。")

    # 分数超过 1000 时，直接 exp(score) 会溢出；减去 running max 后仍能计算。
    q = [[math.sqrt(2), 0]]
    k = [[s, 0] for s in (1000, 999, -1000, 1001)]
    v = [[10, 0], [0, 20], [30, 0], [0, 40]]
    expected, _, _ = attention_reference(q, k, v)
    for tile in (1, 2, 3, 4):
        assert_close(flash_attention_cpu(q, k, v, kv_tile=tile), expected)
    print("大幅分数与跨块最大值更新的稳定性检查通过。")


if __name__ == "__main__":
    main()
