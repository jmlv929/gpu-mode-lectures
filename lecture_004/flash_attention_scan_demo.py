"""把“固定一块 Q，扫描 K、V”展开成 4 轮具体的读取与计算。

运行：python3 lecture_004/flash_attention_scan_demo.py
仅使用标准库，在 CPU 上说明数据切片和数学关系，不模拟 GPU 存储。
向量维度 d=1、分数很小，因此先用未缩放指数权重教学；最后与稳定实现核对。
"""

import math

from flash_attention_demo import (
    assert_close,
    attention_reference,
    flash_attention_cpu,
    matmul,
    transpose,
)


def rounded(matrix):
    return [[round(x, 6) for x in row] for row in matrix]


def main():
    q = [[0.0], [1.0], [2.0], [3.0]]
    k = [[0.0], [0.0], [1.0], [1.0]]
    v = [[10.0], [20.0], [30.0], [40.0]]
    print("完整 Q =", q)
    print("完整 K =", k)
    print("完整 V =", v)
    output = []
    step = 0

    for q_start in (0, 2):
        qb = q[q_start:q_start + 2]
        u, l = [0.0, 0.0], [0.0, 0.0]
        print(f"\n固定 Q 第 {q_start}、{q_start+1} 行：{qb}")
        for kv_start in (0, 2):
            step += 1
            kb = k[kv_start:kv_start + 2]
            vb = v[kv_start:kv_start + 2]
            # d=1，所以缩放因子 1/sqrt(d)=1。
            scores = matmul(qb, transpose(kb))
            weights = [[math.exp(s) for s in row] for row in scores]
            add_u = matmul(weights, vb)
            for row in range(2):
                u[row] += add_u[row][0]
                l[row] += sum(weights[row])
            print(f"  第 {step} 轮：K、V 取第 {kv_start}、{kv_start+1} 行")
            print("    Q 块保持为", qb)
            print("    当前 K 块", kb, "；对应 V 块", vb)
            print("    小得分矩阵", rounded(scores))
            print("    每行累计分子", [round(x, 6) for x in u])
            print("    每行累计分母", [round(x, 6) for x in l])
        block_output = [[a/b] for a,b in zip(u,l)]
        print(f"  完成 O 第 {q_start}、{q_start+1} 行：", rounded(block_output))
        output.extend(block_output)

    reference, _, _ = attention_reference(q, k, v)
    stable = flash_attention_cpu(q, k, v, q_tile=2, kv_tile=2)
    assert_close(output, reference)
    assert_close(output, stable)
    print("\n全部输出：", rounded(output))
    print("与一次性 attention、稳定分块 attention 的结果核对通过。")


if __name__ == "__main__":
    main()
