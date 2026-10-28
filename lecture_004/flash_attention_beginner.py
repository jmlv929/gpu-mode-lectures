"""分块后为什么答案相同？先算标量加权平均，再对照真正的 attention。

运行：python3 lecture_004/flash_attention_beginner.py
只使用 Python 标准库；不运行 GPU，也不模拟实际 smem 或寄存器。
"""

import math

from flash_attention_demo import attention_reference, flash_attention_cpu


def main():
    weights = [1.0, 2.0, 3.0, 4.0]
    values = [10.0, 20.0, 30.0, 40.0]
    full = sum(w * v for w, v in zip(weights, values)) / sum(weights)
    print("一、先把指数权重视为已知，直接算分子和分母")
    print(f"一次性计算：(10+40+90+160)/(1+2+3+4) = {full:g}")
    numerator = denominator = 0.0
    for start in range(0, 4, 2):
        wb = weights[start:start + 2]
        vb = values[start:start + 2]
        add_u = sum(w * v for w, v in zip(wb, vb))
        add_l = sum(wb)
        numerator += add_u
        denominator += add_l
        print(f"第 {start//2+1} 轮：增加分子 {add_u:g}、分母 {add_l:g}；"
              f"累计 u={numerator:g}、l={denominator:g}")
    print(f"最后相除：{numerator:g}/{denominator:g} = {numerator/denominator:g}")
    assert math.isclose(full, numerator / denominator)

    print("\n二、同一组数据，换缩放基准仍得到 30")
    maximum_weight, l, u = 0.0, 0.0, 0.0
    for start in range(0, 4, 2):
        wb = weights[start:start + 2]
        vb = values[start:start + 2]
        new_maximum = max(maximum_weight, max(wb))
        alpha = maximum_weight / new_maximum
        scaled = [w / new_maximum for w in wb]
        l = alpha * l + sum(scaled)
        u = alpha * u + sum(w * v for w, v in zip(scaled, vb))
        maximum_weight = new_maximum
        print(f"第 {start//2+1} 轮：权重缩放基准 C={maximum_weight:g}、"
              f"alpha={alpha:g}；l={l:g}、u={u:g}")
    print(f"最后相除：{u:g}/{l:g} = {u/l:g}")
    assert math.isclose(full, u / l)

    print("\n三、构造 Q、K、V，调用已有的稳定 attention 实现")
    # d=1，所以 q*k/sqrt(d)=log(w)，exp(score) 正好等于 w。
    # log 只用来造出同一组权重；真正的分块算法保持分数最大值 m，避免先算 exp(score)。
    q = [[1.0]]
    k = [[math.log(w)] for w in weights]
    v = [[x] for x in values]
    reference, _, _ = attention_reference(q, k, v)
    actual = flash_attention_cpu(q, k, v, kv_tile=2, trace=True)
    print(f"普通 attention：{reference[0][0]:g}")
    print(f"分块 attention：{actual[0][0]:g}")
    assert math.isclose(reference[0][0], full)
    assert math.isclose(actual[0][0], full)


if __name__ == "__main__":
    main()
