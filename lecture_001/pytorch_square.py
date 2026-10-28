import torch

# 先在 CPU 上用一个小张量展示三种逐元素平方写法，输出均为 [1., 4., 9.]。
a = torch.tensor([1., 2., 3.])

print(torch.square(a))  # 调用 PyTorch 的平方算子。
print(a ** 2)           # 使用幂运算。
print(a * a)            # 对应位置的元素相乘，并非矩阵乘法。

def time_pytorch_function(func, input):
    # CUDA 运算相对 CPU 异步执行：直接用 Python 时钟计时且不做同步，
    # 可能只测到 CPU 提交任务的耗时。这里用 CUDA Event 测量 GPU 执行时间。
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # 预热 5 次，减少首次调用时的初始化、内存分配等开销对测量的影响。
    for _ in range(5):
        func(input)

    # 两个事件记录在当前 CUDA 流中，将待测运算夹在中间。
    start.record()
    func(input)
    end.record()
    # 等待 GPU 完成任务，确保读取时间时两个事件都已执行。
    torch.cuda.synchronize()
    # 返回两个事件之间的耗时，单位为毫秒（ms）。
    return start.elapsed_time(end)

# 创建标准正态分布的随机张量并复制到 GPU；后续代码需要可用的 CUDA 环境。
# 共 1 亿个元素，默认 float32 时仅输入张量就占约 400 MB，运算结果还需额外显存。
b = torch.randn(10000, 10000).cuda()

# 将两种运算符写法包装为函数，方便传入同一个计时函数进行比较。
def square_2(a):
    return a * a

def square_3(a):
    return a ** 2

# 分别测量三种写法；这里只调用函数，返回的毫秒数没有保存或打印。
time_pytorch_function(torch.square, b)
time_pytorch_function(square_2, b)
time_pytorch_function(square_3, b)

print("=============")
print("Profiling torch.square")
print("=============")

# 用 PyTorch Profiler 分别采集三种写法的算子执行信息。
# 实际采集到的 CUDA 信息取决于 PyTorch 版本及当前环境的 Profiler 支持。
with torch.profiler.profile() as prof:
    torch.square(b)

# 按算子名称汇总统计，按 CUDA 总耗时排序，最多显示 10 行。
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("=============")
print("Profiling a * a")
print("=============")

# 单独采集逐元素乘法，避免与其他写法的统计混在一起。
with torch.profiler.profile() as prof:
    square_2(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("=============")
print("Profiling a ** 2")
print("=============")

# 单独采集幂运算，便于比较三种数学等价写法对应的底层算子与耗时。
with torch.profiler.profile() as prof:
    square_3(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
