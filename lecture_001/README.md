## Misc notes and scripts
Used these for the first lecture of the CUDA mode series

Most of them are about profiling or authoring kernels in various GPU programming languages

## 中文学习指南

[从 PyTorch 平方到自己的 GPU Kernel：循序渐进的实验指南](STUDY_GUIDE.zh-CN.md)

包含文件与生成物分类、SSH 远程环境、计时与 Profiler、PyBind/CUDA 扩展、Triton、torch.compile、Nsight，以及原代码和讲义中的注意事项。

已补入本次 RTX 3080 Ti 的两份 Profiler 报告：[第 4.6 节实测图解](STUDY_GUIDE.zh-CN.md#profile-results)。包含 3 张由原始 JSON 绘制的时间线、CPU/GPU 耗时解释，以及同名统计行和 `109.73%` 的原因。
