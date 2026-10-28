from pathlib import Path
import torch
from torchvision.io import read_image, write_png
from torch.utils.cpp_extension import load_inline


def compile_extension():
    cuda_source = Path("grayscale_kernel.cu").read_text()
    # cpp_source 是一段 C++ 源码字符串；这里仅包含函数的“前向声明”：
    #   torch::Tensor       返回类型：PyTorch Tensor 的 C++ 类型；torch:: 表示命名空间。
    #   rgb_to_grayscale    函数名。
    #   torch::Tensor image 参数：接收一个名为 image 的 Tensor。
    #   末尾的分号 ;        表示这里只声明接口；函数定义会包含 { ... } 函数体。
    # 真正的定义在 grayscale_kernel.cu 中：它在 CPU 端检查输入、分配输出，
    # 再用 <<<...>>> 启动 GPU 上的 rgb_to_grayscale_kernel。
    cpp_source = "torch::Tensor rgb_to_grayscale(torch::Tensor image);"

    # 为什么 .cu 已经定义了函数，这里还要声明一次？
    # load_inline 将 cpp_sources 和 cuda_sources 分别写成 .cpp、.cu 后独立编译。
    # functions=["rgb_to_grayscale"] 会让它在 .cpp 中自动生成 pybind11 绑定，
    # 绑定代码需要先知道这个函数的名称、参数和返回类型，因此这里提供声明。
    # .cpp 编译器不会因为函数定义在另一个 .cu 文件里，就自动知道其接口；
    # 最终链接时，.cpp 中对该函数的引用才会与 .cu 中的定义连接起来。
    # 默认还会自动添加 #include <torch/extension.h>，所以这里不必手写该头文件。
    # 编译加载后，Python 调用链为：
    # ext.rgb_to_grayscale(x) -> pybind11 绑定 -> C++ 包装函数 -> GPU kernel。
    # 参考：https://docs.pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load_inline

    # Load the CUDA kernel as a PyTorch extension
    rgb_to_grayscale_extension = load_inline(
        name="rgb_to_grayscale_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["rgb_to_grayscale"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
        # build_directory='./cuda_build',
    )
    return rgb_to_grayscale_extension


def main():
    """
    Use torch cpp inline extension function to compile the kernel in grayscale_kernel.cu.
    Read input image, convert it to grayscale via custom cuda kernel and write it out as png.
    """
    ext = compile_extension()

    x = read_image("Grace_Hopper.jpg").permute(1, 2, 0).cuda()
    print("mean:", x.float().mean())
    print("Input image:", x.shape, x.dtype)

    assert x.dtype == torch.uint8

    y = ext.rgb_to_grayscale(x)

    print("Output image:", y.shape, y.dtype)
    print("mean", y.float().mean())
    write_png(y.permute(2, 0, 1).cpu(), "output.png")


if __name__ == "__main__":
    main()
