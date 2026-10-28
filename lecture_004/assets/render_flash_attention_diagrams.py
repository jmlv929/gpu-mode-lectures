"""生成文档内可直接显示的 PNG 流程图，并保留 SVG 矢量版本。

运行：python3 lecture_004/assets/render_flash_attention_diagrams.py
依赖 Pillow；默认使用 macOS 中文字体，也可通过 --font 指定字体路径。
"""

import argparse
from html import escape
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "background": "#ffffff", "text": "#172b4d", "muted": "#52657d",
    "line": "#8393a7", "blue": "#eaf3ff", "blue_edge": "#7da8dd",
    "green": "#e8f5ed", "green_edge": "#7caf8a", "soft": "#f3f5f8",
}


class Diagram:
    def __init__(self, width, height, font_path, title):
        self.width, self.height, self.scale = width, height, 2
        self.font_path = font_path
        self.image = Image.new("RGB", (width * 2, height * 2), COLORS["background"])
        self.draw = ImageDraw.Draw(self.image)
        self.svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img">',
            f'<title>{escape(title)}</title>',
            f'<rect width="100%" height="100%" fill="{COLORS["background"]}"/>',
        ]

    def text(self, x, y, lines, size=30, color="text", max_width=None):
        if isinstance(lines, str):
            lines = [lines]
        font = ImageFont.truetype(self.font_path, size * self.scale)
        spacing = int(size * 1.55)
        start = y - (len(lines) - 1) * spacing / 2
        for i, line in enumerate(lines):
            if max_width is not None:
                assert self.draw.textlength(line, font=font) <= max_width * 2, line
            self.draw.text((x * 2, (start + i * spacing) * 2), line,
                           font=font, fill=COLORS[color], anchor="mm")
            self.svg.append(
                f'<text x="{x}" y="{start + i * spacing}" '
                f'font-family="Hiragino Sans GB,Microsoft YaHei,sans-serif" '
                f'font-size="{size}" fill="{COLORS[color]}" text-anchor="middle" '
                f'dominant-baseline="central">{escape(line)}</text>'
            )

    def box(self, bounds, lines, size=30, fill="blue", edge="blue_edge"):
        x0, y0, x1, y1 = bounds
        self.draw.rounded_rectangle(tuple(int(v * 2) for v in bounds), radius=36,
                                    fill=COLORS[fill], outline=COLORS[edge], width=3)
        self.svg.append(
            f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" '
            f'rx="18" fill="{COLORS[fill]}" stroke="{COLORS[edge]}" stroke-width="1.5"/>'
        )
        self.text((x0+x1)/2, (y0+y1)/2, lines, size, max_width=x1-x0-32)

    def arrow(self, points):
        self.draw.line([(int(x*2), int(y*2)) for x,y in points],
                       fill=COLORS["line"], width=4, joint="curve")
        coords = " ".join(f"{x},{y}" for x,y in points)
        self.svg.append(f'<polyline points="{coords}" fill="none" '
                        f'stroke="{COLORS["line"]}" stroke-width="2"/>')
        import math
        px, py = points[-2]
        x, y = points[-1]
        angle = math.atan2(y-py, x-px)
        ends = [(x-13*math.cos(angle-a), y-13*math.sin(angle-a)) for a in (-0.5, 0.5)]
        triangle = [(x,y), *ends]
        self.draw.polygon([(int(a*2),int(b*2)) for a,b in triangle], fill=COLORS["line"])
        self.svg.append(f'<polygon points="{" ".join(f"{a},{b}" for a,b in triangle)}" '
                        f'fill="{COLORS["line"]}"/>')

    def save(self, destination):
        self.image.save(destination.with_suffix(".png"))
        destination.with_suffix(".svg").write_text("\n".join(self.svg + ["</svg>"]))


def memory_diagram(font_path, directory):
    diagram = Diagram(1120, 1280, font_path, "FlashAttention 的显存、smem 与寄存器数据流")
    diagram.box((180, 30, 900, 180), ["GPU 显存：完整 Q、K、V", "以及最终输出 O"], 34)
    diagram.arrow([(540,180), (540,275)])
    diagram.text(650,228,"分块加载",26,"muted")
    diagram.box((180,275,900,425), ["smem：当前 Q、K、V 数据块", "Q 块复用，K、V 块分轮替换"], 30)
    diagram.arrow([(500,425), (500,505)])
    diagram.box((140,505,820,655), ["寄存器：线程持有的局部数据", "计算当前小块得分 s、指数权重 w"], 30)
    diagram.arrow([(480,655), (480,735)])
    diagram.box((140,735,820,885), ["寄存器：跨块保留 m、l、u", "m：最大分数   l：分母   u：分子"], 30)
    diagram.arrow([(820,810), (1030,810), (1030,350), (900,350)])
    diagram.text(960,555,["继续下一块", "K、V", "保留累计量"],24,"muted",max_width=180)
    diagram.arrow([(480,885), (480,1050)])
    diagram.text(650,965,["全部块处理完", "O = u / l"],26,"muted")
    diagram.box((180,1050,900,1170), "最终 O 写回 GPU 显存", 32, "green", "green_edge")
    diagram.text(540,1225,"简化数据流：局部得分可丢弃，历史贡献已合并进 m、l、u。",24,"muted",max_width=1040)
    diagram.save(directory / "flash-attention-memory")


def weighted_sum_diagram(font_path, directory):
    diagram = Diagram(1200, 1060, font_path, "分块前后都得到 30：先合并分子、分母，最后相除")
    diagram.text(600,45,"同一组数据，两种执行顺序，结果都是 30",34)
    diagram.box((100,100,1100,215), ["指数权重 w = [1, 2, 3, 4]", "信息数值 V = [10, 20, 30, 40]"],30)
    diagram.text(600,265,"一次性计算：分子 300，分母 10，输出 300 / 10 = 30",28)
    diagram.arrow([(600,305),(600,355)])
    diagram.text(600,400,"分成两轮：每轮马上累计，不必保存旧得分",30)
    diagram.box((65,470,545,660), ["第一轮：A、B", "分子 1×10 + 2×20 = 50", "分母 1 + 2 = 3"],28)
    diagram.box((655,470,1135,660), ["第二轮：C、D", "分子 3×30 + 4×40 = 250", "分母 3 + 4 = 7"],28)
    diagram.arrow([(305,660),(305,735),(600,735),(600,790)])
    diagram.arrow([(895,660),(895,735),(600,735),(600,790)])
    diagram.box((220,790,980,960), ["合并：u = 50 + 250 = 300", "合并：l = 3 + 7 = 10", "最终输出：O = u / l = 30"],30,"green","green_edge")
    diagram.text(600,1015,"保留的是分子与分母，不是两块各自归一化后的平均值。",26,"muted",max_width=1100)
    diagram.save(directory / "flash-attention-block-equivalence")


def scan_diagram(font_path, directory):
    diagram = Diagram(1320, 1050, font_path, "Q 按行切块，每一块 Q 都扫描全部 K、V 行")
    diagram.text(660,45,"固定 Q 的两行，让它们依次看完全部 K、V",34)
    diagram.text(510,155,["K 第 0、1 行：[0]、[0]", "对应 V：[10]、[20]"],27,max_width=470)
    diagram.text(1000,155,["K 第 2、3 行：[1]、[1]", "对应 V：[30]、[40]"],27,max_width=470)
    diagram.text(145,355,["Q 第 0、1 行", "q0 = [0]", "q1 = [1]"],27,max_width=265)
    diagram.box((310,260,710,450),["第 1 轮：同时处理这两行", "q0 得分：[0, 0]", "q1 得分：[0, 0]"],26)
    diagram.box((800,260,1200,450),["第 2 轮：Q 保持不变", "q0 得分：[0, 0]", "q1 得分：[1, 1]"],26)
    diagram.arrow([(710,355),(800,355)])
    diagram.arrow([(1000,450),(1000,525),(510,525),(510,650)])
    diagram.text(820,590,"前两行输出完成，再换 Q 第 2、3 行",26,max_width=900)
    diagram.text(145,745,["Q 第 2、3 行", "q2 = [2]", "q3 = [3]"],27,max_width=265)
    diagram.box((310,650,710,840),["第 3 轮：K、V 从头扫描", "q2 得分：[0, 0]", "q3 得分：[0, 0]"],26,"green","green_edge")
    diagram.box((800,650,1200,840),["第 4 轮：继续下一块 K、V", "q2 得分：[2, 2]", "q3 得分：[3, 3]"],26,"green","green_edge")
    diagram.arrow([(710,745),(800,745)])
    diagram.text(660,925,"每轮取下一组 K 行，同时取同编号的 V 行。",29,max_width=1200)
    diagram.text(660,985,"按行切块，保留每行的完整向量；图中每行恰好只有 1 个数。",25,"muted",max_width=1200)
    diagram.save(directory / "flash-attention-q-kv-scan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--font", default="/System/Library/Fonts/Hiragino Sans GB.ttc")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    memory_diagram(args.font, directory)
    weighted_sum_diagram(args.font, directory)
    scan_diagram(args.font, directory)
    print("已生成 3 张 PNG 和对应的 SVG。")


if __name__ == "__main__":
    main()
