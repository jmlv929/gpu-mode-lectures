"""从原始 Chrome Trace 生成学习指南中的图片和可核对的统计。

运行：python lecture_001/reprots/lecture001-reports/visualize_reports.py
依赖：matplotlib；不需要 CUDA、PyTorch，也不会修改输入报告。
"""
import hashlib
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'lecture001-matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir()) / 'lecture001-font-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parents[1] / 'images'
OUT.mkdir(exist_ok=True)
for font in [Path('/System/Library/Fonts/STHeiti Medium.ttc'),
             Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')]:
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=font).get_name()
        break
plt.rcParams.update({'font.size': 12, 'axes.unicode_minus': False,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.spines.left': False, 'savefig.facecolor': '#ffffff'})
C = {'cpu': '#547AA5', 'overhead': '#D58D37', 'api': '#9773B2',
     'kernel': '#258E73', 'copy': '#348EB7', 'range': '#BCC6D2', 'gap': '#E3E8ED'}


def load(name):
    path = ROOT / 'lecture001-reports' / f'{name}.json'
    raw = path.read_bytes()
    data = json.loads(raw)
    events = [e for e in data['traceEvents'] if e.get('ph') == 'X']
    origin = min(e['ts'] for e in events if e.get('cat') == 'user_annotation')
    return data, events, origin, hashlib.sha256(raw).hexdigest()


def select(events, cat, name=None):
    return sorted([e for e in events if e.get('cat') == cat
                   and (name is None or e['name'] == name)], key=lambda e: e['ts'])


def setup(ax, labels, limits, title, unit='us'):
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - .4, -.7)
    ax.set_xlim(*limits)
    ax.set_title(title, loc='left', pad=15, fontsize=15, fontweight='bold')
    ax.set_xlabel(f'相对本份报告第一个 CPU 标签的时间（{unit}）')
    ax.grid(axis='x', color='#E6EBEF', zorder=0)
    ax.tick_params(axis='y', length=0, pad=10)


def bars(ax, events, row, origin, color, scale=1, label=False):
    for i, e in enumerate(events):
        x, w = (e['ts'] - origin) / scale, e['dur'] / scale
        ax.broken_barh([(x, w)], (row - .22, .44), facecolors=color, zorder=3)
        if label:
            ax.text(x + w / 2, row - .32, f"#{i+1}: {e['dur']:.3f}",
                    ha='center', va='bottom', fontsize=10, color=C['kernel'])


def save(fig, name):
    fig.savefig(OUT / name, dpi=180, bbox_inches='tight')
    plt.close(fig)


sq, se, so, sh = load('square_only')
cp, ce, co, ch = load('with_copy')
sk = select(se, 'kernel')
ck = select(ce, 'kernel')
mem = select(ce, 'gpu_memcpy')
assert len(sk) == 5 and len(ck) == 1 and len(mem) == 1
assert len(select(se, 'gpu_memcpy')) == 0

# 图 1：同一报告的全景与局部放大。图中的空白保留真实时间间隔。
fig, axs = plt.subplots(2, 1, figsize=(13, 8.6), layout='constrained')
fig.suptitle('只做平方：CPU 提交 5 次，GPU 执行 5 个 kernel', fontsize=19, fontweight='bold')
labels = ['CPU 标签 square_only', 'CPU aten::square', '采集缓冲区请求',
          'CPU cudaLaunchKernel', 'GPU 平方 kernel']
setup(axs[0], labels, (0, 2820), '全景：第一次 CPU 范围很长，GPU 计算仍只有约 10 us')
for row, cat, name, col in [(0, 'user_annotation', None, 'range'),
                            (1, 'cpu_op', 'aten::square', 'cpu'),
                            (2, 'overhead', None, 'overhead'),
                            (3, 'cuda_runtime', 'cudaLaunchKernel', 'api'),
                            (4, 'kernel', None, 'kernel')]:
    bars(axs[0], select(se, cat, name), row, so, C[col])
axs[0].text(1150, 2, 'Activity Buffer Request: 1936.003 us',
            ha='center', va='center', color='white', fontsize=11)
axs[0].annotate('第 1 个 kernel: 9.243 us',
                xy=(sk[0]['ts'] - so, 4), xytext=(1350, 3.45),
                arrowprops={'arrowstyle': '->', 'color': C['kernel']}, color=C['kernel'])
setup(axs[1], ['CPU 标签 square_only', 'CPU aten::square',
               'CPU cudaLaunchKernel', 'GPU 平方 kernel'],
      (2420, 2720), '放大第 2–5 次：CPU API 与 GPU 活动可以在时间上重叠')
for row, cat, name, col in [(0, 'user_annotation', None, 'range'),
                            (1, 'cpu_op', 'aten::square', 'cpu'),
                            (2, 'cuda_runtime', 'cudaLaunchKernel', 'api'),
                            (3, 'kernel', None, 'kernel')]:
    bars(axs[1], select(se, cat, name)[1:], row, so, C[col])
for i, e in enumerate(sk[1:], 2):
    axs[1].text(e['ts'] - so + e['dur']/2, 3.32, f"#{i}\n{e['dur']:.3f} us",
                ha='center', va='top', fontsize=10, color=C['kernel'])
axs[1].set_ylim(3.95, -.7)
save(fig, 'profiler-square-only.png')

# 图 2：包含 CPU 数据准备的全景与传输、计算局部。
fig, axs = plt.subplots(2, 1, figsize=(13, 11), layout='constrained',
                        gridspec_kw={'height_ratios': [1, 1.5]})
fig.suptitle('包含上传：CPU 先造数据，再上传，最后才做平方', fontsize=19, fontweight='bold')
setup(axs[0], ['CPU 外层标签', 'CPU aten::randn', 'CPU aten::to（上传）',
               '采集缓冲区请求', 'GPU HtoD 复制', 'GPU 平方 kernel'],
      (0, 12.7), '全景：CPU 随机数生成占 9.347 ms；平方在最右端', unit='ms')
for row, events, col in [(0, select(ce, 'user_annotation'), 'range'),
                         (1, select(ce, 'cpu_op', 'aten::randn'), 'cpu'),
                         (2, select(ce, 'cpu_op', 'aten::to')[:1], 'cpu'),
                         (3, select(ce, 'overhead'), 'overhead'),
                         (4, mem, 'copy'), (5, ck, 'kernel')]:
    bars(axs[0], events, row, co, C[col], scale=1000)
axs[0].text(4.7, 1, 'aten::randn: 9.347 ms', ha='center', va='center', color='white')
axs[0].text(10.9, 2, 'aten::to: 3.006 ms', ha='center', va='center', color='white', fontsize=11)
setup(axs[1], ['CPU 外层标签', 'CPU aten::copy_', 'CPU cudaMemcpyAsync',
               'CPU cudaStreamSynchronize', 'CPU cudaLaunchKernel',
               'GPU HtoD 复制', 'GPU 平方 kernel', 'GPU 同名范围',
               'CPU cudaDeviceSynchronize'],
      (11500, 12650), '放大末尾：上传活动与 API 调用重叠，随后才启动平方')
for row, events, col in [(0, select(ce, 'user_annotation'), 'range'),
                         (1, select(ce, 'cpu_op', 'aten::copy_'), 'cpu'),
                         (2, select(ce, 'cuda_runtime', 'cudaMemcpyAsync'), 'api'),
                         (3, select(ce, 'cuda_runtime', 'cudaStreamSynchronize'), 'api'),
                         (4, select(ce, 'cuda_runtime', 'cudaLaunchKernel'), 'api'),
                         (5, mem, 'copy'), (6, ck, 'kernel'),
                         (7, select(ce, 'gpu_user_annotation'), 'range'),
                         (8, select(ce, 'cuda_runtime', 'cudaDeviceSynchronize'), 'api')]:
    bars(axs[1], events, row, co, C[col])
axs[1].text(mem[0]['ts'] - co + mem[0]['dur']/2, 5,
            '1 次 HtoD: 683.293 us / 4 MiB', ha='center', va='center', color='white')
axs[1].annotate('1 次平方: 10.299 us', xy=(ck[0]['ts'] - co, 6),
                xytext=(11900, 6.5), color=C['kernel'],
                arrowprops={'arrowstyle': '->', 'color': C['kernel']})
save(fig, 'profiler-with-copy.png')

# 图 3：明确区分真实设备活动累计与范围跨度。
m, k = mem[0], ck[0]
gap = k['ts'] - (m['ts'] + m['dur'])
span = k['ts'] + k['dur'] - m['ts']
active = m['dur'] + k['dur']
fig, ax = plt.subplots(figsize=(13, 4.8), layout='constrained')
setup(ax, ['真实时间线\n复制开始 → kernel 结束', '活动累计\n仅复制 + kernel'],
      (0, 790), '为什么显示 109.73%？范围里还有 67.456 us 的间隔')
ax.set_xlabel('从本次 HtoD 开始计时（us）；下行仅表示时长相加')
ax.broken_barh([(0, m['dur'])], (-.22, .44), facecolors=C['copy'])
ax.broken_barh([(m['dur'], gap)], (-.22, .44), facecolors=C['gap'], hatch='///', edgecolors='#A5AFB9')
ax.broken_barh([(m['dur'] + gap, k['dur'])], (-.22, .44), facecolors=C['kernel'])
ax.broken_barh([(0, m['dur'])], (.78, .44), facecolors=C['copy'])
ax.broken_barh([(m['dur'], k['dur'])], (.78, .44), facecolors=C['kernel'])
ax.text(340, 0, f'复制 {m["dur"]:.3f} us', color='white', ha='center', va='center')
ax.text(340, 1, f'复制 {m["dur"]:.3f} us', color='white', ha='center', va='center')
ax.annotate(f'活动间隔 {gap:.3f} us', xy=(m['dur'] + gap/2, 0),
            xytext=(480, -.46), arrowprops={'arrowstyle': '->', 'color': '#596775'}, fontsize=11)
ax.annotate(f'平方 {k["dur"]:.3f} us', xy=(m['dur']+gap+k['dur']/2, 0),
            xytext=(780, -.45), ha='right', color=C['kernel'],
            arrowprops={'arrowstyle': '->', 'color': C['kernel']}, fontsize=11)
ax.text(span + 5, .2, f'{span:.3f} us', ha='left', fontsize=10)
ax.text(active + 5, 1.2, f'{active:.3f} us', ha='left', fontsize=10)
ax.set_title(ax.get_title(loc='left') + '\n761.048 / 693.592 × 100% ≈ 109.73%（原日志的统计口径）',
             loc='left', fontsize=15, pad=25)
save(fig, 'profiler-range-vs-activity.png')

summary = {
    'units': 'us unless specified; ts/dur in Chrome Trace are microseconds',
    'square_only': {'source': 'lecture001-reports/square_only.json', 'sha256': sh,
        'kernel_count': len(sk), 'kernel_durations_us': [e['dur'] for e in sk],
        'kernel_total_us': round(sum(e['dur'] for e in sk), 3),
        'kernel_mean_us': round(sum(e['dur'] for e in sk)/len(sk), 3), 'htod_count': 0},
    'with_copy': {'source': 'lecture001-reports/with_copy.json', 'sha256': ch,
        'kernel_count': len(ck), 'kernel_us': k['dur'], 'htod_count': len(mem),
        'htod_us': m['dur'], 'htod_bytes': m['args']['bytes'],
        'active_total_us': active, 'gap_us': round(gap, 3),
        'first_activity_to_last_end_us': round(span, 3),
        'gpu_annotation_us': select(ce, 'gpu_user_annotation')[0]['dur'],
        'cpu_annotation_us': select(ce, 'user_annotation')[0]['dur'],
        'copy_over_kernel_ratio': m['dur']/k['dur'],
        'copy_fraction_of_active': m['dur']/active},
}
(ROOT / 'trace_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(summary, ensure_ascii=False, indent=2))
