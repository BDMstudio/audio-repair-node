
**这版 v2 可以作为开发基线，但必须先做 4 处结构级修改。**
文档当前的核心问题集中在：**包结构、`pyproject.toml`、特征定义口径、测试计划**。这些问题都能从文档里直接看出来，比如它现在一边写 `python -m audio_repair ...`，一边又把脚本入口写成 `audio-repair = "src.cli:main"`，同时目录又是 `src/cli.py` 这种布局，这三者是冲突的。

---

# 一、必须修改的 4 项

## 1）把“假 src-layout”改成“真 src-layout”

你现在文档里的目录是：

```text
audio-repair-node/
├── src/
│   ├── cli.py
│   ├── types.py
│   ├── config.py
│   ├── core/
│   ├── detection/
│   └── pipeline/
```

同时 CLI 写的是：

```bash
python -m audio_repair analyze ...
```

并且 `pyproject.toml` 里写的是：

```toml
[project.scripts]
audio-repair = "src.cli:main"
```

这套写法不统一。

### 正确改法

把目录改成：

```text
audio-repair-node/
├── src/
│   └── audio_repair/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── types.py
│       ├── config.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── audio_loader.py
│       │   ├── stft_compute.py
│       │   └── vocal_activity_detector.py
│       ├── detection/
│       │   ├── __init__.py
│       │   ├── feature_extractor.py
│       │   ├── harsh_detector.py
│       │   └── distortion_detector.py
│       └── pipeline/
│           ├── __init__.py
│           ├── segment_merger.py
│           ├── repair_router.py
│           └── analyzer.py
├── configs/
├── tests/
├── experiments/
├── pyproject.toml
└── PRD.md
```

### 为什么必须这么改

因为你要同时支持两种调用方式：

```bash
python -m audio_repair analyze ...
audio-repair analyze ...
```

那包名就必须真的叫 `audio_repair`，而不是把 `src` 当包名硬用。

### 对文档的具体替换

把原来的“目录结构”整段替换成上面的版本。
把所有形如：

* `src/cli.py`
* `src/types.py`
* `src/core/...`

全部改成：

* `src/audio_repair/cli.py`
* `src/audio_repair/types.py`
* `src/audio_repair/core/...`

---

## 2）重写 `pyproject.toml`

你现在文档里的 `pyproject.toml` 太薄，而且脚本入口错了。

### 直接替换为这版

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "audio-repair-node"
version = "0.1.0"
description = "AI vocal defect detection and repair routing for harsh/distortion issues"
readme = "PRD.md"
requires-python = ">=3.11"
dependencies = [
    "librosa>=0.10",
    "numpy>=1.24",
    "scipy>=1.11",
    "soundfile>=0.12",
]

[project.optional-dependencies]
dev = [
    "pytest>=7",
    "pytest-cov",
]
experiment = [
    "silero-vad",
]

[project.scripts]
audio-repair = "audio_repair.cli:main"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
```

### 同时新增 `__main__.py`

`src/audio_repair/__main__.py`

```python
from .cli import main

if __name__ == "__main__":
    main()
```

这样下面两条都会成立：

```bash
python -m audio_repair analyze ...
audio-repair analyze ...
```

---

## 3）把 `highband_ratio` 的定义写死，不准再写 `full`

你现在文档在 `feature_extractor.py` 里写的是：

```text
highband_ratio = band_ratio(S, 4000–12000, full)
```

这个 `full` 是废话式定义，落地时一定出岔子。

### 你现在就拍板成下面这版

我建议 V1 用这条，最稳：

```text
highband_ratio = E(4000–12000 Hz) / E(200–4000 Hz)
```

### 为什么选这版

因为主问题是：

* 上面是“坏高频”
* 下面是“人声主体”

拿高频对主体带做比值，最符合前面“不是绝对音量，而是听感烦躁和嘶嘶空洞”的定义。

### 文档具体替换

把 `feature_extractor.py` 表格里的这一行：

```text
| `highband_ratio` | `band_ratio(S, 4000–12000, full)` |
```

改成：

```text
| `highband_ratio` | `band_ratio(S, 4000–12000, 200–4000)` |
```

并在 `stft_compute.py` 说明里补一句：

```text
- `band_ratio(S, freqs, target_band, ref_band) → ndarray[T]`
- V1 中 `highband_ratio` 的 `ref_band` 固定为 `200–4000 Hz`，禁止使用模糊引用如 `full`
```

---

## 4）补齐测试矩阵，别再只测边角料

你现在只列了：

* `test_stft_compute`
* `test_segment_merger`
* `test_repair_router`

这不够。

### 直接改成这套

```text
tests/
├── conftest.py
├── fixtures/
│   ├── sine_440.wav
│   ├── white_noise.wav
│   ├── vocal_harsh_sample.wav
│   ├── vocal_distortion_sample.wav
│   └── instrumental_bright_overlap.wav
├── test_audio_loader.py
├── test_stft_compute.py
├── test_vocal_activity_detector.py
├── test_feature_extractor.py
├── test_harsh_detector.py
├── test_distortion_detector.py
├── test_segment_merger.py
├── test_repair_router.py
└── test_analyzer_e2e.py
```

### 每个测试该测什么

#### `test_vocal_activity_detector.py`

验证：

* 静音段不过门
* 正常 vocal 段过门
* 极短瞬态不会被当成长 vocal 段

#### `test_feature_extractor.py`

验证：

* `harsh_band_ratio_main`
* `harsh_band_ratio_wide`
* `highband_ratio`
* `flatness_air`
* `rms_envelope`

至少保证输出 shape 正确、无 NaN、范围合理。

#### `test_harsh_detector.py`

给一段高频偏刺的人声样本，确认：

* `harsh_score` 上升
* `distortion_score` 不应被同步高估太多

#### `test_distortion_detector.py`

给一段“嘶嘶/空洞”的坏样本，确认：

* `flatness_air` 上升
* `distortion_score` 高于 clean baseline

#### `test_analyzer_e2e.py`

从输入 wav 到输出：

* `defect_segments.json`
* `repair_plan.json`

都成功生成，字段完整。

---

# 二、建议优化的 4 项

## 5）给 `band_flatness()` 补数值安全约束

你文档里已经设计了 `band_flatness(S, freqs, low, high)`，这方向没错。
但如果不补数值安全，后面一定会出现 `nan` 或极端噪声帧乱跳。

### 文档里直接补这几条

在 `stft_compute.py` 段落下新增：

```text
- `band_flatness` 在计算几何均值 / 算术均值比值时必须加 `eps = 1e-10`
- 仅在该频段总能量高于最小门限时计算；低于门限则回退为 0 或使用 masked 输出
- 对所有输出执行 `np.nan_to_num(...)`
```

### 推荐伪实现

```python
def band_flatness(S, freqs, low, high, eps=1e-10):
    band = slice_band(S, freqs, low, high)
    band = np.maximum(band, eps)
    gm = np.exp(np.mean(np.log(band), axis=0))
    am = np.mean(band, axis=0)
    out = gm / np.maximum(am, eps)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
```

---

## 6）把 `vocal_activity_detector` 升级成“双阈值门限”

你现在写的是：

* RMS
* 局部中位数 × 系数
* 最小活跃段过滤 

这对 V1 可以，但太容易在假声尾音、弱气声上砍错。

### 最小修改方案

不要立刻上 Silero。
先把 VAD 改成 **双阈值 + 持续时间**。

### 具体规则

新增配置：

```json
{
  "vad": {
    "enter_ratio": 1.25,
    "exit_ratio": 1.10,
    "median_window_frames": 31,
    "min_active_ms": 120
  }
}
```

### 行为逻辑

* 当 RMS > 局部中位数 × `enter_ratio`，进入 active
* 已经 active 后，只有 RMS < 局部中位数 × `exit_ratio` 才退出
* 这样可以减少边缘抖动

### 文档替换

把：

```text
- 自适应门限：局部中位数 × 系数
```

改成：

```text
- 自适应双阈值门限：局部中位数 × enter_ratio / exit_ratio
- 使用 hysteresis 减少弱唱段、尾音、气声边界抖动
```

---

## 7）把“歌曲级归一化”写具体，不准再只写口号

你现在对 `harsh_detector.py` 的说明是：

> 每个子指标先做歌曲级归一化（相对于该曲自身统计基线）

这话方向对，但实现者看完还是会懵。

### 建议直接写成：

在 `harsh_detector.py` 和 `distortion_detector.py` 下都补：

```text
归一化策略（V1）：
- 仅对 vocal-active frames 统计基线
- baseline 使用中位数 `median`
- spread 使用 IQR（Q3 - Q1）
- 归一化公式：
  norm_x = clip((x - median) / max(IQR, eps), 0, upper_cap)

其中：
- `upper_cap` 默认可取 3.0
- `eps = 1e-6`
```

这样实现者不会自己发明一套乱七八糟的 z-score。

---

## 8）把 Manual Verification 从“听一听”升级成“小型标注集”

你现在写的是：

1. 运行命令
2. 看 JSON
3. 去 DAW 里定位核对听感 

这不够。太主观。

### 直接改成这个流程

新增一个 `validation/` 目录：

```text
validation/
├── manifest.json
├── labels/
│   ├── song01.json
│   ├── song02.json
│   └── ...
└── notes.md
```

### 每首标注内容

```json
{
  "file": "song01_vocal.wav",
  "segments": [
    {
      "start_ms": 18240,
      "end_ms": 18780,
      "label": "distortion"
    },
    {
      "start_ms": 42350,
      "end_ms": 43180,
      "label": "harsh"
    }
  ]
}
```

### 验证指标先别贪

V1 只看 3 个：

* 命中率
* 误报率
* 时间偏移误差（ms）

### 文档里新增一句

```text
Manual Verification 升级为“小型人工标注集验证”，至少包含 10 首 vocal stem，每首手工标注 harsh / distortion 片段，作为阈值调优依据。
```

---

# 三、你可以直接贴回文档的替换片段

## A. 替换 `pyproject.toml`

用我上面给你的整段。

## B. 替换目录结构

用下面这段：

```text
audio-repair-node/
├── src/
│   └── audio_repair/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── types.py
│       ├── config.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── audio_loader.py
│       │   ├── stft_compute.py
│       │   └── vocal_activity_detector.py
│       ├── detection/
│       │   ├── __init__.py
│       │   ├── feature_extractor.py
│       │   ├── harsh_detector.py
│       │   └── distortion_detector.py
│       └── pipeline/
│           ├── __init__.py
│           ├── segment_merger.py
│           ├── repair_router.py
│           └── analyzer.py
├── configs/
│   └── repair_config.json
├── tests/
│   ├── fixtures/
│   ├── test_audio_loader.py
│   ├── test_stft_compute.py
│   ├── test_vocal_activity_detector.py
│   ├── test_feature_extractor.py
│   ├── test_harsh_detector.py
│   ├── test_distortion_detector.py
│   ├── test_segment_merger.py
│   ├── test_repair_router.py
│   ├── test_analyzer_e2e.py
│   └── conftest.py
├── experiments/
│   └── silero_vad_compare.py
├── validation/
│   ├── manifest.json
│   └── labels/
├── pyproject.toml
└── PRD.md
```

## C. 替换 `feature_extractor.py` 表格中的这一行

从：

```text
| `highband_ratio` | `band_ratio(S, 4000–12000, full)` |
```

改成：

```text
| `highband_ratio` | `band_ratio(S, 4000–12000, 200–4000)` |
```

## D. 替换 `vocal_activity_detector.py` 描述

从：

```text
- 自适应门限：局部中位数 × 系数
```

改成：

```text
- 自适应双阈值门限（hysteresis）：
  - enter: 局部中位数 × enter_ratio
  - exit:  局部中位数 × exit_ratio
- 结合最小活跃段时长过滤，减少气声、尾音、瞬态噪声误判
```

---

# 四、拍板后的最终版本

按这条顺序改：

1. **先改目录和 `pyproject.toml`**
2. **再把 `highband_ratio` 写死**
3. **补 detector 测试文件名和测试目标**
4. **最后再升级 VAD 说明和验证计划**
