# AI Vocal De-artifact & Cleanup — V1 实现方案 v3

## 项目定位

为 AI 生成歌曲（Suno 等）的 vocal stem 提供**自动缺陷检测与修复路由**。V1 聚焦两类缺陷：
- **刺耳类 (harsh)**：高频持续占优，烦躁/刮耳
- **失真类 (distortion)**：高频噪声化，嘶嘶/空洞/低码率感

V1 只输出 `defect_segments.json` + `repair_plan.json`，不执行实际修复。

---

## 技术选型

| 层 | 选型 | 理由 |
|---|---|---|
| 语言 | Python 3.11+ | librosa 生态成熟 |
| STFT / 频谱 | `librosa.stft` | 一次计算，全模块复用 magnitude 谱 |
| 频谱特征 | `librosa.feature.*` | spectral_centroid / flatness / bandwidth 接受预计算 `S=` |
| 频段能量 | numpy bin slicing | 零额外开销 |
| 音频 I/O | `soundfile` | 支持 WAV/FLAC/OGG |
| CLI | `argparse` | 零依赖 |
| 测试 | `pytest` | 标准 |
| 包管理 | `uv` | 遵循用户规范 |

> [!NOTE]
> **Silero-VAD** 仅作实验对照（`experiments/`），不进 V1 主链路。

---

## 核心设计：统一 STFT 管线

```python
# 一次 STFT，全局复用
S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))  # (1025, n_frames)
freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)       # (1025,)
# 所有 detector 共享同一份 S 和 freqs
```

---

## 架构数据流

```
vocal.wav ─┐
           ├─→ [audio_loader] ─→ (y, sr) float32 mono
instr.wav ─┘                         │
                                     ▼
                            [stft_compute]  ← 统一入口
                     S_vocal (1025, T)  +  S_instr (1025, T)
                     freqs (1025,)
                                     │
                                     ▼
                        [vocal_activity_detector]
                     active_mask: bool[T]  ← 双阈值 hysteresis
                                     │
                                     ▼
                        [feature_extractor]
              ┌──────────────┼──────────────┐
              ▼                              ▼
       [harsh_detector]              [distortion_detector]
       harsh_scores[T]               distortion_scores[T]
              │                              │
              └──────────┬───────────────────┘
                         ▼
                  [segment_merger]
              DefectSegment[]
                         │
                         ▼
                  [repair_router]
              RepairPlan[]
                         │
                         ▼
          defect_segments.json + repair_plan.json
```

---

## Proposed Changes

### Core — 音频加载与 STFT

#### [NEW] [audio_loader.py](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/audio_repair/core/audio_loader.py)

- `load_audio(path) → (y: ndarray, sr: int)` — soundfile 读取，自动转 mono
- 验证采样率，warn if ∉ {44100, 48000}

#### [NEW] [stft_compute.py](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/audio_repair/core/stft_compute.py)

- `compute_stft(y, sr, n_fft, hop_length) → STFTResult(S, freqs, times, sr, hop_length)`
- 频段工具函数：
  - `band_energy(S, freqs, low, high) → ndarray[T]`
  - `band_ratio(S, freqs, target_band, ref_band) → ndarray[T]`
    - V1 中 `highband_ratio` 的 `ref_band` 固定为 `200–4000 Hz`，禁止模糊引用 `full`
  - `band_flatness(S, freqs, low, high, eps=1e-10) → ndarray[T]`
    - 数值安全：几何均值/算术均值比值必须加 `eps`
    - 频段总能量低于最小门限时回退为 0
    - 所有输出执行 `np.nan_to_num(...)`

```python
# band_flatness 参考实现
def band_flatness(S, freqs, low, high, eps=1e-10):
    band = slice_band(S, freqs, low, high)
    band = np.maximum(band, eps)
    gm = np.exp(np.mean(np.log(band), axis=0))
    am = np.mean(band, axis=0)
    out = gm / np.maximum(am, eps)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
```

---

### Core — 人声活跃检测

#### [NEW] [vocal_activity_detector.py](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/audio_repair/core/vocal_activity_detector.py)

- 基于 `librosa.feature.rms(S=S)` 的短时能量
- **自适应双阈值门限（hysteresis）**：
  - enter: RMS > 局部中位数 × `enter_ratio`（默认 1.25）
  - exit: RMS < 局部中位数 × `exit_ratio`（默认 1.10）
  - 中位数窗口 `median_window_frames = 31`
- 最小活跃段 `min_active_ms = 120`，过滤瞬态噪声
- 输出 `active_mask: ndarray[bool, (T,)]`

---

### Detection — 特征提取 + 缺陷检测

#### [NEW] [feature_extractor.py](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/audio_repair/detection/feature_extractor.py)

从 `STFTResult` 提取全部原始特征（向量化）：

| 特征 | 计算方式 |
|---|---|
| `spectral_centroid` | `librosa.feature.spectral_centroid(S=S)` |
| `spectral_flatness` | `librosa.feature.spectral_flatness(S=S)` |
| `spectral_bandwidth` | `librosa.feature.spectral_bandwidth(S=S)` |
| `harsh_band_ratio_main` | `band_ratio(S, 5000–8000, 200–4000)` |
| `harsh_band_ratio_wide` | `band_ratio(S, 4000–10000, 200–4000)` |
| `highband_ratio` | `band_ratio(S, 4000–12000, 200–4000)` |
| `flatness_air` | `band_flatness(S, 4000–12000)` |
| `rms_envelope` | `librosa.feature.rms(S=S)` |

#### [NEW] [harsh_detector.py](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/audio_repair/detection/harsh_detector.py)

5 个子指标 → 加权求和 → `harsh_score[T]`：

| 子指标 | 权重 | 说明 |
|---|---|---|
| `harsh_band_ratio_main` | 0.35 | 5–8kHz / body |
| `harsh_band_ratio_wide` | 0.20 | 4–10kHz / body |
| `persistence_score` | 0.20 | 滑动窗口内高频占优帧占比 |
| `collision_score` | 0.15 | vocal 高频 vs instr 高频重叠 |
| `centroid_drift` | 0.10 | centroid 偏离歌曲中位基线 |

**归一化策略（V1）**：
- 仅对 vocal-active frames 统计基线
- baseline = `median`，spread = `IQR`（Q3 − Q1）
- 公式：`norm_x = clip((x - median) / max(IQR, eps), 0, upper_cap)`
- `upper_cap = 3.0`，`eps = 1e-6`

#### [NEW] [distortion_detector.py](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/audio_repair/detection/distortion_detector.py)

4 个子指标 → 加权求和 → `distortion_score[T]`：

| 子指标 | 权重 | 说明 |
|---|---|---|
| `flatness_air` | 0.40 | 4–12kHz flatness |
| `highband_ratio` | 0.25 | 高频/主体能量比 |
| `bandwidth_expansion` | 0.20 | spectral bandwidth 偏离基线 |
| `breathy_penalty` | 0.15 | 低 RMS + 高 flatness 上下文加权 |

归一化策略同 harsh_detector。

---

### Pipeline — 分段与路由

#### [NEW] [segment_merger.py](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/audio_repair/pipeline/segment_merger.py)

- 连续超阈值帧 → segment；gap < `merge_gap_ms` → 合并
- 持续时间 < `min_duration_ms` → 丢弃

#### [NEW] [repair_router.py](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/audio_repair/pipeline/repair_router.py)

4 条路由规则，附加 `confidence` + `review_needed`。

#### [NEW] [analyzer.py](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/audio_repair/pipeline/analyzer.py)

主 pipeline 编排器。

---

### 类型、配置、CLI

#### [NEW] [types.py](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/audio_repair/types.py)

`@dataclass`：`STFTResult`, `FrameFeatures`, `DefectSegment`, `RepairStep`, `RepairPlan`, `AnalysisConfig`

#### [NEW] [config.py](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/audio_repair/config.py)

默认配置 + `load_config(path)` 从 JSON 覆盖。VAD 配置新增：

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

#### [NEW] [cli.py](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/audio_repair/cli.py)

```bash
python -m audio_repair analyze --vocal vocal.wav --instrumental instr.wav [--config config.json]
audio-repair analyze --vocal vocal.wav --instrumental instr.wav [--config config.json]
```

#### [NEW] [\_\_main\_\_.py](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/audio_repair/__main__.py)

```python
from .cli import main
if __name__ == "__main__":
    main()
```

---

### 项目基础设施

#### [NEW] [pyproject.toml](file:///mnt/e/vm_share/share_projects/audio-repair-node/pyproject.toml)

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
dev = ["pytest>=7", "pytest-cov"]
experiment = ["silero-vad"]

[project.scripts]
audio-repair = "audio_repair.cli:main"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
```

---

## 目录结构

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
│   ├── conftest.py
│   ├── fixtures/
│   │   ├── sine_440.wav
│   │   ├── white_noise.wav
│   │   ├── vocal_harsh_sample.wav
│   │   ├── vocal_distortion_sample.wav
│   │   └── instrumental_bright_overlap.wav
│   ├── test_audio_loader.py
│   ├── test_stft_compute.py
│   ├── test_vocal_activity_detector.py
│   ├── test_feature_extractor.py
│   ├── test_harsh_detector.py
│   ├── test_distortion_detector.py
│   ├── test_segment_merger.py
│   ├── test_repair_router.py
│   └── test_analyzer_e2e.py
├── experiments/
│   └── silero_vad_compare.py
├── validation/
│   ├── manifest.json
│   └── labels/
├── pyproject.toml
└── PRD.md
```

---

## 开发顺序

| 阶段 | 内容 | 产出 |
|---|---|---|
| **S1** | 项目初始化 + types + audio_loader + stft_compute | WAV → STFT magnitude 矩阵 |
| **S2** | vocal_activity_detector（双阈值 hysteresis） | active_mask |
| **S3** | feature_extractor + band 工具 | 全部 8 个原始特征向量 |
| **S4** | harsh_detector + distortion_detector + 归一化 | 逐帧 score |
| **S5** | segment_merger + repair_router | JSON 输出 |
| **S6** | CLI + analyzer 编排 + 全量测试 | 命令行可运行 |

---

## Verification Plan

### Automated Tests — `uv run pytest tests/ -v`

| 测试文件 | 验证内容 |
|---|---|
| `test_audio_loader` | mono 转换、采样率验证、异常文件处理 |
| `test_stft_compute` | 440Hz 正弦波能量在 body band；白噪声 flatness ≈ 1；band_flatness 无 NaN |
| `test_vocal_activity_detector` | 静音段不过门；正常 vocal 过门；极短瞬态不误判 |
| `test_feature_extractor` | 输出 shape 正确、无 NaN、范围合理 |
| `test_harsh_detector` | 高频刺耳样本 harsh_score 升高；distortion_score 不被同步高估 |
| `test_distortion_detector` | 嘶嘶/空洞样本 flatness_air 升高；distortion_score > clean baseline |
| `test_segment_merger` | 合并、gap 填充、短段过滤 |
| `test_repair_router` | 4 种路由 + confidence + review_needed |
| `test_analyzer_e2e` | 输入 WAV → 输出完整 JSON，字段齐全 |

### Validation Set 验证

```text
validation/
├── manifest.json       # 所有标注文件索引
└── labels/song01.json  # 每首 vocal stem 的人工标注段
```

每首标注格式：
```json
{
  "file": "song01_vocal.wav",
  "segments": [
    { "start_ms": 18240, "end_ms": 18780, "label": "distortion" },
    { "start_ms": 42350, "end_ms": 43180, "label": "harsh" }
  ]
}
```

V1 验证指标（至少 10 首 vocal stem）：
- **命中率**：标注的缺陷段被检出的比例
- **误报率**：非缺陷段被误标的比例
- **时间偏移误差**（ms）：检出段与标注段的起止偏差
