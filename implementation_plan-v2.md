# AI Vocal De-artifact & Cleanup — V1 实现方案（Python）

## 项目定位

为 AI 生成歌曲（Suno 等）的 vocal stem 提供**自动缺陷检测与修复路由**。V1 聚焦两类缺陷：
- **刺耳类 (harsh)**：高频持续占优，烦躁/刮耳
- **失真类 (distortion)**：高频噪声化，嘶嘶/空洞/低码率感

V1 只输出 `defect_segments.json` + `repair_plan.json`，不执行实际修复。

---

## 技术选型

| 层 | 选型 | 理由 |
|---|---|---|
| 语言 | Python 3.11+ | librosa 生态成熟，音频分析首选 |
| STFT / 频谱 | `librosa.stft` | 一次计算，全模块复用 magnitude 谱 |
| 频谱特征 | `librosa.feature.*` | spectral_centroid / flatness / bandwidth 均接受预计算 `S=` 参数 |
| 频段能量 | **numpy bin slicing** | 从 magnitude 直接按 bin 切频段，零额外开销 |
| 音频 I/O | `soundfile` (`sf.read`) | 纯 C 绑定，支持 WAV/FLAC/OGG |
| 数值计算 | `numpy` / `scipy.signal` | 标准科学计算栈 |
| 人声活跃检测 | **能量包络 + 自适应门限** | RMS 短时能量 + 局部中位数自适应阈值，比固定阈值更稳 |
| 可选预处理 | 系统 `ffmpeg` | 非 WAV 输入时预转换 |
| CLI | `argparse`（标准库） | 零额外依赖 |
| 测试 | `pytest` | 标准 |
| 包管理 | `uv` | 遵循用户规范 |

> [!NOTE]
> **Silero-VAD** 仅作实验对照，不进 V1 主链路。可在 `experiments/` 下做对比测试。

---

## 核心设计：统一 STFT 管线

```python
# 一次 STFT，全局复用
S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))  # (1025, n_frames)
freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)       # (1025,)

# 所有 detector 共享同一份 S 和 freqs，不重复计算
```

这是和旧方案最大的区别：**不再逐帧逐模块各算一遍 FFT**，而是一次 STFT 产生 `(n_bins, n_frames)` 矩阵，所有检测器通过 numpy 切片取各自频段。

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
                     active_mask: bool[T]  ← 基于 RMS 自适应门限
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
              DefectSegment[]  ← 合并相邻帧、过滤短段
                         │
                         ▼
                  [repair_router]
              RepairPlan[]  ← 路由修复链
                         │
                         ▼
          defect_segments.json + repair_plan.json
```

---

## Proposed Changes

### Core — 音频加载与 STFT

#### [NEW] [audio_loader.py](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/core/audio_loader.py)

- `load_audio(path) → (y: ndarray, sr: int)` — soundfile 读取，自动转 mono
- 验证采样率，warn if ∉ {44100, 48000}

#### [NEW] [stft_compute.py](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/core/stft_compute.py)

- `compute_stft(y, sr, n_fft, hop_length) → STFTResult(S, freqs, times, sr, hop_length)`
- 一次计算，返回 dataclass 供全管线复用
- 包含 `band_energy(S, freqs, low, high) → ndarray[T]` 等频段工具函数
- 包含 `band_flatness(S, freqs, low, high) → ndarray[T]` 局部频段 flatness
- 包含 `band_ratio(S, freqs, target_band, ref_band) → ndarray[T]` 频段能量比

---

### Core — 人声活跃检测

#### [NEW] [vocal_activity_detector.py](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/core/vocal_activity_detector.py)

- 基于 `librosa.feature.rms(S=S)` 的短时能量
- **自适应门限**：局部中位数 × 系数（而非固定阈值），对不同响度段落自动适配
- 可选：最小活跃段时长过滤（去除瞬态噪声误判）
- 输出 `active_mask: ndarray[bool, (T,)]`

---

### Detection — 特征提取 + 缺陷检测

#### [NEW] [feature_extractor.py](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/detection/feature_extractor.py)

从 `STFTResult` 提取全部原始特征（向量化，不逐帧循环）：

| 特征 | 计算方式 |
|---|---|
| `spectral_centroid` | `librosa.feature.spectral_centroid(S=S)` |
| `spectral_flatness` | `librosa.feature.spectral_flatness(S=S)` |
| `spectral_bandwidth` | `librosa.feature.spectral_bandwidth(S=S)` |
| `harsh_band_ratio_main` | `band_ratio(S, 5000–8000, 200–4000)` |
| `harsh_band_ratio_wide` | `band_ratio(S, 4000–10000, 200–4000)` |
| `highband_ratio` | `band_ratio(S, 4000–12000, full)` |
| `flatness_air` | `band_flatness(S, 4000–12000)` |
| `rms_envelope` | `librosa.feature.rms(S=S)` |

#### [NEW] [harsh_detector.py](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/detection/harsh_detector.py)

5 个子指标 → 加权求和 → `harsh_score[T]`：

| 子指标 | 权重 | 说明 |
|---|---|---|
| `harsh_band_ratio_main` | 0.35 | 5–8kHz 对 body 的能量比 |
| `harsh_band_ratio_wide` | 0.20 | 4–10kHz 扩展窗口 |
| `persistence_score` | 0.20 | 滑动窗口内高频占优帧占比 |
| `collision_score` | 0.15 | vocal 高频 vs instr 高频重叠程度 |
| `centroid_drift` | 0.10 | centroid 偏离歌曲中位基线的程度 |

每个子指标先做**歌曲级归一化**（相对于该曲自身统计基线），确保阈值跨曲可迁移。

#### [NEW] [distortion_detector.py](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/detection/distortion_detector.py)

4 个子指标 → 加权求和 → `distortion_score[T]`：

| 子指标 | 权重 | 说明 |
|---|---|---|
| `flatness_air` | 0.40 | 4–12kHz flatness（越高越像噪声） |
| `highband_ratio` | 0.25 | 高频能量占比 |
| `bandwidth_expansion` | 0.20 | spectral bandwidth 偏离基线 |
| `breathy_penalty` | 0.15 | 低 RMS + 高 flatness 上下文（气声/假声加权） |

---

### Pipeline — 分段与路由

#### [NEW] [segment_merger.py](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/pipeline/segment_merger.py)

- 向量化实现：在 score 数组上找连续超阈值区间
- 相邻段 gap < `merge_gap_ms` → 合并
- 持续时间 < `min_duration_ms` → 丢弃
- 输出 `list[DefectSegment]`

#### [NEW] [repair_router.py](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/pipeline/repair_router.py)

PRD 4 条路由规则，逐 segment 判定：

```
harsh_only    → ["deess_light"]
distort_only  → ["declip_light", "voice_denoise_light?"]
both_high     → ["declip_light", "voice_denoise_light?", "deess_light"]
skip          → ["skip"]
```

附加 `confidence`（基于 score 距阈值的距离）和 `review_needed` 标记。

#### [NEW] [analyzer.py](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/pipeline/analyzer.py)

- 主 pipeline 编排器，串联所有模块
- 输入：config + file paths → 输出：写 JSON 文件

---

### 类型、配置、CLI

#### [NEW] [types.py](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/types.py)

- dataclass：`STFTResult`, `FrameFeatures`, `DefectSegment`, `RepairStep`, `RepairPlan`, `AnalysisConfig`
- 全部使用 `@dataclass` + type hints

#### [NEW] [config.py](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/config.py)

- 默认配置常量 + `load_config(path)` 从 JSON 加载覆盖

#### [NEW] [cli.py](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/cli.py)

```bash
python -m audio_repair analyze --vocal vocal.wav --instrumental instr.wav [--config config.json] [--output-dir outputs/]
```

---

### 项目基础设施

#### [NEW] [pyproject.toml](file:///mnt/e/vm_share/share_projects/audio-repair-node/pyproject.toml)

```toml
[project]
name = "audio-repair-node"
version = "0.1.0"
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
audio-repair = "src.cli:main"
```

---

## 目录结构

```
audio-repair-node/
├── src/
│   ├── __init__.py
│   ├── cli.py                          # CLI 入口
│   ├── types.py                        # 数据类型
│   ├── config.py                       # 配置加载
│   ├── core/
│   │   ├── __init__.py
│   │   ├── audio_loader.py             # 音频读取
│   │   ├── stft_compute.py             # 统一 STFT + 频段工具
│   │   └── vocal_activity_detector.py  # 人声活跃检测
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── feature_extractor.py        # 特征提取
│   │   ├── harsh_detector.py           # 刺耳检测
│   │   └── distortion_detector.py      # 失真检测
│   └── pipeline/
│       ├── __init__.py
│       ├── segment_merger.py           # 分段合并
│       ├── repair_router.py            # 修复路由
│       └── analyzer.py                 # 主 pipeline
├── configs/
│   └── repair_config.json              # 默认配置
├── tests/
│   ├── test_stft_compute.py            # 频段计算：正弦波/白噪声验证
│   ├── test_segment_merger.py          # 合并逻辑
│   ├── test_repair_router.py           # 路由规则
│   └── conftest.py                     # 共享 fixtures
├── experiments/
│   └── silero_vad_compare.py           # Silero VAD 对比实验（不进主链）
├── pyproject.toml
└── PRD.md
```

---

## 开发顺序

| 阶段 | 内容 | 产出 |
|---|---|---|
| **S1** | 项目初始化 + types + audio_loader + stft_compute | 能读 WAV → 输出 STFT magnitude 矩阵 |
| **S2** | vocal_activity_detector | 自适应门限标记 active 帧 |
| **S3** | feature_extractor + band 工具 | 全部 9 个原始特征向量 |
| **S4** | harsh_detector + distortion_detector | 逐帧 score 输出 |
| **S5** | segment_merger + repair_router | `defect_segments.json` + `repair_plan.json` |
| **S6** | CLI + analyzer 编排 + 端到端测试 | 命令行可运行完整流程 |

---

## Verification Plan

### Automated Tests

```bash
uv run pytest tests/ -v
```

| 测试 | 验证 |
|---|---|
| `test_stft_compute` | 440Hz 正弦波能量集中在 body band；白噪声 flatness ≈ 1 |
| `test_segment_merger` | 合并、gap 填充、短段过滤逻辑正确 |
| `test_repair_router` | 4 种路由 + confidence + review_needed 全覆盖 |

### Manual Verification

1. 你提供 Suno vocal stem + instrumental
2. 运行 `uv run audio-repair analyze --vocal ... --instrumental ...`
3. 检查输出 JSON 中标记的时间段，在 DAW 中定位核对听感
