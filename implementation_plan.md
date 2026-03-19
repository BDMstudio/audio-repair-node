# AI Vocal De-artifact & Cleanup — V1 实现方案

## 项目定位

为 AI 生成歌曲（Suno 等）的 vocal stem 提供**自动缺陷检测与修复路由**。V1 聚焦两类缺陷：
- **刺耳类 (harsh)**：高频持续占优，烦躁/刮耳
- **失真类 (distortion)**：高频噪声化，嘶嘶/空洞/低码率感

V1 不做：全曲母带、音准修复、节奏修复、审美判断、编曲重建。

---

## User Review Required

> [!IMPORTANT]
> **技术栈选型**：项目名为 `audio-repair-node`，方案采用 **Node.js + TypeScript**。若你更倾向 Python（librosa 生态更成熟），请明确告知。

> [!IMPORTANT]
> **音频特征库选择**：V1 推荐 **Meyda**（轻量、纯 JS、支持离线逐帧提取 spectralCentroid / spectralFlatness / spectralSpread）。备选 Essentia.js（WASM、算法更全但更重）。Meyda 不直接提供"指定频段能量比"计算，需要我们在 FFT 结果上手动切 band 计算 `harsh_band_ratio` 和 `highband_ratio`——这完全可行，但需确认你接受这个方案。

> [!IMPORTANT]
> **V1 不含实际修复执行**：V1 只输出 `defect_segments.json` + `repair_plan.json`，不集成 RX / pure:deess 自动执行。实际修复需用户手动在 DAW 中按 repair_plan 操作。是否符合预期？

> [!WARNING]
> **WAV 读取**：使用 `wav-decoder`（纯 JS 解码 WAV）或 `audiobuffer-to-wav` + `node-web-audio-api`。若输入可能包含 mp3/flac 等格式，需额外依赖 `ffmpeg` 做预转换。V1 是否只支持 WAV 输入？

---

## 技术选型

| 层 | 选型 | 理由 |
|---|---|---|
| 语言 | TypeScript (Node.js) | 项目名 `audio-repair-node`，强类型保障 |
| 音频解码 | `wav-decoder` | 纯 JS，零 native 依赖，解码 WAV → Float32Array |
| 特征提取 | `Meyda` | 轻量，支持离线帧提取，提供 spectralCentroid / spectralFlatness / spectralSpread |
| FFT / 频段计算 | 手写工具函数 | 基于 Meyda 暴露的 FFT buffer 或自行做 FFT，按频段切 bin 计算能量比 |
| 人声活跃检测 | RMS 阈值 + ZCR | 过滤静音/极低能量帧 |
| CLI | `commander` | 标准 Node.js CLI 框架 |
| 配置 | JSON 文件 | 直接复用 PRD 的 `repair_config.json` 模板 |
| 测试 | `vitest` | 快、TS 原生支持 |
| 构建 | `tsup` | 零配置 TS 打包 |
| 包管理 | `pnpm` | 遵循用户规范 |

---

## 架构概览

```
vocal.wav ─┐
           ├─→ [AudioLoader] ─→ Float32Array (mono, 44.1k)
instr.wav ─┘                         │
                                     ▼
                              [FrameSplitter]
                            (2048 samples, hop 512)
                                     │
                                     ▼
                           [VocalActivityDetector]
                          (RMS > threshold → active)
                                     │
                                     ▼
                           [FeatureExtractor]
                    ┌────────────────┼────────────────┐
                    ▼                                  ▼
             [HarshDetector]                   [DistortionDetector]
          harsh_band_ratio_5_8              flatness_air (4-12k)
          harsh_band_ratio_4_10             highband_ratio
          persistence_score                 bandwidth_expansion
          collision_score                   breathy_penalty
          centroid_drift
                    │                                  │
                    └────────────┬─────────────────────┘
                                 ▼
                          [ScoreAggregator]
                     (加权求和 → harsh_score, distortion_score)
                                 │
                                 ▼
                          [SegmentMerger]
                  (相邻缺陷帧合并，过滤短段)
                                 │
                                 ▼
                          [RepairRouter]
                     (按阈值路由修复链)
                                 │
                                 ▼
              defect_segments.json + repair_plan.json
```

---

## Proposed Changes

### Core — 音频加载

#### [NEW] [audio-loader.ts](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/core/audio-loader.ts)

- 读取 WAV 文件 → 解码为 `{ sampleRate, channelData: Float32Array }`
- 自动转 mono（多声道取均值）
- 验证采样率（warn if != 44100 / 48000）

---

### Core — 帧处理

#### [NEW] [frame-splitter.ts](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/core/frame-splitter.ts)

- 将 Float32Array 按 `frameSize=2048`, `hopSize=512` 切帧
- 返回帧迭代器 `Generator<{ index, samples, timeMs }>`

#### [NEW] [vocal-activity-detector.ts](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/core/vocal-activity-detector.ts)

- 基于 RMS 阈值判断帧是否为 vocal-active
- 可选：结合 ZCR 过滤纯噪声帧

---

### Core — 特征提取

#### [NEW] [feature-extractor.ts](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/core/feature-extractor.ts)

- 对每个 vocal-active 帧提取以下特征：
  - **Meyda 直出**：`spectralCentroid`, `spectralFlatness`, `spectralSpread`
  - **手动计算**：`harsh_band_ratio_5_8`, `harsh_band_ratio_4_10`, `highband_ratio`, `flatness_air`（对 FFT magnitude 按频段切 bin，计算能量比和局部 flatness）
- 同时对 `instrumental.wav` 提取对应帧的高频能量，用于 `collision_score`

#### [NEW] [band-utils.ts](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/core/band-utils.ts)

- FFT bin → 频率映射工具
- `bandEnergy(fftMag, freqLow, freqHigh, sampleRate)` — 指定频段能量
- `bandFlatness(fftMag, freqLow, freqHigh, sampleRate)` — 指定频段 spectral flatness
- `bandRatio(fftMag, targetBand, totalBand, sampleRate)` — 频段能量占比

---

### Detection — 缺陷检测

#### [NEW] [harsh-detector.ts](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/detection/harsh-detector.ts)

- 输入：逐帧特征序列 + 伴奏特征序列
- 计算 5 个子指标：
  1. `harsh_band_ratio_main` (5-8kHz / body)
  2. `harsh_band_ratio_wide` (4-10kHz / body)
  3. `persistence_score` (连续高频帧占比)
  4. `collision_score` (vocal 高频 vs instrumental 高频重叠)
  5. `centroid_drift` (spectral centroid 偏离歌曲基线程度)
- 加权求和 → `harsh_score`

#### [NEW] [distortion-detector.ts](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/detection/distortion-detector.ts)

- 输入：逐帧特征序列
- 计算 4 个子指标：
  1. `flatness_air` (4-12kHz spectral flatness)
  2. `highband_ratio` (高频能量占比)
  3. `bandwidth_expansion` (spectral spread 偏离基线)
  4. `breathy_penalty` (基于 RMS 包络特征的气声/假声上下文加权)
- 加权求和 → `distortion_score`

---

### Pipeline — 分段与路由

#### [NEW] [segment-merger.ts](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/pipeline/segment-merger.ts)

- 将连续缺陷帧合并为 segment（gap < `merge_gap_ms` 则合并）
- 过滤持续时间 < `min_duration_ms` 的 segment
- 输出 `DefectSegment[]`

#### [NEW] [repair-router.ts](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/pipeline/repair-router.ts)

- 按 PRD 路由规则：
  - harsh only → `["deess_light"]`
  - distortion only → `["declip_light", "voice_denoise_light?"]`
  - both high → `["declip_light", "voice_denoise_light?", "deess_light"]`
  - skip → `["skip"]`
- 添加 confidence 和 review_needed 标记

#### [NEW] [analyzer.ts](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/pipeline/analyzer.ts)

- **主 pipeline 编排器**，串联上述所有模块
- 输入：config + 文件路径 → 输出：`defect_segments.json` + `repair_plan.json`

---

### CLI 入口

#### [NEW] [cli.ts](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/cli.ts)

- `npx audio-repair-node analyze --vocal vocal.wav --instrumental instr.wav [--config config.json]`
- 解析参数、加载配置、调用 analyzer、写入输出文件

---

### 类型与配置

#### [NEW] [types.ts](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/types.ts)

- 所有核心接口定义：`AudioData`, `Frame`, `FrameFeatures`, `HarshMetrics`, `DistortionMetrics`, `DefectSegment`, `RepairPlan`, `AnalysisConfig`

#### [NEW] [default-config.ts](file:///mnt/e/vm_share/share_projects/audio-repair-node/src/default-config.ts)

- 默认配置，直接从 PRD JSON 模板转为 TS 常量

---

### 项目基础设施

#### [NEW] [package.json](file:///mnt/e/vm_share/share_projects/audio-repair-node/package.json)

```json
{
  "name": "audio-repair-node",
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "build": "tsup src/cli.ts --format esm --dts",
    "dev": "tsx src/cli.ts",
    "test": "vitest run",
    "test:watch": "vitest"
  }
}
```

依赖清单：
- `meyda` — 音频特征提取
- `wav-decoder` — WAV 解码
- `commander` — CLI
- `tsx` — 开发运行
- `tsup` — 构建
- `typescript` — 类型
- `vitest` — 测试

#### [NEW] [tsconfig.json](file:///mnt/e/vm_share/share_projects/audio-repair-node/tsconfig.json)

- `strict: true`, `target: ES2022`, `module: ESNext`

---

## 目录结构

```
audio-repair-node/
├── src/
│   ├── cli.ts                          # CLI 入口
│   ├── types.ts                        # 类型定义
│   ├── default-config.ts               # 默认配置
│   ├── core/
│   │   ├── audio-loader.ts             # WAV 读取
│   │   ├── frame-splitter.ts           # 帧切分
│   │   ├── vocal-activity-detector.ts  # 人声活跃检测
│   │   ├── feature-extractor.ts        # 特征提取
│   │   └── band-utils.ts              # 频段计算工具
│   ├── detection/
│   │   ├── harsh-detector.ts           # 刺耳检测
│   │   └── distortion-detector.ts      # 失真检测
│   └── pipeline/
│       ├── segment-merger.ts           # 分段合并
│       ├── repair-router.ts            # 修复路由
│       └── analyzer.ts                 # 主 pipeline
├── configs/
│   └── repair_config.json              # 默认配置文件
├── tests/
│   ├── band-utils.test.ts              # 频段计算单测
│   ├── segment-merger.test.ts          # 分段合并单测
│   ├── repair-router.test.ts           # 路由逻辑单测
│   └── fixtures/                       # 测试用音频片段
├── package.json
├── tsconfig.json
└── PRD.md
```

---

## 开发顺序（对应 PRD P0）

| 阶段 | 内容 | 产出 |
|---|---|---|
| **S1** | 项目初始化 + 类型定义 + 音频加载 | 能读取 WAV 并输出 Float32Array |
| **S2** | 帧切分 + 人声活跃检测 | 能标记 vocal-active 帧 |
| **S3** | 频段工具 + 特征提取 | 能逐帧输出所有 9 个子指标 |
| **S4** | harsh / distortion 检测器 + 加权评分 | 能输出逐帧 score |
| **S5** | 分段合并 + 修复路由 | 输出 `defect_segments.json` + `repair_plan.json` |
| **S6** | CLI 封装 + 端到端测试 | 可从命令行运行完整流程 |

---

## Verification Plan

### Automated Tests

运行命令：`pnpm test`

| 测试 | 验证内容 |
|---|---|
| `band-utils.test.ts` | 正弦波在正确频段有能量、其他频段趋近 0；flatness 对白噪声 ≈ 1、对纯音 ≈ 0 |
| `segment-merger.test.ts` | 相邻帧正确合并、gap 内帧正确合并、短段被过滤 |
| `repair-router.test.ts` | 4 种路由规则（harsh only / distortion only / both / skip）正确路由；confidence 和 review_needed 标记正确 |
| `frame-splitter.test.ts` | 帧数 = `ceil((len - frameSize) / hopSize) + 1`；时间戳正确 |

### Manual Verification

> [!NOTE]
> 需要你提供一个真实的 Suno 生成的 vocal stem 作为测试素材。我可以用它跑完整 pipeline，输出 `defect_segments.json` 后你人工核对检测结果是否与听感一致。

1. 准备测试素材：`vocal.wav` + `instrumental.wav`（从 Suno 分轨）
2. 运行：`pnpm dev analyze --vocal inputs/vocal.wav --instrumental inputs/instrumental.wav`
3. 检查 `outputs/defect_segments.json` 中标记的时间段
4. 在 DAW 中定位这些时间段，人工确认是否确实存在刺耳/失真问题
5. 确认无明显误报（正常段被标为缺陷）或漏报（有缺陷段未被标记）
