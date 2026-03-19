三份可落地的东西：

1. **PRD 精简版**
2. **SKILL.md 风格执行规范**
3. **JSON 配置模板**

先说一句难听但有用的话：
**V1 不要做“万能去 AI 味引擎”，那是典型自嗨型需求。**
先老老实实只做两类缺陷：**刺耳类** 和 **失真类**。
这两类已经够值钱，而且能和 RX 的批处理 / Module Chain 思路对上；RX 11 的发布说明也明确提到 Repair Assistant 生成 Module Chain、De-clip 分析改进，这正适合做“检测后路由处理”的半自动链路。

---

# 一、PRD 精简版

## 1. 项目名

**AI Vocal De-artifact & Cleanup**

## 2. 目标

对 AI 生成歌曲的 `vocal stem` 做自动检测与修复路由，优先处理两类问题：

* **刺耳类**：高频长期占优，和伴奏高频撞车，导致烦躁、刮耳、发紧
* **失真类**：高频噪声化，出现空洞、嘶嘶、老 MP3 感

V1 只做：

* 检测
* 分段
* 路由
* 输出处理建议或执行计划

V1 不做：

* 全曲自动母带
* 音准修复
* 节奏修复
* 审美判断
* 编曲重建

## 3. 核心用户场景

用户在 Suno 里选中一个满意版本，但发现：

* 副歌假声、气声发刺
* 高频亮部带“AI 数码毛边”
* 人声空间感脏、糊、空
* 某些高音段像低码率 mp3

用户不想学 DAW，只想：

* 扔进 stem
* 自动检测问题段
* 给出修复链或自动跑一遍轻修复
* 导出更自然的 vocal 结果

## 4. 输入

必需：

* `vocal.wav`
* `instrumental.wav`

可选：

* `full_mix.wav`
* `segments.json`
* `lyrics_timestamps.json`
* `bpm.json`

## 5. 输出

必需：

* `defect_segments.json`
* `repair_plan.json`

可选：

* `vocal.cleaned.wav`
* `preview_report.md`

## 6. 检测依据

V1 以规则为主，不先上黑盒模型。

### 刺耳类代理指标

* `5–8 kHz` / `4–10 kHz` 高频占比
* 持续时长
* 与伴奏高频重叠程度
* spectral centroid 持续偏高

### 失真类代理指标

* `4–12 kHz` 高频带 spectral flatness
* 高频占比
* spectral bandwidth 异常扩散
* 气声 / 假声 / 高音长音上下文加权

其中，`spectral flatness` 用来衡量声音更像噪声还是更像音调；`spectral centroid` 是频谱重心；`spectral bandwidth` 是围绕重心的扩散程度。librosa 对这三者的定义都很清楚。

## 7. 修复路由

* 刺耳类 → `deess_light`
* 失真类 → `declip_light` → `denoise_light?`
* 双高 → `declip_light` → `denoise_light?` → `deess_light`

## 8. 成功标准

修复后满足以下至少 3 条：

* 齿、刺、尖显著下降
* 嘶嘶感、空洞感下降
* 人声更干净
* 不明显塑料化
* 不明显闷掉
* 放回伴奏后仍自然

## 9. 失败标准

任一成立即判失败：

* 2–4 kHz 主体带塌陷太多
* 字头字尾糊掉
* 人声像水下或塑料袋
* A/B 对比自然度下降

---

# 二、SKILL.md 风格执行规范

下面这版可以直接当子代理规范。

```md
# SKILL: ai-vocal-deartifact-cleanup

## Role
你是一个自动音频修复子节点，负责对 AI 生成歌曲的 vocal stem 进行缺陷检测、分段、路由和轻修复计划生成。
你的目标不是“美化一切”，而是优先识别并压制两类重复性问题：
1. 刺耳类（harsh / sibilant / fake-bright）
2. 失真类（clip-like / noisy-high-band / low-bitrate-like）

## Inputs
Required:
- vocal.wav
- instrumental.wav

Optional:
- full_mix.wav
- segment metadata
- lyric timestamp metadata
- bpm metadata

## Outputs
Required:
- defect_segments.json
- repair_plan.json

Optional:
- vocal.cleaned.wav
- preview_report.md

## Core Principles
1. 只处理 vocal stem，不在 full mix 上做主判断
2. 先检测，再路由，不允许盲目全局修复
3. 默认只做 light 级别处理
4. 如果无法确认，则标记 review_needed，不强行处理
5. 任何处理都必须保留 raw/light/medium 回退版本

## Defect Classes

### Class A: harsh
Definition:
- 高频持续占优
- 与伴奏高频亮元素重叠
- 听感烦躁、刺、刮耳

Primary signals:
- harsh_band_ratio_5_8
- harsh_band_ratio_4_10
- persistence_score
- collision_score
- centroid_drift

### Class B: distortion
Definition:
- 高频噪声化
- 像低码率 mp3 或数字嘶嘶感
- 常见于气声、假声、高音长音

Primary signals:
- flatness_air
- highband_ratio
- bandwidth_expansion
- breathy_penalty

## Default Thresholds
- T_h = 0.60
- T_h_low = 0.45
- T_d = 0.58
- T_d_low = 0.42

## Routing Rules

### Rule 1
If harsh_score >= T_h and distortion_score < T_d_low:
route = ["deess_light"]

### Rule 2
If distortion_score >= T_d and harsh_score < T_h_low:
route = ["declip_light"]
If noise_modifier is high:
append ["voice_denoise_light"]

### Rule 3
If harsh_score >= T_h and distortion_score >= T_d:
route = ["declip_light", "voice_denoise_light?", "deess_light"]

### Rule 4
If both scores are below threshold:
route = ["skip"]

## Safety Rules
- Never process the entire track at medium level by default
- Never apply dereverb unless space_modifier is confirmed
- If centroid drops too much after repair, fallback to lighter version
- If 2–4 kHz body band collapses, revert
- If repaired segment sounds duller but not cleaner, revert

## Review Triggers
Mark `review_needed: true` if:
- harsh_score and distortion_score are both unstable
- segment duration < 120 ms but score is extreme
- accompaniment bleed is too strong
- vocal stem quality is too poor to separate defect type

## Report Style
For each segment, output:
- start/end
- primary_class
- harsh_score
- distortion_score
- modifiers
- recommended_route
- confidence
- review_needed

## Notes
- Use relative thresholds per song baseline
- Use vocal-active frames only
- Merge adjacent defect frames into repair segments
- Prefer under-processing to over-processing
```

---

# 三、JSON 配置模板

这份你可以直接塞给 agent 或 pipeline。

```json
{
  "node_name": "ai_vocal_deartifact_cleanup",
  "version": "0.1.0",
  "input": {
    "vocal_path": "inputs/vocal.wav",
    "instrumental_path": "inputs/instrumental.wav",
    "full_mix_path": "inputs/full_mix.wav",
    "segments_path": "inputs/segments.json",
    "lyrics_timestamps_path": "inputs/lyrics_timestamps.json"
  },
  "analysis": {
    "frame_size": 2048,
    "hop_size": 512,
    "use_vocal_active_frames_only": true,
    "bands": {
      "body_band": [200, 4000],
      "harsh_band_main": [5000, 8000],
      "harsh_band_wide": [4000, 10000],
      "air_noise_band": [4000, 12000]
    }
  },
  "features": {
    "harsh": {
      "enable_harsh_band_ratio_main": true,
      "enable_harsh_band_ratio_wide": true,
      "enable_persistence_score": true,
      "enable_collision_score": true,
      "enable_centroid_drift": true
    },
    "distortion": {
      "enable_flatness_air": true,
      "enable_highband_ratio": true,
      "enable_bandwidth_expansion": true,
      "enable_breathy_penalty": true
    }
  },
  "weights": {
    "harsh_score": {
      "harsh_band_ratio_main": 0.35,
      "harsh_band_ratio_wide": 0.20,
      "persistence_score": 0.20,
      "collision_score": 0.15,
      "centroid_drift": 0.10
    },
    "distortion_score": {
      "flatness_air": 0.40,
      "highband_ratio": 0.25,
      "bandwidth_expansion": 0.20,
      "breathy_penalty": 0.15
    }
  },
  "thresholds": {
    "harsh_high": 0.60,
    "harsh_low": 0.45,
    "distortion_high": 0.58,
    "distortion_low": 0.42,
    "min_harsh_duration_ms": 220,
    "min_distortion_duration_ms": 180,
    "merge_gap_ms_harsh": 120,
    "merge_gap_ms_distortion": 100
  },
  "routing": {
    "harsh_only": ["deess_light"],
    "distortion_only": ["declip_light", "voice_denoise_light?"],
    "both_high": ["declip_light", "voice_denoise_light?", "deess_light"],
    "skip": ["skip"]
  },
  "modifiers": {
    "enable_noise_modifier": true,
    "enable_space_modifier": true
  },
  "safety": {
    "export_versions": ["raw", "light", "medium"],
    "fallback_on_body_band_loss": true,
    "fallback_on_centroid_collapse": true,
    "fallback_on_over_smoothing": true,
    "prefer_light_processing": true
  },
  "output": {
    "defect_segments_path": "outputs/defect_segments.json",
    "repair_plan_path": "outputs/repair_plan.json",
    "cleaned_vocal_path": "outputs/vocal.cleaned.wav",
    "preview_report_path": "outputs/preview_report.md"
  }
}
```

---

# 四、repair_plan.json 结构建议

这个是给后续执行器用的，不是给人看的。

```json
[
  {
    "segment_id": "seg_0007",
    "start_ms": 18240,
    "end_ms": 18780,
    "primary_class": "distortion",
    "harsh_score": 0.41,
    "distortion_score": 0.77,
    "modifiers": ["noise_high", "breathy_context"],
    "recommended_route": [
      {
        "tool": "rx",
        "action": "declip_light"
      },
      {
        "tool": "rx",
        "action": "voice_denoise_light"
      }
    ],
    "confidence": 0.81,
    "review_needed": false
  },
  {
    "segment_id": "seg_0013",
    "start_ms": 42350,
    "end_ms": 43180,
    "primary_class": "harsh",
    "harsh_score": 0.78,
    "distortion_score": 0.32,
    "modifiers": ["collision_high"],
    "recommended_route": [
      {
        "tool": "pure_deess",
        "action": "deess_light"
      }
    ],
    "confidence": 0.74,
    "review_needed": false
  }
]
```

---

# 五、工具映射建议

这里不要贪多，就两个核心处理器够了。

## 方案 1：偏自动、少按钮

* `RX 11`：负责 `declip / denoise / dereverb / batch`
* `pure:deess`：负责 `deess_light`

理由很简单：
RX 天生适合批处理和模块链；pure:deess 的定位就是 **AI-powered detection and processing** 加 **automatic parametrization**，更像“插上就能跑”的去齿音节点。

## 方案 2：全走 RX

* `RX Repair Assistant`
* `De-clip`
* `Voice De-noise`
* 可选 `De-reverb`

这套少一个插件，但 de-ess 颗粒度和自动感会差一点。

---

# 六、目录结构建议

```text
audio-repair-node/
├─ inputs/
│  ├─ vocal.wav
│  ├─ instrumental.wav
│  ├─ full_mix.wav
│  └─ segments.json
├─ configs/
│  └─ repair_config.json
├─ outputs/
│  ├─ defect_segments.json
│  ├─ repair_plan.json
│  ├─ vocal.cleaned.wav
│  └─ preview_report.md
├─ skills/
│  └─ SKILL.md
└─ logs/
   └─ run.log
```

---

# 七、V1 开发优先级

别乱开支线，按这个顺序做：

### P0

* vocal-active frame 过滤
* harsh / distortion 两类检测
* 分段合并
* repair_plan 输出

### P1

* modifiers：noise / space
* 处理后 A/B 安全检查
* light / medium 自动回退

### P2

* 引入 BPM、主副歌上下文
* 引入歌词时间戳，定位特定字词
* 做 UI 或 CLI 报告

---
