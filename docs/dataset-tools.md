# Dataset Tools

数据集准备工具，用于下载、处理和验证实验所需的多模态数据集。

## 支持的数据集

| 数据集 | 脚本 | 说明 |
|--------|------|------|
| Video-MME | `prepare_video_mme.py` | 视频理解基准，需 yt-dlp 下载 |
| AudioCaps | `prepare_audiocaps.py` | 音频描述数据集 |
| MSVD-QA | `prepare_msvd_qa.py` | 视频问答数据集 |
| ActivityNet-QA | `generate_activitynet_manifest.py` | 长视频活动理解 |

## 快速开始

```bash
# 准备 Video-MME (前100个样本)
python tools/prepare_video_mme.py --out-root /root/autodl-tmp/data --max-samples 100

# 准备 AudioCaps
python tools/prepare_audiocaps.py --out-root /root/autodl-tmp/data --max-samples 100

# 准备 MSVD-QA
python tools/prepare_msvd_qa.py --out-root /root/autodl-tmp/data --max-samples 100
```

## Manifest 格式

所有数据集统一输出 `manifest.csv`，格式如下：

| 列名 | 类型 | 说明 |
|------|------|------|
| `sample_id` | string | 样本唯一标识 |
| `video_path` | string | 视频文件绝对路径 |
| `audio_path` | string | 音频文件路径（可选） |
| `question` | string | 问题文本 |
| `answer` | string | 答案（可选） |
| `duration_s` | float | 媒体时长（秒） |
| `has_audio` | bool | 是否包含音频轨道 |

## 工具集 (tools/datasets/)

### build_manifest_from_dir.py

从本地目录扫描媒体文件，自动生成 manifest：

```bash
python tools/datasets/build_manifest_from_dir.py \
    --input-dir /path/to/videos \
    --output manifest.csv \
    --default-question "Describe this video."
```

### merge_manifests.py

合并多个 manifest 文件：

```bash
python tools/merge_manifests.py \
    --inputs a.csv b.csv c.csv \
    --output merged.csv
```

## 依赖

- `yt-dlp`: 用于下载 YouTube 视频
- `ffprobe`: 用于探测媒体信息（时长、音频轨道）
- `pandas`: 数据处理
