# GPT Codex Task: Layer 3 Motion Vector 提取 PoC

## 背景

FasterOmni 的 Layer 3 (Motion-Aware 采样) 需要从视频 P/B 帧中提取 Motion Vector (MV) 信息，用于判断哪些时间段运动剧烈，从而做非均匀帧采样。

**目标**：写一个独立的 PoC 脚本，验证 ffmpeg 能否从视频中导出 MV 数据，并计算每个 GOP 的"运动剧烈程度"。

## 什么是 Motion Vector

H.264/H.265 编码中，P 帧和 B 帧不存储完整像素，而是存储每个宏块（16×16）相对于参考帧的位移向量 (dx, dy)，这就是 Motion Vector。

- MV 幅度大 → 该区域运动剧烈
- MV 幅度小 → 该区域静态
- MV 可以从码流中直接提取，**不需要解码像素**

## 任务

### 1. 编写 MV 提取函数

使用 ffmpeg 的 `codecview` 或 `mestimate` 滤镜导出 MV 数据。

**方案 A（推荐）：ffmpeg + codecview + 解析**

ffmpeg 有一个 `-flags2 +export_mvs` 选项可以导出 MV 到 sidedata，配合 `ffprobe` 的 `-show_frames` 可以获取。但这种方式输出量大。

**方案 B（更实用）：用 Python 的 av 库**

```python
import av

def extract_motion_vectors(video_path: str) -> list[dict]:
    """提取视频中所有 P/B 帧的 Motion Vector。
    
    Returns:
        list of dict, each dict contains:
        - frame_idx: int
        - frame_type: str ('P' or 'B')
        - timestamp_sec: float
        - mv_magnitude_mean: float  # 平均 MV 幅度
        - mv_magnitude_max: float   # 最大 MV 幅度
        - mv_count: int             # MV 数量
    """
```

PyAV (av 库) 可以通过 `frame.side_data` 获取 MV：
```python
container = av.open(video_path)
for frame in container.decode(video=0):
    for sd in frame.side_data:
        if sd.type == av.sidedata.MotionVectors:
            # sd.to_ndarray() 返回结构化数组
            # 每个 MV 有: source, w, h, src_x, src_y, dst_x, dst_y, flags
            mvs = sd.to_ndarray()
            dx = mvs['dst_x'] - mvs['src_x']
            dy = mvs['dst_y'] - mvs['src_y']
            magnitude = np.sqrt(dx**2 + dy**2)
```

**注意**：需要在打开容器时启用 MV 导出：
```python
container = av.open(video_path)
container.streams.video[0].codec_context.export_mvs = True  # 关键！
# 或者
container = av.open(video_path, options={'flags2': '+export_mvs'})
```

### 2. 按 GOP 聚合 MV

结合现有的 `parse_gops()` 返回的 GOP 信息，按 GOP 聚合 MV 统计：

```python
def compute_gop_motion_scores(video_path: str, gops: list) -> list[float]:
    """计算每个 GOP 的运动评分。
    
    Args:
        video_path: 视频文件路径
        gops: parse_gops() 返回的 GOP 列表
        
    Returns:
        list of float, 每个 GOP 的运动评分（MV 平均幅度）
    """
```

### 3. 可视化

生成一张图，展示视频时间轴上的 MV 幅度分布：
- X 轴：时间 (秒)
- Y 轴：MV 平均幅度
- 用垂直线标注 GOP 边界
- 颜色区分高运动 / 低运动区域

### 输出

脚本：`/root/scripts/tools/mv_extraction_poc.py`

运行方式：
```bash
cd /root/scripts
python tools/mv_extraction_poc.py /root/autodl-tmp/data/Video-MME/videos/data/323v_FtWqvo.mp4
```

输出：
- 终端打印每个 GOP 的运动评分
- 生成可视化图 `/root/autodl-tmp/results/figures/mv_profile_<video_name>.png`

## 依赖

```bash
pip install av numpy matplotlib
```

`av` (PyAV) 通常已预装在 CUDA 环境中。如果没有，用 `pip install av`。

## 已有代码可复用

- GOP 解析：`from fasteromni.modules.gop_parser import parse_gops`
- 该函数返回 `GOPAnalysis` 对象，包含 `gops` 列表，每个 GOP 有 `start_frame`, `end_frame`, `start_time_sec`, `end_time_sec`

## 重要约束

1. **不需要 GPU**，纯 CPU 操作
2. **不修改现有代码**，只新增 `tools/mv_extraction_poc.py`
3. 如果 `av` 库不支持 MV 导出（某些版本可能不支持），fallback 到 ffmpeg 命令行方案
4. 脚本应该能处理没有 P/B 帧的视频（全 I 帧编码）→ 返回空结果
5. 打印执行耗时，验证 MV 提取的性能开销

## 期望输出示例

```
Video: 323v_FtWqvo.mp4
Duration: 45.2s, Total frames: 1356, GOPs: 68

MV Extraction: 234ms (0.17ms/frame)

GOP Motion Scores:
  GOP  0 [0.0s - 0.7s]:  score=2.3  (low motion)
  GOP  1 [0.7s - 1.3s]:  score=5.8  (moderate)
  GOP  2 [1.3s - 2.0s]:  score=15.2 (HIGH MOTION) ★
  ...
  GOP 67 [44.5s - 45.2s]: score=1.1  (low motion)

Top-5 highest motion GOPs: [2, 15, 31, 42, 55]
Bottom-5 lowest motion GOPs: [0, 3, 12, 50, 67]

Motion profile saved to: /root/autodl-tmp/results/figures/mv_profile_323v_FtWqvo.png
```
