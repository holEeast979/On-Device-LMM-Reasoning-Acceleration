# GPT Codex Task: 实现 Adaptive kr（Layer 1 硬约束）

## 背景

FasterOmni 项目通过 GOP 级帧选择实现视频推理加速。当前的 `keep_ratio` (kr) 是固定值，存在一个严重问题：

**问题**：当视频的 I 帧数量 × kr > max_frames 时，pipeline 先选帧再截断，导致：
1. AV-LRM 精心选出的高分帧被均匀降采样覆盖（sparse 路径）
2. I 帧时间聚类导致截断后覆盖度不如均匀采样（Pareto 非单调的根因）
3. 中长视频无法受益于稀疏化（帧数仍被截到 max_frames）

**解法**：在帧选择 **之前** 计算 adaptive kr，保证选出的帧数 ≤ max_frames，消除二次截断。

公式：
```
kr_adaptive = min(kr_base, max_frames / n_candidates)
```
其中 `n_candidates` 是可选帧的数量（sparse 路径是 valid_gops 数，naive 路径也是 valid_gops 数）。

## 需要修改的文件

**唯一需要改的文件**：`fasteromni/pipeline.py`

## 具体修改点

### 1. `_select_sparse` 方法（约 line 328-442）

**当前逻辑**：
```python
# Step 3: 打分 + 选择（使用原始 keep_ratio）
scored_gops = select_gops(scored_gops, keep_ratio=keep_ratio, ...)

# Step 4: 解码后截断（问题所在！）
if max_frames > 0 and len(i_frames) > max_frames:
    indices = np.linspace(0, len(i_frames) - 1, max_frames).astype(int)
    i_frames = [i_frames[i] for i in indices]
```

**改为**：
```python
# Step 3 之前：计算 adaptive kr
n_valid = len([sg for sg in scored_gops if sg.combined_score >= 0])
if max_frames > 0 and n_valid > 0:
    kr_adaptive = min(keep_ratio, max_frames / n_valid)
else:
    kr_adaptive = keep_ratio

# Step 3: 打分 + 选择（使用 adaptive kr）
scored_gops = select_gops(scored_gops, keep_ratio=kr_adaptive, ...)

# Step 4: 解码（保留 max_frames 截断作为安全兜底，但正常情况不会触发）
# 不删除原有的截断代码，作为防御性编程保留
```

**同时在 metadata 中记录**：
```python
metadata = {
    ...
    "kr_requested": keep_ratio,      # 用户请求的 kr
    "kr_adaptive": kr_adaptive,       # 实际使用的 kr
    "adaptive_triggered": kr_adaptive < keep_ratio,  # 是否触发了自适应
}
```

### 2. `_select_naive` 方法（约 line 444-583）

**当前逻辑**（iframe_uniform 分支）：
```python
valid_gops = [g for g in gop_analysis.gops if g.num_frames >= 10]
K = max(1, math.ceil(len(valid_gops) * keep_ratio))
if max_frames > 0:
    K = min(K, max_frames)
```

这里 `min(K, max_frames)` 已经隐式实现了 adaptive kr，但没有记录。需要：

1. **计算并记录 kr_adaptive**（与 sparse 保持一致的日志格式）：
```python
valid_gops = [g for g in gop_analysis.gops if g.num_frames >= 10]
n_valid = len(valid_gops)

# Adaptive kr 计算
if max_frames > 0 and n_valid > 0:
    kr_adaptive = min(keep_ratio, max_frames / n_valid)
else:
    kr_adaptive = keep_ratio

K = max(1, math.ceil(n_valid * kr_adaptive))
# 不再需要 min(K, max_frames)，因为 kr_adaptive 已经保证 K <= max_frames
# 但保留作为安全兜底
if max_frames > 0:
    K = min(K, max_frames)
```

2. **在 metadata 中记录**：
```python
metadata = {
    ...
    "kr_requested": keep_ratio,
    "kr_adaptive": kr_adaptive,
    "adaptive_triggered": kr_adaptive < keep_ratio,
}
```

### 3. PipelineResult dataclass（约 line 90-124）

新增两个字段：
```python
kr_requested: float = 0.0        # 用户请求的 keep_ratio
kr_adaptive: float = 0.0         # 实际使用的 adaptive keep_ratio
```

### 4. `run_sparse` 和 `run_naive` 方法

在这两个方法中，从 metadata 提取 adaptive kr 信息写入 result：
```python
result.kr_requested = float(selected.metadata.get("kr_requested", 0.0))
result.kr_adaptive = float(selected.metadata.get("kr_adaptive", 0.0))
```

## 约束条件（重要！）

1. **不改变短视频行为**：当 `n_iframes * kr_base <= max_frames` 时，kr_adaptive = kr_base，行为与改前完全一致
2. **不删除 max_frames 截断代码**：保留 `if len(i_frames) > max_frames` 的截断作为安全兜底（防御性编程），但正常情况下不应该触发
3. **不改 eval_videomme.py / eval_mvbench.py**：CLI 参数不变，adaptive kr 是 pipeline 内部行为
4. **不改 `_select_baseline`、`_select_text_only` 等**：只改 sparse 和 naive 路径
5. **保持 `keep_ratio_actual` 字段不变**：这个字段记录的是实际选中比例，adaptive kr 信息用新增字段记录

## 测试验证

修改完成后，用以下命令快速验证：

### 测试 1：短视频（不触发 adaptive，结果应与改前一致）
```bash
cd /root/scripts
python -c "
from fasteromni.pipeline import SparseInferencePipeline, print_result
pipe = SparseInferencePipeline()
pipe.load_model()

# 短视频 baseline
video = '/root/autodl-tmp/data/Video-MME/videos/---QDT1gorI.mp4'
question = 'What happened in this video?'

# sparse kr=0.5
r = pipe.run_sparse(video, question, max_new_tokens=1, keep_ratio=0.5, max_frames=32)
print(f'sparse: kr_req={r.kr_requested}, kr_adapt={r.kr_adaptive}, frames={r.num_frames_input}')

# naive_iframe kr=0.5
r = pipe.run_naive(video, question, strategy='iframe_uniform', max_new_tokens=1, keep_ratio=0.5, max_frames=32)
print(f'naive: kr_req={r.kr_requested}, kr_adapt={r.kr_adaptive}, frames={r.num_frames_input}')
"
```

### 测试 2：验证 adaptive 触发（用低 max_frames 强制触发）
```bash
cd /root/scripts
python -c "
from fasteromni.pipeline import SparseInferencePipeline, print_result
pipe = SparseInferencePipeline()
pipe.load_model()

video = '/root/autodl-tmp/data/Video-MME/videos/---QDT1gorI.mp4'
question = 'What happened in this video?'

# 用 max_frames=4 强制触发 adaptive
r = pipe.run_sparse(video, question, max_new_tokens=1, keep_ratio=0.5, max_frames=4)
print(f'sparse: kr_req={r.kr_requested}, kr_adapt={r.kr_adaptive}, frames={r.num_frames_input}, adaptive={r.kr_adaptive < r.kr_requested}')

r = pipe.run_naive(video, question, strategy='iframe_uniform', max_new_tokens=1, keep_ratio=0.5, max_frames=4)
print(f'naive: kr_req={r.kr_requested}, kr_adapt={r.kr_adaptive}, frames={r.num_frames_input}, adaptive={r.kr_adaptive < r.kr_requested}')
"
```

**预期**：
- 测试 1：`kr_adaptive == kr_requested == 0.5`（短视频不触发）
- 测试 2：`kr_adaptive < kr_requested`（max_frames=4 强制触发），且 `num_frames_input <= 4`

## 文件位置

- 修改文件：`/root/scripts/fasteromni/pipeline.py`（938 行）
- 参考文件（只读）：`/root/scripts/fasteromni/modules/sparse.py`（了解 `select_gops` 的接口）
- 参考文件（只读）：`/root/scripts/fasteromni/eval_videomme.py`（了解参数传递）

## 不需要做的事

- ❌ 不需要实现 Layer 2（基于视频内容/任务类型的动态 kr）
- ❌ 不需要改 eval 脚本的 CLI 参数
- ❌ 不需要改 CSV 输出格式（adaptive kr 信息目前只记录在 PipelineResult 中）
- ❌ 不需要改 baseline / text_only / audio_only / video_only 路径
