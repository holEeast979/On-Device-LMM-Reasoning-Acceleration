# GPT 代码任务 B：MVBench 评估接入

## 目标

创建 `fasteromni/eval_mvbench.py`，将 MVBench benchmark 接入现有 pipeline，复用 `pipeline.py` 的所有推理模式。

## 前置条件

- 模型：`/root/autodl-tmp/Qwen2.5-Omni-7B`
- MVBench 数据：`/root/autodl-tmp/data/MVBench/`
  - JSON 标注：`json/` 目录，20 个 JSON 文件（每个 200 题，共 4,000 题）
  - 视频文件：`video/` 目录，11 个子目录，3,333 个视频
- 现有框架：`fasteromni/eval_videomme.py`（参考模板）
- pipeline：`fasteromni/pipeline.py` 中的 `SparseInferencePipeline`

## MVBench 数据格式

### JSON 结构
每个 JSON 文件是一个 list，每个 item：
```json
{
  "video": "166583.webm",           // 视频文件名或相对路径
  "question": "What is the action performed by the person in the video?",
  "candidates": ["Not sure", "Scattering something down", "Piling something up"],
  "answer": "Piling something up"   // 正确答案是候选项的文本（不是字母！）
}
```

### 关键差异（vs Video-MME）
1. **答案格式**：Video-MME 的 answer 是字母 "A/B/C/D"，MVBench 的 answer 是候选项文本
2. **候选项数量不固定**：1-5 个不等（大多数是 2-4 个）
3. **无 duration 分类**：所有视频都是短视频（5-20s），不需要按时长分组
4. **有 task_type**：20 个任务类型，来自 JSON 文件名

### 视频路径解析（重要！）
视频文件在 `video/` 下的嵌套子目录中，路径解析逻辑：

```python
def resolve_video_path(video_field: str, video_base: str) -> Optional[str]:
    """
    解析 JSON 中的 video 字段到实际文件路径。
    
    video_field 可能是：
    - 纯文件名："166583.webm" -> video/ssv2_video/166583.webm
    - 带子目录："left/4504_frame52.mp4" -> video/vlnqa/left/4504_frame52.mp4
    
    策略：遍历 video_base 下的每个顶级子目录，尝试拼接。
    """
    for subdir in os.listdir(video_base):
        candidate = os.path.join(video_base, subdir, video_field)
        if os.path.exists(candidate):
            return candidate
    # 兜底：递归搜索（仅用文件名）
    basename = os.path.basename(video_field)
    for root_d, dirs, files in os.walk(video_base):
        if basename in files:
            return os.path.join(root_d, basename)
    return None
```

### 不可用的任务（跳过）
- `episodic_reasoning`：视频是帧目录（TVQA frames），不是视频文件
- `fine_grained_pose`：需要 NTU RGB+D 数据集，未完整下载

**实际可用：18 个任务 × 200 题 = 3,600 题**

## 需要实现的内容

### 1. 数据加载

```python
@dataclass
class MVBenchSample:
    task_type: str          # JSON 文件名（去 .json），如 "action_antonym"
    video_field: str        # JSON 中原始 video 字段
    video_path: str         # 解析后的完整路径
    question: str
    candidates: List[str]   # 原始候选项列表
    answer_text: str        # 正确答案文本
    answer_letter: str      # 转换后的字母 (A/B/C/D/E)
    sample_id: str          # 唯一 ID: "{task_type}_{index}"

def load_mvbench_samples(
    json_dir: str = "/root/autodl-tmp/data/MVBench/json/",
    video_dir: str = "/root/autodl-tmp/data/MVBench/video/",
    skip_tasks: List[str] = ["episodic_reasoning", "fine_grained_pose"],
    max_per_task: int = 0,  # 0 = all
) -> List[MVBenchSample]:
    ...
```

注意：
- `answer_letter` 通过在 `candidates` 中查找 `answer_text` 的索引来确定（A=0, B=1, C=2...）
- 如果 `answer_text` 不在 `candidates` 中，跳过该题并打印警告
- 视频路径解析失败的题目也跳过

### 2. Prompt 格式化

```python
def format_mvbench_prompt(question: str, candidates: List[str]) -> str:
    """
    格式化 MVBench 选择题 prompt。
    
    输出格式：
    {question}
    A. {candidate_0}
    B. {candidate_1}
    C. {candidate_2}
    ...
    Answer with the option letter only.
    """
```

动态生成选项字母（A/B/C/D/E），因为候选项数量不固定。

### 3. 答案提取

完全复用 `eval_videomme.py` 中的 `extract_answer_letter()` 函数。
但需要扩展支持 E（5 个选项时）：

```python
VALID_LETTERS = set("ABCDE")  # MVBench 最多 5 个选项
```

### 4. 评估记录

```python
@dataclass
class MVBenchRecord:
    sample_id: str
    task_type: str
    video_field: str
    mode: str
    keep_ratio: float = 0.0
    alpha: float = 0.0
    gt_answer: str = ""      # 字母
    pred_answer: Optional[str] = None
    pred_raw: str = ""
    correct: bool = False
    generate_ms: float = 0.0
    total_ms: float = 0.0
    visual_tokens: int = 0
    audio_tokens: int = 0
    total_tokens: int = 0
    num_frames: int = 0
    error: str = ""
```

### 5. 单条评估 & 批量评估

参考 `eval_videomme.py` 的 `run_single()` 和 `run_evaluation()`：
- 完全复用 pipeline 的所有 mode（baseline/sparse/naive_iframe/text_only/audio_only/video_only）
- **不需要 max_frames 参数**（MVBench 视频都 <30s，不会 OOM）——但保留参数默认 0（无限制）
- 使用相同的超时保护（`_Timeout`）和增量 CSV 机制
- 每条输出进度（task_type + 准确率）

### 6. 结果汇总

按 task_type 分组汇总（而不是 duration）：

```python
def summarize_mvbench(records: List[MVBenchRecord], label: str) -> Dict:
    """
    汇总 MVBench 结果。
    
    返回：
    - overall accuracy
    - per-task accuracy（20 个任务类型）
    - avg generate_ms, visual_tokens 等
    """
```

打印格式：
```
==================================================
MVBENCH EVALUATION SUMMARY
==================================================
          Mode |   Accuracy |     N | Err | Gen(ms) | VisTok
-------------------------------------------------------------
      baseline |     65.0%  |  3600 |   0 |    1500 |   4800
  sparse(0.5)  |     63.5%  |  3600 |   0 |     800 |   2400

Per-Task Breakdown:
          Task |  baseline | sparse(0.5) | naive_iframe
-------------------------------------------------------
action_antonym |    72.0%  |      70.0%  |       71.5%
  action_count |    58.0%  |      56.5%  |       57.0%
       ...
```

### 7. 命令行接口

```python
def main():
    parser = argparse.ArgumentParser(description="MVBench Evaluation")
    parser.add_argument("--max-per-task", type=int, default=0,
                        help="Max samples per task (0 = all 200)")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Max frames (0=unlimited, MVBench videos are short)")
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Specific tasks to evaluate (default: all available)")
    parser.add_argument("--modes", nargs="+", default=["baseline", "sparse"],
                        choices=["baseline", "text_only", "audio_only", "video_only",
                                 "sparse", "sparse_no_audio",
                                 "naive_uniform", "naive_random", "naive_iframe"],
                        help="Modes to evaluate")
    parser.add_argument("--out-dir", 
                        default="/root/autodl-tmp/results/fasteromni/mvbench")
    args = parser.parse_args()
```

Usage:
```bash
# 快速验证（每任务 5 题）
python fasteromni/eval_mvbench.py --max-per-task 5 --modes baseline

# 全量评估（3 个核心 mode）
python fasteromni/eval_mvbench.py --modes baseline sparse naive_iframe

# 指定任务
python fasteromni/eval_mvbench.py --tasks action_antonym moving_count --modes baseline
```

## 必须复用的现有代码

1. **`fasteromni/pipeline.py`** 中的 `SparseInferencePipeline` — 所有推理逻辑
2. **`eval_videomme.py`** 中的：
   - `extract_answer_letter()` — 答案提取（需扩展支持 E）
   - `_Timeout` 类 — 超时保护
   - `_safe_extract_audio()` + monkey-patch 逻辑 — 音频死锁防护（**必须完整复制文件头部的 monkey-patch 代码**）
   - 增量 CSV 机制 — 中断恢复

## 输出目录结构

```
/root/autodl-tmp/results/fasteromni/mvbench/
├── baseline/
│   ├── baseline_details.csv
│   └── baseline_summary.json
├── sparse/
│   ├── sparse_details.csv
│   └── sparse_summary.json
├── naive_iframe/
│   ├── naive_iframe_details.csv
│   └── naive_iframe_summary.json
└── mvbench_combined_summary.json
```

## CSV 字段

```
sample_id, task_type, video_field, mode, keep_ratio, alpha,
gt_answer, pred_answer, correct, generate_ms, total_ms,
visual_tokens, audio_tokens, total_tokens, num_frames, error, pred_raw
```

## 注意事项

1. **Monkey-patch 必须在文件最顶部**，在任何 import qwen 之前。完整复制 `eval_videomme.py` 前 100 行的 monkey-patch 代码
2. **max_frames 默认 0**（无限制），因为 MVBench 视频都很短
3. **答案匹配用字母**：先把 answer_text 转成 answer_letter，模型输出也提取字母，字母比对
4. **增量 CSV 的 resume key 用 `sample_id`**（不是 question_id）
5. **进度输出必须 flush**，方便 tmux 实时查看
6. **视频路径解析失败时打印警告并跳过**，不要 crash
7. **文件编码 UTF-8**，print 用 `ensure_ascii=False`

## 验证命令

```bash
# Smoke test: 每任务 2 题, 只跑 baseline
python fasteromni/eval_mvbench.py --max-per-task 2 --modes baseline

# 期望输出：
# - 18 个 task × 2 题 = 36 题
# - 每题输出进度行
# - 最后打印 per-task 汇总表
# - 生成 CSV 和 JSON 到 out-dir
```
