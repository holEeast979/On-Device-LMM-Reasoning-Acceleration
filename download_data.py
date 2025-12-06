from datasets import load_dataset
import os
import random
import torch  # 如果需要设置 seed，用 torch 或 numpy

# 项目根目录（假设脚本在 my_project/ 下，调整如果不同）
project_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(project_dir, 'data')

# 设置随机种子以确保可复现（实验要求固定 seed）
seed = 42
random.seed(seed)
# 如果用 numpy 或 torch（可选，如果 datasets shuffle 用到）
# import numpy as np; np.random.seed(seed)
# torch.manual_seed(seed)

# 创建子集文件夹（如果不存在）
os.makedirs(os.path.join(data_dir, 'VQAv2_subset'), exist_ok=True)
os.makedirs(os.path.join(data_dir, 'MSVD-QA_subset'), exist_ok=True)
os.makedirs(os.path.join(data_dir, 'AudioCaps_subset'), exist_ok=True)

# 加载 VQAv2 validation 并取前 500 samples（顺序取，确保简单可复现；若需随机，用 shuffle）
vqa_dataset = load_dataset(
    "lmms-lab/VQAv2", 
    split="validation", 
    cache_dir=os.path.join(data_dir, 'VQAv2')
)
# 取前 500（实验起步 500）- 使用 select 方法返回 Dataset 对象
vqa_subset = vqa_dataset.select(range(500))
# 保存为 Arrow 格式（datasets 默认，便于后续加载）
vqa_subset.save_to_disk(os.path.join(data_dir, 'VQAv2_subset'))
# 可选：保存为 JSON（如果需要手动检查）- 如果失败则跳过
try:
    vqa_subset.to_json(os.path.join(data_dir, 'VQAv2_subset', 'subset.json'))
except Exception as e:
    print(f"警告：无法保存 VQAv2 为 JSON 格式（可能包含无法序列化的数据）：{e}")

# 加载 MSVD-QA test 并取随机 300 samples（方案 200-300，取 300；用 shuffle 随机取，确保多样）
msvd_dataset = load_dataset(
    "morpheushoc/msvd-qa", 
    split="test", 
    cache_dir=os.path.join(data_dir, 'MSVD-QA')
)
# Shuffle 并取前 300
msvd_dataset = msvd_dataset.shuffle(seed=seed)
msvd_subset = msvd_dataset.select(range(300))
msvd_subset.save_to_disk(os.path.join(data_dir, 'MSVD-QA_subset'))
try:
    msvd_subset.to_json(os.path.join(data_dir, 'MSVD-QA_subset', 'subset.json'))
except Exception as e:
    print(f"警告：无法保存 MSVD-QA 为 JSON 格式（可能包含无法序列化的数据）：{e}")

# 加载 AudioCaps test 并取前 300 samples（顺序取）
audiocaps_dataset = load_dataset(
    "d0rj/audiocaps", 
    split="test", 
    cache_dir=os.path.join(data_dir, 'AudioCaps')
)
audiocaps_subset = audiocaps_dataset.select(range(300))
audiocaps_subset.save_to_disk(os.path.join(data_dir, 'AudioCaps_subset'))
try:
    audiocaps_subset.to_json(os.path.join(data_dir, 'AudioCaps_subset', 'subset.json'))
except Exception as e:
    print(f"警告：无法保存 AudioCaps 为 JSON 格式（可能包含无法序列化的数据）：{e}")

print("子集已保存到 data/ 下对应文件夹。后续实验可加载如: from datasets import load_from_disk; subset = load_from_disk('data/VQAv2_subset')")