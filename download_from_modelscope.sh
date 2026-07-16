#!/bin/bash
# 从魔塔社区下载 bert-base-chinese 模型

echo "======================================"
echo "从魔塔社区下载 BERT 模型"
echo "======================================"

python3 << PYTHON
from modelscope import snapshot_download
import os

# 模型 ID（魔塔社区的 bert-base-chinese）
model_id = 'tiansz/bert-base-chinese'

# 下载到指定目录
model_dir = '/root/models/bert-base-chinese'

print(f"正在下载模型: {model_id}")
print(f"保存路径: {model_dir}")

try:
    downloaded_path = snapshot_download(
        model_id, 
        cache_dir=model_dir,
        revision='master'
    )
    print(f"\n✓ 模型下载成功！")
    print(f"模型路径: {downloaded_path}")
except Exception as e:
    print(f"\n✗ 下载失败: {e}")
PYTHON

echo ""
echo "======================================"
echo "下载完成"
echo "======================================"
