"""
使用更稳定的方式下载模型 - 版本2
直接使用镜像站下载
"""
import os

# 必须在导入 transformers 之前设置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 重新导入以确保环境变量生效
import importlib
import sys

# 清除已导入的 huggingface_hub 模块
if 'huggingface_hub' in sys.modules:
    del sys.modules['huggingface_hub']

from transformers import BertModel, BertTokenizer

def download_with_retry(model_name='bert-base-chinese', max_retries=3):
    """带重试的模型下载"""
    for attempt in range(max_retries):
        try:
            print(f"尝试 {attempt + 1}/{max_retries}: 下载 {model_name}")

            # 方法1: 使用 trust_remote_code 和强制下载
            tokenizer = BertTokenizer.from_pretrained(
                model_name,
                force_download=False,
                resume_download=True,
            )
            print("✓ 分词器下载成功")

            model = BertModel.from_pretrained(
                model_name,
                force_download=False,
                resume_download=True,
            )
            print("✓ 模型下载成功")

            return True

        except Exception as e:
            print(f"✗ 尝试 {attempt + 1} 失败: {str(e)[:100]}")
            if attempt < max_retries - 1:
                import time
                wait_time = (attempt + 1) * 5
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print("\n所有尝试都失败了")
                return False

if __name__ == '__main__':
    print("=" * 60)
    print("开始下载 bert-base-chinese 模型")
    print("=" * 60)
    success = download_with_retry()

    if not success:
        print("\n" + "=" * 60)
        print("下载失败。备用方案：")
        print("1. 手动下载模型到本地")
        print("2. 使用 modelscope 下载")
        print("=" * 60)
