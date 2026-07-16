"""
修复 Hugging Face Hub 下载问题的工具函数
"""
import os
import sys

def setup_huggingface_env():
    """
    设置 Hugging Face 环境变量以避免下载问题
    在导入 transformers 之前调用此函数
    """
    # 方案1: 使用国内镜像（autodl 服务器推荐）
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    # 方案2: 禁用符号链接（避免某些文件系统问题）
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

    # 方案3: 设置更长的超时时间
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'

    # 方案4: 离线模式（如果模型已下载）
    # os.environ['TRANSFORMERS_OFFLINE'] = '1'
    # os.environ['HF_HUB_OFFLINE'] = '1'

    print("✓ Hugging Face 环境变量已设置")
    print(f"  - 使用镜像: {os.environ.get('HF_ENDPOINT', 'default')}")
    print(f"  - 超时时间: {os.environ.get('HF_HUB_DOWNLOAD_TIMEOUT', 'default')}")

def check_model_cache(model_name='bert-base-chinese'):
    """检查模型是否已经缓存"""
    cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
    if os.path.exists(cache_dir):
        models = os.listdir(cache_dir)
        model_cached = any(model_name.replace('/', '--') in m for m in models)
        if model_cached:
            print(f"✓ 模型 {model_name} 已在缓存中")
            return True
        else:
            print(f"✗ 模型 {model_name} 不在缓存中，需要下载")
            return False
    else:
        print("✗ Hugging Face 缓存目录不存在")
        return False

if __name__ == '__main__':
    setup_huggingface_env()
    check_model_cache()
