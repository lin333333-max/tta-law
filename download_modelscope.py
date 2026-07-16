"""
使用 ModelScope 下载模型（国内最稳定）
ModelScope 是阿里云的模型仓库，在国内访问非常稳定
"""

def download_from_modelscope():
    """使用 ModelScope 下载模型"""
    try:
        # 先检查是否安装了 modelscope
        import modelscope
        print("✓ ModelScope 已安装")
    except ImportError:
        print("正在安装 ModelScope...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'modelscope', '-q'])
        print("✓ ModelScope 安装成功")

    from modelscope import snapshot_download
    import os

    model_id = 'tiansz/bert-base-chinese'
    cache_dir = os.path.expanduser('~/.cache/modelscope/hub')

    print(f"从 ModelScope 下载模型: {model_id}")
    print(f"缓存目录: {cache_dir}")

    try:
        model_dir = snapshot_download(model_id, cache_dir=cache_dir)
        print(f"✓ 模型下载成功！")
        print(f"模型路径: {model_dir}")

        # 创建软链接到 huggingface cache（可选）
        import shutil
        hf_cache = os.path.expanduser('~/.cache/huggingface/hub')
        os.makedirs(hf_cache, exist_ok=True)

        print("\n现在可以在代码中使用:")
        print(f"  BertTokenizer.from_pretrained('{model_dir}')")
        print(f"  BertModel.from_pretrained('{model_dir}')")

        return model_dir

    except Exception as e:
        print(f"✗ 下载失败: {e}")
        return None


if __name__ == '__main__':
    print("=" * 60)
    print("使用 ModelScope 下载 BERT 模型（国内最稳定）")
    print("=" * 60)
    model_path = download_from_modelscope()

    if model_path:
        print("\n" + "=" * 60)
        print("下载成功！")
        print("=" * 60)
