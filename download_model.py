"""
使用更稳定的方式下载 Hugging Face 模型
"""
import os
from transformers import BertModel, BertTokenizer

# 设置环境变量使用镜像（可选）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def download_model(model_name='bert-base-chinese'):
    """下载模型和分词器"""
    print(f"正在下载模型: {model_name}")
    
    try:
        # 下载分词器
        print("下载分词器...")
        tokenizer = BertTokenizer.from_pretrained(model_name)
        print("✓ 分词器下载成功")
        
        # 下载模型
        print("下载模型...")
        model = BertModel.from_pretrained(model_name)
        print("✓ 模型下载成功")
        
        print(f"\n模型已缓存到: ~/.cache/huggingface/")
        return True
        
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        return False

if __name__ == '__main__':
    download_model()
