"""
GPU检测脚本
"""
import torch

print("=" * 60)
print("🔍 GPU 检测")
print("=" * 60)

# 检查CUDA是否可用
cuda_available = torch.cuda.is_available()
print(f"\nCUDA可用: {cuda_available}")

if cuda_available:
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  名称: {torch.cuda.get_device_name(i)}")
        print(f"  显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")

        # 显存使用情况
        if hasattr(torch.cuda, 'memory_allocated'):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  已分配: {allocated:.2f} GB")
            print(f"  已保留: {reserved:.2f} GB")

    # 推荐的设备
    print(f"\n推荐使用设备: cuda:0")
    print(f"当前默认设备: {torch.cuda.current_device()}")
else:
    print("\n❌ 未检测到CUDA")
    print("可能的原因:")
    print("  1. 未安装CUDA")
    print("  2. PyTorch是CPU版本")
    print("  3. GPU驱动问题")

    print("\n检查PyTorch版本:")
    print(f"  PyTorch版本: {torch.__version__}")
    print(f"  是否为CUDA版本: {'+cu' in torch.__version__}")

print("\n" + "=" * 60)

# 测试简单的张量操作
print("\n🧪 测试张量操作")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n使用设备: {device}")

# 创建测试张量
x = torch.randn(1000, 1000).to(device)
y = torch.randn(1000, 1000).to(device)

import time
start = time.time()
z = torch.matmul(x, y)
end = time.time()

print(f"矩阵乘法耗时: {(end - start) * 1000:.2f} ms")
print(f"结果形状: {z.shape}")
print(f"结果设备: {z.device}")

print("\n✅ GPU检测完成!")
