import torch
import platform

import sys

print("=" * 30)
print(f"OS Detected:      {platform.system()}")
print(f"Python Version:   {sys.version.split()[0]}")
print(f"PyTorch Version:  {torch.__version__}")
print(f"CUDA Available:   {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Detected:     {torch.cuda.get_device_name(0)}")
    x = torch.ones(1).cuda()  # 测试一下显存分配
    print("✅ GPU Memory Access: Success")
else:
    print("❌ CUDA Not Available")
print("=" * 30)