import os
os.environ['CUDA_HOME'] = '/home/yao/miniconda3/envs/qfm'  # 强制指定 CUDA_HOME 为当前 Conda 环境路径
import sys
import platform
import torch
import time


def print_status(component, status, version=None, extra=""):
    """漂亮的打印格式"""
    ver_str = f"(v{version})" if version else ""
    print(f"[{status}] {component:<20} {ver_str} {extra}")


def run_verification():
    print("=" * 60)
    print(f"🔍 QFM 环境终极自检 (WSL2 + RTX 3090 Ti)")
    print("=" * 60)

    # 1. 基础环境
    print_status("OS Platform", "✅", platform.system(), f"- {platform.release()}")
    print_status("Python", "✅", sys.version.split()[0])

    # 2. PyTorch & CUDA
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        print_status("PyTorch", "✅", torch.__version__)
        print_status("CUDA", "✅", torch.version.cuda)
        print_status("GPU", "✅", gpu_name, f"(Compute Capability: {capability[0]}.{capability[1]})")

        # 显存测试
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            print(f"    └── VRAM: {free_mem / 1024 ** 3:.2f} GB Free / {total_mem / 1024 ** 3:.2f} GB Total")
        except:
            pass
    else:
        print_status("PyTorch", "❌", torch.__version__, "CUDA NOT AVAILABLE!")
        return

    # 3. 检查 Flash Attention
    try:
        import flash_attn
        print_status("Flash Attention", "✅", flash_attn.__version__)
    except ImportError as e:
        print_status("Flash Attention", "❌", extra=f"Import Error: {e}")
    except Exception as e:
        print_status("Flash Attention", "⚠️", extra=f"Error: {e}")

    # 4. 检查 DeepSpeed
    try:
        import deepspeed
        # DeepSpeed 需要 ninja，这里也能顺便测一下
        print_status("DeepSpeed", "✅", deepspeed.__version__, f"(Ops: {deepspeed.ops.__path__[0]})")
    except ImportError:
        print_status("DeepSpeed", "❌", extra="Not installed")
    except Exception as e:
        print_status("DeepSpeed", "⚠️", extra=f"Error: {e}")

    # 5. 真实算力测试 (Matrix Multiplication)
    print("-" * 60)
    print("🚀 正在运行 Tensor Core 算力测试...")
    try:
        # 创建两个较大的随机矩阵放到 GPU
        size = 4096
        a = torch.randn(size, size, device='cuda', dtype=torch.float16)
        b = torch.randn(size, size, device='cuda', dtype=torch.float16)

        # 预热
        torch.mm(a, b)

        # 计时
        start = time.time()
        c = torch.mm(a, b)
        torch.cuda.synchronize()  # 等待计算完成
        end = time.time()

        t = end - start
        tflops = (2 * size ** 3) / t / 1e12
        print(f"✅ 计算成功！耗时: {t * 1000:.2f} ms | 性能: {tflops:.2f} TFLOPS")
        print("🎉 环境配置完美，可以开始训练了！")

    except Exception as e:
        print(f"❌ 计算测试失败: {e}")

    print("=" * 60)


if __name__ == "__main__":
    run_verification()