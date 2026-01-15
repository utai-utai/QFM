import importlib.metadata
import importlib.util
import os
import platform
import sys
import time

if platform.system() == "Linux":  # 🔧 PyCharm/WSL 专用补丁 (防止 DeepSpeed 报错)
    os.environ["CUDA_HOME"] = "/home/yao/miniconda3/envs/qfm"  # 请确保这个路径是你 Conda 环境的真实路径


def print_status(component, status, version=None, extra=""):
    """格式化打印状态"""
    icon = {"OK": "✅", "MISS": "❌", "WARN": "⚠️", "OPT": "⚪"}
    ver_str = f"(v{version})" if version else ""
    print(f"[{icon.get(status, '?')}] {component:<20} {ver_str:<15} {extra}")


def get_version(package_name):
    """安全获取包版本"""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_import(package_name, import_name=None):
    """尝试 import 库并返回状态"""
    if import_name is None:
        import_name = package_name

    try:
        if importlib.util.find_spec(import_name) is not None:
            ver = get_version(package_name)
            return "OK", ver, ""
        else:
            return "MISS", None, "Not installed"
    except Exception as e:
        return "WARN", None, f"Import Error: {str(e)}"


def run_comprehensive_test():
    print("=" * 70)
    print("🔍 QFM 项目全栈依赖自检")
    print(f"🕒 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ----------------------------------------------------
    # 1. 核心系统与 Python
    # ----------------------------------------------------
    print("\n🛠️  [System & Python]")
    py_ver = sys.version.split()[0]
    py_status = "OK" if sys.version_info >= (3, 10) else "WARN"
    print_status("OS Platform", "OK", platform.system(), f"{platform.release()}")
    print_status("Python", py_status, py_ver, ">=3.10 Required")

    # ----------------------------------------------------
    # 2. 深度学习基础 (Torch Stack)
    # ----------------------------------------------------
    print("\n🧠 [Deep Learning Core]")
    import numpy
    import torch

    # Numpy
    print_status("Numpy", "OK", numpy.__version__)

    # Torch
    print_status("PyTorch", "OK", torch.__version__)

    # CUDA
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        cap_str = f"{capability[0]}.{capability[1]}"
        v_ram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print_status("CUDA", "OK", torch.version.cuda)
        print_status("GPU Device", "OK", gpu_name, f"({cap_str}) | V_RAM: {v_ram:.1f} GB")
    else:
        print_status("CUDA", "WARN", None, "Running on CPU (No GPU detected)")

    # ----------------------------------------------------
    # 3. 模型与数据生态 (Model Ecosystem)
    # ----------------------------------------------------
    print("\n📦 [Model & Data Libraries]")

    # 列表：(PyPI包名, Python import名)
    libs = [
        ("transformers", "transformers"),
        ("diffusers", "diffusers"),
        ("accelerate", "accelerate"),
        ("safetensors", "safetensors"),
        ("pillow", "PIL"),
        ("tqdm", "tqdm"),
        ("torchvision", "torchvision"),
    ]

    for pkg, imp in libs:
        status, ver, msg = check_import(pkg, imp)
        print_status(pkg, status, ver, msg)

    # 检查 CLI 工具 (pre-commit)
    pc_ver = get_version("pre-commit")
    if pc_ver:
        print_status("pre-commit", "OK", pc_ver, "(CLI Tool)")
    else:
        print_status("pre-commit", "MISS", None, "Dev tool missing")

    # ----------------------------------------------------
    # 4. 高性能加速 (GPU Optional)
    # ----------------------------------------------------
    print("\n🚀 [GPU Acceleration (Optional)]")

    # 检查构建工具
    for pkg in ["packaging", "ninja"]:
        status, ver, msg = check_import(pkg)
        print_status(pkg, status, ver, msg)

    # 检查 Flash Attention
    try:
        import flash_attn

        print_status("flash-attn", "OK", flash_attn.__version__)
    except ImportError:
        flash_attn = None
        status = "OPT" if not cuda_available else "MISS"
        print_status("flash-attn", status, flash_attn, "Optional (Required for 3090 Ti)")
    except Exception as e:
        print_status("flash-attn", "WARN", None, f"Error: {e}")

    # 检查 DeepSpeed (最难搞的一个)
    try:
        import deepspeed

        ds_ver = deepspeed.__version__
        ops_status = "Ops OK" if hasattr(deepspeed, "ops") else "Ops Warning"  # 尝试访问 ops 以确认编译状态
        print_status("deepspeed", "OK", ds_ver, ops_status)
    except ImportError:
        deepspeed = None
        status = "OPT" if not cuda_available else "MISS"
        print_status("deepspeed", status, deepspeed, "Optional")
    except Exception as e:
        print_status("deepspeed", "WARN", None, f"Error: {str(e)}")

    # ----------------------------------------------------
    # 5. 实战算力测试 (Smoke Test)
    # ----------------------------------------------------
    if cuda_available:
        print("-" * 70)
        print("🔥  Running Tensor Core Benchmark...")
        try:
            size = 4096
            a = torch.randn(size, size, device="cuda", dtype=torch.float16)
            b = torch.randn(size, size, device="cuda", dtype=torch.float16)

            # Warmup
            torch.mm(a, b)
            torch.cuda.synchronize()

            # Test
            start = time.time()
            torch.mm(a, b)
            torch.cuda.synchronize()
            end = time.time()

            t = end - start
            t_flops = (2 * size**3) / t / 1e12
            print(f"✅  Matrix Mul ({size}x{size}): {t * 1000:.2f} ms | Performance: {t_flops:.2f} T_FLOPS")
            print("🎉  All critical systems operational!")

        except Exception as e:
            print(f"❌  Compute Error: {e}")
    else:
        print("\n⚠️  Skipping benchmark (No GPU)")

    print("=" * 70)


if __name__ == "__main__":
    run_comprehensive_test()
