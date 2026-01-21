import os
import sys

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, "src"))

from qfm.model_moe import MiniFluxDiT  # noqa: E402


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def inspect_model():
    print("=" * 60)
    print("🕵️  QFM Model Inspector")
    print("=" * 60)

    # 1. 实例化模型
    print("⏳ Instantiating Model...")
    model = MiniFluxDiT(
        input_size=64,  # 默认 512px (64 latent)
        hidden_size=1024,
        depth=4,  # 打印个浅一点的方便看
        num_heads=16,
        num_experts=8,
    )

    print(f"✅ Model Created. Total Parameters: {count_parameters(model) / 1e6:.2f} M")
    print("-" * 60)

    # 2. 打印各层结构
    print("🏗️  Model Architecture Summary:")
    print("-" * 60)

    print(f"🔹 Input Embedder: {model.x_embedder}")
    print(f"🔹 Positional Embed: {model.pos_embed.shape} (Fixed Reference)")
    print(f"🔹 Condition Proj: {model.cond_proj}")

    print(f"\n🔹 Backbone ({len(model.blocks)} Layers):")
    # 只打印第一个 Block 的详情，避免刷屏
    block0 = model.blocks[0]
    print("   [Block 0] Type: MoEDiTBlock")
    print(f"     ├── Self-Attn: {block0.attn}")
    print(f"     ├── Cross-Attn: {block0.cross_attn}")
    print(f"     ├── MoE Layer: SparseMoELayer (Top2Router, {len(block0.moe.experts)} Experts)")
    print(f"     └── AdaLN: {block0.adaLN_modulation}")

    print(f"\n🔹 Final Layer: {model.final_layer}")
    print("-" * 60)

    # 3. 冒烟测试：测试动态分辨率 (Bucketing Test)
    print("🔥 Smoke Test: Dynamic Resolution (Bucketing)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 测试用例：模拟一个长方形的 Batch，假设输入图片是 512x768 -> Latent 是 64x96
    B = 2
    C = 4
    H_latent = 64
    W_latent = 96

    x = torch.randn(B, C, H_latent, W_latent).to(device)
    t = torch.rand(B).to(device)
    clip_vec = torch.randn(B, 768).to(device)
    qwen_ctx = torch.randn(B, 128, 1536).to(device)

    print(f"   Input Shape: {x.shape} (Aspect Ratio != 1:1)")

    try:
        with torch.no_grad():
            output = model(x, t, clip_vec, qwen_ctx)
        print(f"   Output Shape: {output.shape}")

        if output.shape == x.shape:
            print("✅ Test Passed: Output shape matches Input shape.")
            print("✨ Position Embedding Interpolation is working!")
        else:
            print("❌ Test Failed: Output shape mismatch.")
    except Exception as e:
        print(f"❌ Test Failed with error:\n{e}")

    print("=" * 60)


if __name__ == "__main__":
    inspect_model()
