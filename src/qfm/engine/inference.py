import os

import torch
from diffusers import AutoencoderKL
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPTextModel, CLIPTokenizer

from qfm.config import cfg
from qfm.core.logger import logger
from qfm.model_moe import MiniFluxDiT
from qfm.utils import flux_ode_euler


def run_inference(
    ckpt_path: str,
    prompt: str,
    output_path: str = "result.png",
    steps: int = 50,
    width: int = 512,  # 新增: 支持自定义宽
    height: int = 512,  # 新增: 支持自定义高
    seed: int = 42,  # 新增: 固定种子以便复现
):
    """
    执行推理的主函数 (支持任意长宽比)
    """
    # 0. 设定种子
    torch.manual_seed(seed)

    # 1. 环境设定
    device = cfg.get_device()
    dtype = cfg.get_dtype(device)

    logger.info(f"🎨 Inference task started on {device} ({dtype})")
    logger.info(f'   Prompt: "{prompt}"')
    logger.info(f"   Resolution: {width}x{height}")
    logger.info(f"   Checkpoint: {ckpt_path}")

    # 检查权重是否存在
    if not os.path.exists(ckpt_path):
        logger.error(f"❌ Checkpoint not found at: {ckpt_path}")
        return

    # 2. 加载辅助模型 (VAE, CLIP, Qwen)
    logger.info("Loading generic models (VAE, CLIP, Qwen)...")

    # VAE
    vae = AutoencoderKL.from_pretrained(cfg.VAE_ID).to(device, dtype=dtype)

    # CLIP
    clip_tok = CLIPTokenizer.from_pretrained(cfg.CLIP_ID)
    clip_model = CLIPTextModel.from_pretrained(cfg.CLIP_ID).to(device, dtype=dtype)

    # Qwen
    qwen_tok = AutoTokenizer.from_pretrained(cfg.LLM_ID)
    qwen_model = AutoModelForCausalLM.from_pretrained(cfg.LLM_ID).to(device, dtype=dtype)

    # 3. 加载 DiT 模型
    logger.info("Loading DiT model...")
    dit = MiniFluxDiT(
        input_size=64,
        hidden_size=cfg.model.hidden_size,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        num_experts=cfg.model.num_experts,
        qwen_dim=cfg.model.qwen_dim,
        clip_dim=cfg.model.clip_dim,
    ).to(device, dtype=dtype)

    # 加载权重
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        # 兼容处理：检查是用 torch.save(model.state_dict()) 存的，还是 save({'model_state_dict': ...}) 存的
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        dit.load_state_dict(state_dict)
        dit.eval()
        logger.info("✅ DiT weights loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load checkpoint: {e}")
        return

    # 4. 推理过程
    logger.info("Generating image...")

    with torch.no_grad():
        # A. 编码 Prompt
        # CLIP
        c_in = clip_tok([prompt], padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
        clip_vec = clip_model(**c_in).pooler_output  # [1, 768]

        # Qwen
        q_in = qwen_tok([prompt], padding="max_length", max_length=128, truncation=True, return_tensors="pt").to(device)
        qwen_emb = qwen_model(**q_in, output_hidden_states=True).hidden_states[-1]  # [1, 128, 1536]

        # B. 生成初始噪声 (根据目标分辨率计算 Latent 尺寸)
        # VAE 压缩率为 8，所以 latent 尺寸 = image 尺寸 / 8
        h_latent = height // 8
        w_latent = width // 8

        latents = torch.randn(1, 4, h_latent, w_latent, device=device, dtype=dtype)

        # C. ODE 采样 (Flow Matching 逆过程)
        # 注意：flux_ode_euler 内部会调用 model(x, t, ...)，model 会自动处理动态位置编码插值
        latents = flux_ode_euler(dit, latents, clip_vec, qwen_emb, num_steps=steps)

        # D. VAE 解码
        latents = latents / 0.13025
        image_tensor = vae.decode(latents).sample

        # E. 后处理与保存
        image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
        image = image_tensor.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")[0]

        pil_img = Image.fromarray(image)
        pil_img.save(output_path)
        logger.info(f"✨ Success! Image saved to: {os.path.abspath(output_path)}")


# 允许单独测试运行
if __name__ == "__main__":
    # 自动寻找最新的 checkpoint
    if os.path.exists(cfg.CKPT_DIR):
        ckpts = sorted([f for f in os.listdir(cfg.CKPT_DIR) if f.endswith(".pth")])
        if ckpts:
            latest_ckpt = os.path.join(cfg.CKPT_DIR, ckpts[-1])
            run_inference(
                latest_ckpt,
                prompt="A cinematic shot of a robot standing in the rain, cyberpunk city background, 8k resolution",
                width=512,  # 试试改成 768
                height=512,  # 试试改成 384
                steps=50,
            )
        else:
            logger.warning("No checkpoints found in checkpoints/ folder.")
    else:
        logger.warning("Checkpoints folder does not exist.")
