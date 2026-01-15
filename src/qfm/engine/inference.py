import os

import torch
from diffusers import AutoencoderKL
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPTextModel, CLIPTokenizer

# === QFM 模块导入 ===
from qfm.config import cfg
from qfm.core.logger import logger
from qfm.model_moe import MiniFluxDiT
from qfm.utils import flux_ode_euler


def run_inference(ckpt_path: str, prompt: str, output_path: str = "result.png", steps: int = 50):
    """
    执行推理的主函数
    :param ckpt_path: 权重文件路径 (.pth)
    :param prompt: 提示词
    :param output_path: 图片保存路径
    :param steps: 采样步数 (默认50)
    """

    # 1. 环境设定
    device = cfg.get_device()
    dtype = cfg.get_dtype(device)  # 修正：必须传入 device

    logger.info(f"🎨 Inference task started on {device} ({dtype})")
    logger.info(f'   Prompt: "{prompt}"')
    logger.info(f"   Checkpoint: {ckpt_path}")

    # 检查权重是否存在
    if not os.path.exists(ckpt_path):
        logger.error(f"❌ Checkpoint not found at: {ckpt_path}")
        return

    # 2. 加载模型组件
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
        input_size=cfg.model.input_size,
        hidden_size=cfg.model.hidden_size,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        num_experts=cfg.model.num_experts,
        qwen_dim=cfg.model.qwen_dim,
        clip_dim=cfg.model.clip_dim,
    ).to(device, dtype=dtype)

    # 加载权重
    try:
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
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
        clip_vec = clip_model(**c_in).pooler_output

        # Qwen (注意: 这里长度设为128，需与预处理保持一致，或者在config里加一个 max_seq_len)
        q_in = qwen_tok([prompt], padding="max_length", max_length=128, truncation=True, return_tensors="pt").to(device)
        qwen_emb = qwen_model(**q_in, output_hidden_states=True).hidden_states[-1]

        # B. 生成初始噪声
        latents = torch.randn(1, 4, cfg.model.input_size, cfg.model.input_size, device=device, dtype=dtype)

        # C. ODE 采样 (Flow Matching 逆过程)
        latents = flux_ode_euler(dit, latents, clip_vec, qwen_emb, num_steps=steps)

        # D. VAE 解码
        # 反缩放 -> Decode
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
    # 测试用的默认参数
    TEST_CKPT = os.path.join(cfg.CKPT_DIR, "flux_moe_ep10.pth")  # 假设跑到了第10轮
    TEST_PROMPT = "A futuristic city with flying cars, cyberpunk style"

    # 如果找不到权重，就只打印一条警告，方便调试代码逻辑
    if not os.path.exists(TEST_CKPT):
        logger.warning(f"Test checkpoint not found at {TEST_CKPT}, please adjust path in __main__")
    else:
        run_inference(TEST_CKPT, TEST_PROMPT)
