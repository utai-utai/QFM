import torch


@torch.no_grad()
def flux_ode_euler(model, latents, clip_vec, qwen_emb, num_steps=50):
    """
    Euler Method ODE Solver for Rectified Flow
    dt = 1 / num_steps
    x_{t+1} = x_t + v(x_t, t) * dt
    """
    device = latents.device
    dtype = latents.dtype

    # 欧拉步长
    dt = 1.0 / num_steps

    # 从 t=0 (纯噪声) 到 t=1 (图像)
    for i in range(num_steps):
        # 当前时间 t
        t_value = i / num_steps
        t_tensor = torch.full((latents.shape[0],), t_value, device=device, dtype=dtype)

        # 预测速度场 v
        # 注意：我们的模型 forward 接收 (x, t, clip, qwen)
        v_predict = model(latents, t_tensor, clip_vec, qwen_emb)

        # 更新位置
        latents = latents + v_predict * dt

    return latents


#
# @torch.no_grad()
# def flux_ode_euler_cfg(
#         model,
#         latents,
#         clip_cond, qwen_cond,  # 真实的文本特征
#         clip_uncond, qwen_uncond,  # 空字符串的文本特征
#         num_steps=50,
#         cfg_scale=4.0  # CFG 放大系数，通常在 3.0 ~ 7.0 之间
# ):
#     """
#     Euler Method ODE Solver for Rectified Flow with Classifier-Free Guidance
#     """
#     device = latents.device
#     dtype = latents.dtype
#     dt = 1.0 / num_steps
#
#     # 从 t=0 (纯噪声) 到 t=1 (图像)
#     for i in range(num_steps):
#         t_value = i / num_steps
#
#         # 1. 批次翻倍 (Batch Doubling)
#         # 为了不跑两次 forward，我们把 cond 和 uncond 拼在一起，一次性算完
#         # latents 形状变成 [2B, C, H, W] (后续会被你的 x_embedder 压成 [2B, L, D])
#         latents_input = torch.cat([latents, latents], dim=0)
#         t_tensor = torch.full((latents_input.shape[0],), t_value, device=device, dtype=dtype)
#
#         clip_input = torch.cat([clip_cond, clip_uncond], dim=0)
#         qwen_input = torch.cat([qwen_cond, qwen_uncond], dim=0)
#
#         # 2. 预测速度场 v
#         # v_predict 里面包含了 2B 个结果：前 B 个是带文本的，后 B 个是不带文本的
#         v_predict = model(latents_input, t_tensor, clip_input, qwen_input)
#
#         # 3. 拆解结果 (Chunking)
#         v_cond, v_uncond = v_predict.chunk(2, dim=0)
#
#         # 4. 🚀 核心 CFG 魔法公式
#         # 方向 = 随便画的方向 + CFG倍数 * (听话的方向 - 随便画的方向)
#         v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
#
#         # 5. 更新位置 (欧拉步进)
#         latents = latents + v_cfg * dt
#
#     return latents
