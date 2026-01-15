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
