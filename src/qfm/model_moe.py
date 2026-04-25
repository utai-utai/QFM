import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# Part A: MoE 组件 (Experts & Router)
# ==========================================


class Expert(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(intermediate_size, hidden_size),
        )

    def forward(self, x):
        return self.net(x)


class Top2Router(nn.Module):
    def __init__(self, hidden_size, num_experts):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts)

    def forward(self, x):
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        top_weights, top_indices = torch.topk(probs, k=2, dim=-1)
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-6)
        return top_weights, top_indices


class SparseMoELayer(nn.Module):
    def __init__(self, hidden_size, num_experts=8, expert_capacity=4):
        super().__init__()
        self.router = Top2Router(hidden_size, num_experts)
        intermediate_size = hidden_size * expert_capacity
        self.experts = nn.ModuleList([Expert(hidden_size, intermediate_size) for _ in range(num_experts)])

    def forward(self, x):
        batch, seq_len, dim = x.shape  # x: [B, L, D] batch=B, seq_len=L, dim=D
        weights, indices = self.router(x)  # weights: [B, L, 2], indices: [B, L, 2] (因为是Top-2)
        self.last_indices = indices.detach()  # [B, L, 2] (切断梯度，常用于后续计算负载均衡Loss)
        final_output = torch.zeros_like(x)  # [B, L, D] (最初的 3D 零矩阵置物架)

        # 拍平序列维度，方便做掩码操作
        flat_x = x.view(-1, dim)  # [B*L, D]
        flat_output = final_output.view(-1, dim)  # [B*L, D] (和 final_output 共享内存)
        flat_indices = indices.view(-1, 2)  # [B*L, 2]
        flat_weights = weights.view(-1, 2)  # [B*L, 2]

        for i, expert in enumerate(self.experts):  # 遍历所有专家
            mask = flat_indices == i  # [B*L, 2] (布尔值矩阵，找当前专家 i 的位置)
            batch_mask = mask.any(dim=-1)  # [B*L] (1D 布尔值向量，判断每个 Token 是否需要专家 i)

            if batch_mask.any():  # 如果有哪怕 1 个 Token 需要专家 i，就执行计算，假设有 K 个 Token 被分配给了当前专家 i
                selected_input = flat_x[batch_mask]  # [K, D] (把需要的 K 个 Token 抽出来)
                expert_out = expert(selected_input)  # [K, D] (经过 MLP 网络，维度不变)

                expanded_mask = mask[batch_mask]  # [K, 2] (抽出这 K 个 Token 的 True/False 原始状态)
                selected_weights = flat_weights[batch_mask]  # [K, 2] (抽出这 K 个 Token 对应的权重)

                # 把 True/False 变成 1.0/0.0，与权重相乘，并在特征维度求和
                voting_weights = (selected_weights * expanded_mask.float()).sum(dim=1, keepdim=True)  # [K, 1]

                # 专家输出 [K, D] 乘以 权重 [K, 1]（利用广播机制变成 [K, D]），然后累加回原内存位置
                flat_output[batch_mask] += expert_out * voting_weights  # 左边是 [K, D]，右边也是 [K, D]

        final_output = flat_output.reshape(batch, seq_len, dim)  # [B, L, D]
        return final_output  # [B, L, D]


# ==========================================
# Part B: DiT Block 组件
# ==========================================


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MoEDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, qwen_dim, num_experts=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, kdim=qwen_dim, vdim=qwen_dim, num_heads=num_heads, batch_first=True
        )

        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.moe = SparseMoELayer(hidden_size, num_experts)

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True))
        nn.init.constant_(self.adaLN_modulation[1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[1].bias, 0)

    def forward(self, x, c, context_emb):
        params = self.adaLN_modulation(c).chunk(9, dim=1)
        (shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca, shift_moe, scale_moe, gate_moe) = params

        x_norm = modulate(self.norm1(x), shift_sa, scale_sa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_sa.unsqueeze(1) * attn_out

        x_norm = modulate(self.norm2(x), shift_ca, scale_ca)
        cross_out, _ = self.cross_attn(query=x_norm, key=context_emb, value=context_emb)
        x = x + gate_ca.unsqueeze(1) * cross_out

        x_norm = modulate(self.norm3(x), shift_moe, scale_moe)
        moe_out = self.moe(x_norm)
        x = x + gate_moe.unsqueeze(1) * moe_out

        return x


# ==========================================
# Part C: 主模型架构
# ==========================================


class MiniFluxDiT(nn.Module):
    def __init__(
        self,
        input_size=64,
        patch_size=2,
        in_channels=4,
        hidden_size=1024,
        depth=8,
        num_heads=16,
        num_experts=8,
        qwen_dim=1536,
        clip_dim=768,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.in_channels = in_channels

        self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

        num_patches = (input_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))

        self.t_embedder_mlp = nn.Sequential(nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 256))
        self.cond_proj = nn.Linear(256 + clip_dim, hidden_size)

        self.blocks = nn.ModuleList([MoEDiTBlock(hidden_size, num_heads, qwen_dim, num_experts) for _ in range(depth)])

        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, patch_size * patch_size * in_channels, bias=True),
        )
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.x_embedder.weight, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.constant_(self.final_layer[-1].weight, 0)
        nn.init.constant_(self.final_layer[-1].bias, 0)

    def unpatchify(self, x, h, w):
        """
        支持非正方形图片
        h, w: 是 patch 的数量 (例如 32, 48)
        """
        c = self.in_channels
        p = self.patch_size
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))  # x: [B, L, P*P*C] -> [B, H, W, P, P, C]
        x = torch.einsum("nhwpqc->nchpwq", x)  # [B, H, W, P, P, C] -> [B, C, H, P, W, P]
        return x.reshape(shape=(x.shape[0], c, h * p, w * p))  # -> [B, C, H*P, W*P]

    def get_timestep_embedding(self, t, dim=256):
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

    def forward(self, x, t, clip_vec, qwen_context):
        B, C, H, W = x.shape  # x: [B, 4, H, W] (注意 H, W 可能不是 64x64)

        # 1. Embed Inputs
        x = self.x_embedder(x)  # [B, D, H/p, W/p]，x_embedder 是卷积，会自动处理不同分辨率
        grid_h, grid_w = x.shape[2], x.shape[3]  # 记录现在的 grid size，后面 unpatchify 要用
        x = x.flatten(2).transpose(1, 2)  # [B, L, D]

        # 2. Positional Embedding 插值
        if x.shape[1] != self.pos_embed.shape[1]:
            orig_size = int(
                math.sqrt(self.pos_embed.shape[1])
            )  # 算出原本预设的 2D 网格边长 (假设原本是 32x32 = 1024个Patch)
            pos_embed = self.pos_embed.permute(0, 2, 1).reshape(
                1, -1, orig_size, orig_size
            )  # 把 1D 的位置编码还原回 2D 的形状 [1, D, 32, 32]
            pos_embed = F.interpolate(
                pos_embed, size=(grid_h, grid_w), mode="bicubic", align_corners=False
            )  # 双三次插值 (Bicubic Interpolation)：插值到现在的尺寸 [1, D, grid_h, grid_w]
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # 压扁回去: [1, L_new, D]
            x = x + pos_embed
        else:
            x = x + self.pos_embed

        # 3. Embed Conditions
        t_emb = self.get_timestep_embedding(t)  # [B, 256] 将时间 t 变成正弦/余弦向量
        t_emb = t_emb.to(dtype=x.dtype)
        t_emb = self.t_embedder_mlp(t_emb)  # [B, 256] 提纯时间特征
        cond = torch.cat([t_emb, clip_vec], dim=-1)  # 拼上 CLIP 全局文本向量
        c = self.cond_proj(cond)  # [B, 1024] 融合为一个统一的全局条件 c

        # 4. Backbone
        for block in self.blocks:
            x = block(x, c, qwen_context)

        # 5. Output
        x = self.final_layer(x)
        x = self.unpatchify(x, grid_h, grid_w)

        return x
