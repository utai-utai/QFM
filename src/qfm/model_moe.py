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
        # x: [B, L, D] -> logits: [B, L, E]
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        # 选出 Top-2
        top_weights, top_indices = torch.topk(probs, k=2, dim=-1)
        # 归一化权重
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-6)
        return top_weights, top_indices


class SparseMoELayer(nn.Module):
    def __init__(self, hidden_size, num_experts=8, expert_capacity=4):
        super().__init__()
        self.router = Top2Router(hidden_size, num_experts)
        intermediate_size = hidden_size * expert_capacity
        self.experts = nn.ModuleList([Expert(hidden_size, intermediate_size) for _ in range(num_experts)])

    def forward(self, x):
        batch, seq_len, dim = x.shape
        weights, indices = self.router(x)

        final_output = torch.zeros_like(x)

        flat_x = x.view(-1, dim)
        flat_output = final_output.view(-1, dim)
        flat_indices = indices.view(-1, 2)
        flat_weights = weights.view(-1, 2)

        # 简单循环实现 (MVP适用)
        for i, expert in enumerate(self.experts):
            mask = flat_indices == i
            batch_mask = mask.any(dim=-1)

            if batch_mask.any():
                selected_input = flat_x[batch_mask]
                expert_out = expert(selected_input)

                expanded_mask = mask[batch_mask]
                selected_weights = flat_weights[batch_mask]
                voting_weights = (selected_weights * expanded_mask.float()).sum(dim=1, keepdim=True)

                flat_output[batch_mask] += expert_out * voting_weights

        return final_output


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

        # 9 chunks: (shift, scale, gate) * 3 blocks
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True))
        # Zero Init
        nn.init.constant_(self.adaLN_modulation[1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[1].bias, 0)

    def forward(self, x, c, context_emb):
        # c: Global Condition (Time + Style)
        # context_emb: Spatial Condition (Qwen)

        params = self.adaLN_modulation(c).chunk(9, dim=1)
        (shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca, shift_moe, scale_moe, gate_moe) = params

        # 1. Self-Attention
        x_norm = modulate(self.norm1(x), shift_sa, scale_sa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_sa.unsqueeze(1) * attn_out

        # 2. Cross-Attention (Input: x, Key/Value: Qwen)
        x_norm = modulate(self.norm2(x), shift_ca, scale_ca)
        cross_out, _ = self.cross_attn(query=x_norm, key=context_emb, value=context_emb)
        x = x + gate_ca.unsqueeze(1) * cross_out

        # 3. MoE
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
        input_size=64,  # 64x64 latent size (512px image)
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

        # Image Embedding
        self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

        # Positional Embedding
        num_patches = (input_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))

        # Condition Embedding (Time=256 + CLIP=768 -> 1024)
        # 注意：Time Embedding 我们会在 forward 里动态生成
        self.t_embedder_mlp = nn.Sequential(nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 256))
        self.cond_proj = nn.Linear(256 + clip_dim, hidden_size)

        # Blocks
        self.blocks = nn.ModuleList([MoEDiTBlock(hidden_size, num_heads, qwen_dim, num_experts) for _ in range(depth)])

        # Final Layer
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

    def unpatchify(self, x):
        c = 4
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def get_timestep_embedding(self, t, dim=256):
        # t: [B] float
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

    def forward(self, x, t, clip_vec, qwen_context):
        # x: [B, 4, 64, 64]
        # t: [B]

        # 1. Embed Inputs
        x = self.x_embedder(x).flatten(2).transpose(1, 2)  # [B, L, D]
        x = x + self.pos_embed

        # 2. Embed Conditions
        t_emb = self.get_timestep_embedding(t)  # [B, 256]
        t_emb = self.t_embedder_mlp(t_emb)

        cond = torch.cat([t_emb, clip_vec], dim=-1)  # [B, 256+768]
        c = self.cond_proj(cond)  # [B, Hidden]

        # 3. Backbone
        for block in self.blocks:
            x = block(x, c, qwen_context)

        # 4. Output
        x = self.final_layer(x)
        x = self.unpatchify(x)
        return x
