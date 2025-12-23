import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from atomworks.ml.encoding_definitions import AF3SequenceEncoding
from einops import rearrange
from rfd3.model.layers.attention import (
    GatedCrossAttention,
    LocalAttentionPairBias,
)
from rfd3.model.layers.block_utils import (
    build_valid_mask,
    create_attention_indices,
    group_atoms,
    ungroup_atoms,
)
from rfd3.model.layers.layer_utils import (
    AdaLN,
    EmbeddingLayer,
    LinearBiasInit,
    RMSNorm,
    Transition,
    collapse,
    linearNoBias,
)
from rfd3.model.layers.pairformer_layers import PairformerBlock
from torch.nn.functional import one_hot

from foundry import DISABLE_CHECKPOINTING
from foundry.common import exists

logger = logging.getLogger(__name__)


# SwiGLU transition block with adaptive layernorm
class ConditionedTransitionBlock(nn.Module):
    def __init__(self, c_token, c_s, n=2):
        super().__init__()
        self.ada_ln = AdaLN(c_a=c_token, c_s=c_s)
        self.linear_1 = linearNoBias(c_token, c_token * n)
        self.linear_2 = linearNoBias(c_token, c_token * n)
        self.linear_output_project = nn.Sequential(
            LinearBiasInit(c_s, c_token, biasinit=-2.0),
            nn.Sigmoid(),
        )
        self.linear_3 = linearNoBias(c_token * n, c_token)

    def forward(
        self,
        Ai,  # [B, I, C_token]
        Si,  # [B, I, C_token]
    ):
        Ai = self.ada_ln(Ai, Si)
        # BUG: This is not the correct implementation of SwiGLU
        # Bi = torch.sigmoid(self.linear_1(Ai)) * self.linear_2(Ai)
        # FIX: This is the correct implementation of SwiGLU
        Bi = torch.nn.functional.silu(self.linear_1(Ai)) * self.linear_2(Ai)

        # Output projection (from adaLN-Zero)
        return self.linear_output_project(Si) * self.linear_3(Bi)


class PositionPairDistEmbedder(nn.Module):
    """
    位置配对距离嵌入器 (Algorithm 14: Position Pair Distance Embedding)
    Position Pair Distance Embedder

    将原子对的参考位置编码为配对特征。使用逆配对距离编码空间关系。
    Encodes reference positions of atom pairs into pairwise features.
    Uses inverse pairwise distance to encode spatial relationships.

    算法步骤 / Algorithm Steps:
    1. 计算配对距离向量 d_lm = ref_pos_l - ref_pos_m
    2. 编码逆配对距离: 1/(1 + ||d_lm||^2)
    3-4. 嵌入mask信息并应用到特征上

    参数 / Parameters:
        c_atompair: 配对特征维度 / Pairwise feature dimension
        embed_frame: 是否嵌入完整的距离向量(3D) / Whether to embed full distance vector (3D)
    """
    def __init__(self, c_atompair, embed_frame=True):
        super().__init__()
        self.embed_frame = embed_frame
        if embed_frame:
            # 嵌入完整的3D距离向量 / Embed full 3D distance vector
            self.process_d = linearNoBias(3, c_atompair)

        # Algorithm 14 步骤2: 编码逆配对距离 / Step 2: Encode inverse pairwise distance
        self.process_inverse_dist = linearNoBias(1, c_atompair)
        # Algorithm 14 步骤3-4: 嵌入mask / Step 3-4: Embed mask
        self.process_valid_mask = linearNoBias(1, c_atompair)

    def forward_af3(self, D_LL, V_LL):
        """Forward the same way reference positions are embeded in AF3"""

        P_LL = self.process_d(D_LL) * V_LL

        # Embed pairwise inverse squared distances, and the valid mask
        if self.training:
            P_LL = (
                P_LL
                + self.process_inverse_dist(
                    1 / (1 + torch.linalg.norm(D_LL, dim=-1, keepdim=True) ** 2)
                )
                * V_LL
            )
            P_LL = P_LL + self.process_valid_mask(V_LL.to(P_LL.dtype)) * V_LL
        else:
            P_LL[V_LL[..., 0]] += self.process_inverse_dist(
                1
                / (1 + torch.linalg.norm(D_LL[V_LL[..., 0]], dim=-1, keepdim=True) ** 2)
            )
            P_LL[V_LL[..., 0]] += self.process_valid_mask(
                V_LL[V_LL[..., 0]].to(P_LL.dtype)
            )
        return P_LL

    def forward(self, ref_pos, valid_mask):
        """
        算法步骤 / Algorithm steps (Algorithm 14):
        1. 计算配对距离 / Compute pairwise distances
        2. 编码逆配对距离 / Encode inverse pairwise distance
        3-4. 嵌入mask / Embed mask
        """
        # 步骤1: 计算配对距离向量 / Step 1: Compute pairwise distance vectors
        D_LL = ref_pos.unsqueeze(-2) - ref_pos.unsqueeze(-3)  # [L, L, 3] or [B, L, L, 3]
        V_LL = valid_mask  # [L, L, 1] or [B, L, L, 1]

        if self.embed_frame:
            # 嵌入完整的3D距离框架 / Embed full 3D distance frame
            return self.forward_af3(D_LL, V_LL)

        # 步骤2: 编码逆配对距离 / Step 2: Encode inverse pairwise distance
        # 计算距离的平方: ||d_lm||^2
        norm = torch.linalg.norm(D_LL, dim=-1, keepdim=True) ** 2
        norm = torch.clamp(norm, min=1e-6)  # 避免除以零 / Avoid division by zero
        # 逆距离编码: 1/(1 + ||d_lm||^2)
        inv_dist = 1 / (1 + norm)
        P_LL = self.process_inverse_dist(inv_dist) * V_LL

        # 步骤3-4: 添加mask嵌入 / Step 3-4: Add mask embedding
        P_LL = P_LL + self.process_valid_mask(V_LL.to(P_LL.dtype)) * V_LL
        return P_LL


class OneDFeatureEmbedder(nn.Module):
    """
    一维特征嵌入器 (Algorithm 15: One-dimension Feature Embedder)
    One-dimension Feature Embedder

    将多个1D特征(残基类型、原子类型等)嵌入并求和为单个向量。
    Embeds and sums multiple 1D features (residue type, atom type, etc.) into a single vector.

    算法步骤 / Algorithm Steps:
    1. 对每个1D特征f_i进行独立嵌入: e_i = Embed(f_i)
    2. 求和所有嵌入: e = Σ e_i

    这种加法聚合允许模型组合多个特征源。
    This additive aggregation allows the model to compose multiple feature sources.

    参数 / Args:
        features (dict): 特征名称及其通道数的字典 / Dictionary of feature names and their number of channels
        output_channels (int): 输出嵌入维度 / Output dimension of the projected embedding
    """

    def __init__(self, features, output_channels):
        super().__init__()
        # 过滤存在的特征 / Filter existing features
        self.features = {k: v for k, v in features.items() if exists(v)}
        total_embedding_input_features = sum(self.features.values())

        # 为每个特征创建独立的嵌入层 / Create independent embedding layer for each feature
        self.embedders = nn.ModuleDict(
            {
                feature: EmbeddingLayer(
                    n_channels, total_embedding_input_features, output_channels
                )
                for feature, n_channels in self.features.items()
            }
        )

    def forward(self, f, collapse_length):
        """
        前向传播 / Forward pass

        参数 / Args:
            f: 特征字典 / Feature dictionary
            collapse_length: 折叠长度(I或L) / Collapse length (I or L)

        返回 / Returns:
            嵌入特征的总和 / Sum of embedded features: [collapse_length, output_channels]
        """
        # Algorithm 15: 对每个1D特征嵌入并求和 / Embed each 1D feature and sum
        return sum(
            tuple(
                self.embedders[feature](collapse(f[feature].float(), collapse_length))
                for feature, n_channels in self.features.items()
                if exists(n_channels)
            )
        )


class SinusoidalDistEmbed(nn.Module):
    """
    正弦距离嵌入 (Algorithm 13: Sinusoidal Distance Embedding)
    Sinusoidal Distance Embedding

    对配对距离应用正弦嵌入,类似于Transformer中的位置编码。
    Applies sinusoidal embedding to pairwise distances, similar to positional encoding in Transformers.

    算法步骤 / Algorithm Steps:
    1. 计算配对距离 ||p_l - p_m||
    2-4. 正弦嵌入:
         - 频率: ω_k = 1 / (10000^(2k/D))
         - 角度: θ_lmk = d_lm * ω_k
         - 嵌入: e_lm = [sin(θ) || cos(θ)]
    5-7. 应用并嵌入mask

    参数 / Args:
        c_atompair (int): 输出投影嵌入维度(必须为偶数) / Output dimension (must be even)
        n_freqs (int): sin/cos对数,总正弦维度 = 2 * n_freqs / Number of sin/cos pairs
    """

    def __init__(self, c_atompair, n_freqs=32):
        super().__init__()
        assert c_atompair % 2 == 0, "Output embedding dim must be even"

        self.n_freqs = n_freqs  # Number of sin/cos pairs → total sinusoidal dim = 2 * n_freqs
        self.c_atompair = c_atompair

        # 投影正弦嵌入到输出维度 / Project sinusoidal embedding to output dimension
        self.output_proj = linearNoBias(2 * n_freqs, c_atompair)
        # Algorithm 13 步骤5-7: 嵌入mask / Step 5-7: Embed mask
        self.process_valid_mask = linearNoBias(1, c_atompair)

    def forward(self, pos, valid_mask):
        """
        前向传播 / Forward pass

        参数 / Args:
            pos: [L, 3] 或 [B, L, 3] 原子位置 / Atom positions
            valid_mask: [L, L, 1] 或 [B, L, L, 1] 有效性mask / Validity mask

        返回 / Returns:
            P_LL: [L, L, c_atompair] 或 [B, L, L, c_atompair] 嵌入的配对特征
        """
        # ===== 步骤1: 计算配对距离 / Step 1: Compute pairwise distances =====
        D_LL = pos.unsqueeze(-2) - pos.unsqueeze(-3)  # [L, L, 3] or [B, L, L, 3]
        dist_matrix = torch.linalg.norm(D_LL, dim=-1)  # [L, L] or [B, L, L]

        # ===== 步骤2-4: 正弦嵌入 / Step 2-4: Sinusoidal embedding =====
        # 步骤2: 计算频率 ω_k = 1 / (10000^(2k/D))
        half_dim = self.n_freqs
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half_dim, dtype=torch.float32)
            / half_dim
        ).to(dist_matrix.device)  # [n_freqs]

        # 步骤3: 计算角度 θ_lmk = d_lm * ω_k
        angles = dist_matrix.unsqueeze(-1) * freq  # [..., n_freqs]

        # 步骤4: 应用sin和cos生成嵌入 / Apply sin and cos to generate embedding
        sin_embed = torch.sin(angles)  # [..., n_freqs]
        cos_embed = torch.cos(angles)  # [..., n_freqs]
        sincos_embed = torch.cat([sin_embed, cos_embed], dim=-1)  # [..., 2*n_freqs]

        # 线性投影到输出维度 / Linear projection to output dimension
        P_LL = self.output_proj(sincos_embed)  # [..., c_atompair]
        P_LL = P_LL * valid_mask

        # ===== 步骤5-7: 添加mask嵌入 / Step 5-7: Add mask embedding =====
        P_LL = P_LL + self.process_valid_mask(valid_mask.to(P_LL.dtype)) * valid_mask
        return P_LL


class LinearEmbedWithPool(nn.Module):
    def __init__(self, c_token):
        super().__init__()
        self.c_token = c_token
        self.linear = linearNoBias(3, c_token)

    def forward(self, R_L, tok_idx):
        B = R_L.shape[0]
        I = int(tok_idx.max().item()) + 1
        A_I_shape = (
            B,
            I,
            self.c_token,
        )
        Q_L = self.linear(R_L)
        A_I = (
            torch.zeros(A_I_shape, device=R_L.device, dtype=Q_L.dtype)
            .index_reduce(
                -2,
                tok_idx.long(),
                Q_L,
                "mean",
                include_self=False,
            )
            .clone()
        )
        return A_I


class SimpleRecycler(nn.Module):
    def __init__(
        self,
        c_s,
        c_z,
        template_embedder,
        msa_module,
        n_pairformer_blocks,
        pairformer_block,
    ):
        super().__init__()
        self.c_z = c_z
        self.process_zh = nn.Sequential(
            RMSNorm(c_z),
            linearNoBias(c_z, c_z),
        )
        self.process_sh = nn.Sequential(
            RMSNorm(c_s),
            linearNoBias(c_s, c_s),
        )
        self.pairformer_stack = nn.ModuleList(
            [
                PairformerBlock(c_s=c_s, c_z=c_z, **pairformer_block)
                for _ in range(n_pairformer_blocks)
            ]
        )
        # Templates and msa's removed:
        # self.template_embedder = TemplateEmbedder(c_z=c_z, **template_embedder)
        # self.msa_module = MSAModule(**msa_module)

    def forward(
        self,
        f,
        S_inputs_I,
        S_init_I,
        Z_init_II,
        S_I,
        Z_II,
    ):
        Z_II = Z_init_II + self.process_zh(Z_II)

        # Templates and msa's removed:
        # Z_II = Z_II + self.template_embedder(f, Z_II)
        # Z_II = self.msa_module(f, Z_II, S_inputs_I)

        S_I = S_init_I + self.process_sh(S_I)
        for block in self.pairformer_stack:
            S_I, Z_II = block(S_I, Z_II)
        return S_I, Z_II


class RelativePositionEncodingWithIndexRemoval(nn.Module):
    """
    Usual RPE but utilizes `is_motif_atom_3d_unindexed` to ensure within-chain position is spoofed.
    """

    def __init__(self, r_max, s_max, c_z):
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.c_z = c_z

        self.num_tok_pos_bins = (
            2 * self.r_max + 2
        ) + 1  # original af3 + 1 for unknown index
        self.linear = linearNoBias(
            2 * self.num_tok_pos_bins + (2 * self.s_max + 2) + 1, c_z
        )

    def forward(self, f):
        b_samechain_II = f["asym_id"].unsqueeze(-1) == f["asym_id"].unsqueeze(-2)
        b_same_entity_II = f["entity_id"].unsqueeze(-1) == f["entity_id"].unsqueeze(-2)
        d_residue_II = torch.where(
            b_samechain_II,
            torch.clip(
                f["residue_index"].unsqueeze(-1)
                - f["residue_index"].unsqueeze(-2)
                + self.r_max,
                0,
                2 * self.r_max,
            ),
            2 * self.r_max + 1,
        )
        b_sameresidue_II = f["residue_index"].unsqueeze(-1) == f[
            "residue_index"
        ].unsqueeze(-2)
        tok_distance = (
            f["token_index"].unsqueeze(-1) - f["token_index"].unsqueeze(-2) + self.r_max
        )
        d_token_II = torch.where(
            b_samechain_II * b_sameresidue_II,
            torch.clip(
                tok_distance,
                0,
                2 * self.r_max,
            ),
            2 * self.r_max + 1,
        )

        # Chain distances are kept
        d_chain_II = torch.where(
            # NOTE: Implementing bugfix from the Protenix Technical report, where we use `same_entity` instead of `not same_chain` (as in the AF-3 pseudocode)
            # Reference: https://github.com/bytedance/Protenix/blob/main/Protenix_Technical_Report.pdf
            b_same_entity_II,
            torch.clip(
                f["sym_id"].unsqueeze(-1) - f["sym_id"].unsqueeze(-2) + self.s_max,
                0,
                2 * self.s_max,
            ),
            2 * self.s_max + 1,
        )
        A_relchain_II = one_hot(d_chain_II.long(), 2 * self.s_max + 2)

        #########################################################
        # Cancel out distances from unidexed motifs
        unindexing_pair_mask = f[
            "unindexing_pair_mask"
        ]  # [L, L] representing the parts which shouldnt' talk to one another

        # Special position case
        d_token_II[unindexing_pair_mask] = self.num_tok_pos_bins - 1
        d_residue_II[unindexing_pair_mask] = self.num_tok_pos_bins - 1

        A_relpos_II = one_hot(d_residue_II.long(), self.num_tok_pos_bins)
        A_reltoken_II = one_hot(d_token_II, self.num_tok_pos_bins)
        #########################################################

        return self.linear(
            torch.cat(
                [
                    A_relpos_II,
                    A_reltoken_II,
                    b_same_entity_II.unsqueeze(-1),
                    A_relchain_II,
                ],
                dim=-1,
            ).to(torch.float)
        )


class VirtualPredictor(nn.Module):
    def __init__(self, c_atom):
        super(VirtualPredictor, self).__init__()
        self.process_atom_embeddings = nn.Sequential(
            RMSNorm((c_atom,)), linearNoBias(c_atom, 1)
        )

    def forward(self, Q_L):
        return self.process_atom_embeddings(Q_L)


class SequenceHead(nn.Module):
    def __init__(self, c_token):
        super(SequenceHead, self).__init__()

        # Distogram feature extraction
        self.dist_fc1 = nn.Linear(196, 128)
        self.dist_relu = nn.ReLU()
        self.dist_fc2 = nn.Linear(128, 64)

        # Embedding feature extraction
        self.embed_fc1 = nn.Linear(c_token, 128)
        self.embed_relu = nn.ReLU()
        self.embed_fc2 = nn.Linear(128, 64)

        # Fusion layer
        self.fusion_fc = nn.Linear(128, 32)

        # Sequence encoding
        self.sequence_encoding_ = AF3SequenceEncoding()

    def forward(self, A_I, Q_L, X_L, f):
        B, L, _ = X_L.shape
        max_res_id = f["atom_to_token_map"].max().item() + 1

        # Detach tensors to avoid gradients through main module
        # X_L = X_L.detach()
        # A_I = A_I.detach()
        # Q_L = Q_L.detach()

        # Compute distograms
        residue_distogram = torch.zeros(B, max_res_id, 14, 14, device=X_L.device)
        for i in range(max_res_id):
            residue_mask = f["atom_to_token_map"] == i
            if residue_mask.sum() == 14:
                coords = X_L[:, residue_mask]  # (B, 14, 3)
                residue_distogram[:, i] = torch.cdist(coords, coords)

        # Flatten distogram
        dist_features = residue_distogram.view(B, max_res_id, 196)

        # Pass through separate MLPs
        dist_out = self.dist_fc1(dist_features)
        dist_out = self.dist_relu(dist_out)
        dist_out = self.dist_fc2(dist_out)

        embed_out = self.embed_fc1(A_I)
        embed_out = self.embed_relu(embed_out)
        embed_out = self.embed_fc2(embed_out)

        # Fusion via concatenation
        fused = torch.cat([dist_out, embed_out], dim=-1)
        Seq_I = self.fusion_fc(fused)

        indices = self.decode(Seq_I)

        return Seq_I, indices

    def decode(self, Seq_I):
        indices = Seq_I.argmax(dim=-1)  # [B, L]
        return indices


class LinearSequenceHead(nn.Module):
    def __init__(self, c_token):
        super().__init__()
        n_tok_all = 32
        disallowed_idxs = AF3SequenceEncoding().encode(["UNK", "X", "DX", "<G>"])
        mask = torch.ones(n_tok_all, dtype=torch.bool)
        mask[disallowed_idxs] = False
        self.register_buffer("valid_out_mask", mask)
        self.linear = nn.Linear(c_token, n_tok_all)

    def forward(self, A_I, **_):
        logits = self.linear(A_I)
        indices = self.decode(logits)
        return logits, indices

    def decode(self, logits):
        # logits: [D, L, 28]
        # indices: [D, L] in [0,32-1]
        D, I, _ = logits.shape
        probs = F.softmax(logits, dim=-1)
        probs = probs * self.valid_out_mask[None, None, :].to(probs.device)
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
        indices = probs.argmax(axis=-1)
        return indices


class Upcast(nn.Module):
    def __init__(
        self, c_token, c_atom, method="broadcast", cross_attention_block=None, n_split=6
    ):
        super().__init__()
        self.method = method
        self.n_split = n_split
        if self.method == "broadcast":
            self.project = nn.Sequential(
                RMSNorm((c_token,)), linearNoBias(c_token, c_atom)
            )
        elif self.method == "cross_attention":
            self.gca = GatedCrossAttention(
                c_query=c_atom, c_kv=c_token // self.n_split, **cross_attention_block
            )
        else:
            raise ValueError(f"Unknown upcast method: {self.method}")

    def forward_(self, Q_IA, A_I, valid_mask=None):
        if self.method == "broadcast":
            Q_IA = Q_IA + self.project(A_I)[..., None, :]
        elif self.method == "cross_attention":
            assert exists(A_I) and exists(valid_mask)
            # Split Tokens
            A_I = rearrange(A_I, "b n (s c) -> b n s c", s=self.n_split)
            n_tokens, n_atom_per_tok = Q_IA.shape[1], Q_IA.shape[2]

            # Attention mask: ..., n_atom_per_tok, n_split
            attn_mask = torch.full(
                (n_tokens, 1, n_atom_per_tok), True, device=Q_IA.device
            )
            attn_mask[~valid_mask.view_as(attn_mask)] = False

            attn_mask = torch.ones(
                (n_tokens, n_atom_per_tok, self.n_split), device=A_I.device, dtype=bool
            )
            attn_mask[~valid_mask, :] = False

            Q_IA = Q_IA + self.gca(q=Q_IA, kv=A_I, attn_mask=attn_mask)
        return Q_IA

    def forward(self, Q_L, A_I, tok_idx):
        valid_mask = build_valid_mask(tok_idx)
        Q_IA = ungroup_atoms(Q_L, valid_mask)
        Q_IA = self.forward_(Q_IA, A_I, valid_mask)
        Q_L = group_atoms(Q_IA, valid_mask)
        return Q_L


class Downcast(nn.Module):
    """
    下投影 (Algorithm 9: Downcast)
    Downcast

    将atom级特征池化到token级特征。使用交叉注意力或平均池化。
    Pools atom-level features to token-level features using cross-attention or mean pooling.

    算法步骤 / Algorithm Steps (Algorithm 9):
    1. 按token ID分组原子: group_atoms(q_ia)
    2. 交叉注意力池化: GatedCrossAttention (Q=a_i, KV=q_ia)
       或平均池化: mean(q_ia) per token
    3. (可选) 添加单轨迹特征 s_i
    4. 返回更新的token特征 a_i

    参数 / Parameters:
        c_atom: Atom特征维度 / Atom feature dimension
        c_token: Token特征维度 / Token feature dimension
        c_s: (可选) 单轨迹特征维度 / Optional single track feature dimension
        method: "mean" (平均池化) 或 "cross_attention" / Pooling method
    """

    def __init__(
        self, c_atom, c_token, c_s=None, method="mean", cross_attention_block=None
    ):
        super().__init__()
        self.method = method
        self.c_token = c_token
        self.c_atom = c_atom

        # 可选: 处理单轨迹特征 / Optional: process single track features
        if c_s is not None:
            self.process_s = nn.Sequential(
                RMSNorm((c_s,)),
                linearNoBias(c_s, c_token),
            )
        else:
            self.process_s = None

        # 池化方法 / Pooling method
        if self.method == "mean":
            # 平均池化: 投影并求平均 / Mean pooling: project and average
            self.project = linearNoBias(c_atom, c_token)
        elif self.method == "cross_attention":
            # Algorithm 11: GatedCrossAttention - Q=token, KV=atoms
            self.gca = GatedCrossAttention(
                c_query=c_token,
                c_kv=c_atom,
                **cross_attention_block,
            )
        else:
            raise ValueError(f"Unknown downcast method: {self.method}")

    def forward_(self, Q_IA, A_I, S_I=None, valid_mask=None):
        """
        核心Downcast操作 / Core downcast operation

        参数 / Args:
            Q_IA: [B, I, max_atoms, c_atom] 分组的atom特征 / Grouped atom features
            A_I: [B, I, c_token] 当前token特征 / Current token features
            S_I: [B, I, c_s] (可选) 单轨迹特征 / Optional single track features
            valid_mask: [I, max_atoms] 有效atom mask / Valid atom mask

        返回 / Returns:
            A_I: [B, I, c_token] 更新后的token特征 / Updated token features
        """
        # ===== Algorithm 9 步骤2: 池化操作 / Step 2: Pooling operation =====
        if self.method == "mean":
            # 平均池化: project并除以有效atom数 / Mean pooling: project and divide by valid atom count
            A_I_update = self.project(Q_IA).sum(-2) / valid_mask.sum(-1, keepdim=True)
        elif self.method == "cross_attention":
            # Algorithm 11: GatedCrossAttention
            assert exists(A_I) and exists(valid_mask)
            # Attention mask: ..., 1, n_atom_per_tok (1个查询token对应token内的atoms)
            attn_mask = valid_mask[..., None, :]
            A_I_update = self.gca(
                q=A_I[..., None, :],  # Q: 单个token特征
                kv=Q_IA,  # KV: token内的所有atom特征
                attn_mask=attn_mask
            ).squeeze(-2)

        # 残差连接 / Residual connection
        A_I = A_I + A_I_update if exists(A_I) else A_I_update

        # ===== Algorithm 9 步骤4: (可选) 添加单轨迹特征 / Step 4: (Optional) Add single track features =====
        if self.process_s is not None:
            A_I = A_I + self.process_s(S_I)
        return A_I

    def forward(self, Q_L, A_I, S_I=None, tok_idx=None):
        """
        前向传播:将atom特征池化到token特征 / Forward: pool atom features to token features

        参数 / Args:
            Q_L: [B, L, c_atom] 或 [L, c_atom] Atom特征 / Atom features
            A_I: [B, I, c_token] 或 [I, c_token] 当前token特征 / Current token features
            S_I: [B, I, c_s] 或 [I, c_s] (可选) 单轨迹特征 / Optional single track features
            tok_idx: [L] atom到token的映射 / Atom to token mapping

        返回 / Returns:
            A_I: 更新后的token特征 / Updated token features
        """
        # ===== Algorithm 9 步骤1: 按token ID分组原子 / Step 1: Group atoms by token ID =====
        valid_mask = build_valid_mask(tok_idx)

        # 处理批次维度 / Handle batch dimension
        if Q_L.ndim == 2:
            squeeze = True
            Q_L = Q_L.unsqueeze(0)
        else:
            squeeze = False

        A_I = A_I.unsqueeze(0) if exists(A_I) and A_I.ndim == 2 else A_I
        S_I = S_I.unsqueeze(0) if exists(S_I) and S_I.ndim == 2 else S_I

        # 将atom特征重新组织为 [B, I, max_atoms, c_atom]
        Q_IA = ungroup_atoms(Q_L, valid_mask)

        # 执行池化操作 / Perform pooling operation
        A_I = self.forward_(Q_IA, A_I, S_I, valid_mask=valid_mask)

        if squeeze:
            A_I = A_I.squeeze(0)
        return A_I


######################################################################################
##########################     Local Atom Transformer       ##########################
######################################################################################


class LocalTokenTransformer(nn.Module):
    """
    局部Token Transformer (Algorithm 6: Local token transformer)
    Local Token Transformer

    在token级别应用SL2稀疏注意力(序列局部 + 结构局部)。
    Applies SL2 sparse attention (sequence-local + structure-local) at token level.

    算法步骤 / Algorithm Steps (Algorithm 6):
    1. 创建SL2稀疏注意力索引 (序列局部 + 结构局部)
    2-8. 循环通过多个transformer块:
         4. (可选) Upcast - 如果提供了c_skip
         6. SparseAttentionPairBias - 带配对偏置的稀疏注意力 (Algorithm 8)
         7. ConditionedTransitionBlock - 条件化的transition

    关键特性 / Key Features:
    - SL2稀疏注意力:仅关注序列邻居和结构邻居
    - 内存高效:避免完整的I×I注意力矩阵
    - 配对偏置:Z_II作为注意力偏置引导注意力

    参数 / Parameters:
        c_token: Token特征维度 / Token feature dimension
        c_tokenpair: Token配对特征维度 / Token pairwise feature dimension
        c_s: 单轨迹特征维度 / Single track feature dimension
        n_block: Transformer块数量 / Number of transformer blocks
        n_local_tokens: 序列局部邻居数 (默认8) / Number of sequence-local neighbors
        n_keys: 结构局部键数 (默认32) / Number of structure-local keys
    """
    def __init__(
        self,
        c_token,
        c_tokenpair,
        c_s,
        n_block,
        diffusion_transformer_block,
        n_registers=None,
        n_local_tokens=8,
        n_keys=32,
    ):
        super().__init__()
        self.n_local_tokens = n_local_tokens  # 序列局部注意力邻居数
        self.n_keys = n_keys  # 结构局部注意力键数
        # 创建transformer块栈 / Create transformer block stack
        self.blocks = nn.ModuleList(
            [
                StructureLocalAtomTransformerBlock(
                    c_atom=c_token,
                    c_s=c_s,
                    c_atompair=c_tokenpair,
                    **diffusion_transformer_block,
                )
                for _ in range(n_block)
            ]
        )

    def forward(self, A_I, S_I, Z_II, f, X_L, full=False):
        """
        前向传播 / Forward pass

        参数 / Args:
            A_I: [B, I, c_token] Token特征 / Token features
            S_I: [B, I, c_s] 单轨迹特征 / Single track features
            Z_II: [B, I, I, c_tokenpair] Token配对特征(用作注意力偏置) / Token pair features (as attention bias)
            f: 特征字典 / Feature dictionary
            X_L: [B, I, 3] Token坐标(C-alpha位置) / Token coordinates (C-alpha positions)
            full: 是否使用完整注意力(非稀疏) / Whether to use full attention (non-sparse)

        返回 / Returns:
            A_I: [B, I, c_token] 更新后的token特征 / Updated token features
        """
        # ===== Algorithm 6 步骤1: 创建SL2稀疏注意力索引 / Step 1: Create SL2 sparse attention indices =====
        # 结合序列局部和结构局部注意力
        indices = create_attention_indices(
            X_L=X_L,  # 用于计算结构局部邻居 / For computing structure-local neighbors
            f=f,
            tok_idx=torch.arange(A_I.shape[1], device=A_I.device),
            n_attn_keys=self.n_keys,  # 结构局部键数 / Structure-local keys
            n_attn_seq_neighbours=self.n_local_tokens,  # 序列局部邻居数 / Sequence-local neighbors
        )

        # ===== Algorithm 6 步骤2-8: 循环通过transformer块 / Step 2-8: Loop through transformer blocks =====
        for i, block in enumerate(self.blocks):
            # 设置checkpointing以节省内存 / Set checkpointing to save memory
            block.attention_pair_bias.use_checkpointing = not DISABLE_CHECKPOINTING

            # 步骤6: SparseAttentionPairBias (Algorithm 8) + 步骤7: ConditionedTransitionBlock
            # A_I: [B, I, c_token] Token特征
            # S_I: [B, I, c_s] 单轨迹特征(用于条件化)
            # Z_II: [B, I, I, c_tokenpair] 配对特征(用作注意力偏置)
            A_I = block(
                A_I,
                S_I,
                Z_II,
                indices=indices,  # SL2稀疏注意力索引
                full=full,  # 是否使用完整注意力(内存换速度)
            )

        return A_I


class LocalAtomTransformer(nn.Module):
    def __init__(self, c_atom, c_s, c_atompair, atom_transformer_block, n_blocks):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                StructureLocalAtomTransformerBlock(
                    c_atom=c_atom,
                    c_s=c_s,
                    c_atompair=c_atompair,
                    **atom_transformer_block,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, Q_L, C_L, P_LL, **kwargs):
        for block in self.blocks:
            Q_L = block(Q_L, C_L, P_LL, **kwargs)
        return Q_L


class StructureLocalAtomTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        c_atom,
        c_s,
        c_atompair,
        dropout,
        no_residual_connection_between_attention_and_transition,
        **transformer_block,
    ):
        super().__init__()
        assert not no_residual_connection_between_attention_and_transition
        self.c_s = c_s
        self.dropout = nn.Dropout(dropout)
        self.attention_pair_bias = LocalAttentionPairBias(
            c_a=c_atom, c_s=c_s, c_pair=c_atompair, **transformer_block
        )
        if exists(c_s):
            self.transition_block = ConditionedTransitionBlock(c_token=c_atom, c_s=c_s)
        else:
            self.transition_block = Transition(c=c_atom, n=4)

    def forward(
        self,
        Q_L,  # [..., I, C_token]
        C_L,  # [..., I, C_s]
        P_LL,  # [..., I, I, C_tokenpair]
        f=None,
        chunked_pairwise_embedder=None,
        initializer_outputs=None,
        **kwargs,
    ):
        Q_L = Q_L + self.dropout(
            self.attention_pair_bias(
                Q_L,
                C_L,
                P_LL,
                f=f,
                chunked_pairwise_embedder=chunked_pairwise_embedder,
                initializer_outputs=initializer_outputs,
                **kwargs,
            )
        )
        if exists(C_L):
            Q_L = Q_L + self.transition_block(Q_L, C_L)
        else:
            Q_L = Q_L + self.transition_block(Q_L)
        return Q_L


class CompactStreamingDecoder(nn.Module):
    def __init__(
        self,
        c_atom,
        c_atompair,
        c_token,
        c_s,
        c_tokenpair,
        atom_transformer_block,
        upcast,
        downcast,
        n_blocks,
        diffusion_transformer_block=False,
    ):
        super().__init__()
        self.n_blocks = n_blocks

        self.upcast = nn.ModuleList(
            [Upcast(c_atom=c_atom, c_token=c_token, **upcast) for _ in range(n_blocks)]
        )
        self.atom_transformer = nn.ModuleList(
            [
                StructureLocalAtomTransformerBlock(
                    c_atom=c_atom,
                    c_s=c_atom,
                    c_atompair=c_atompair,
                    **atom_transformer_block,
                )
                for _ in range(n_blocks)
            ]
        )
        self.downcast = Downcast(c_atom=c_atom, c_token=c_token, c_s=c_s, **downcast)

    def forward(
        self,
        A_I,
        S_I,
        Z_II,
        Q_L,
        C_L,
        P_LL,
        tok_idx,
        indices,
        f=None,
        chunked_pairwise_embedder=None,
        initializer_outputs=None,
    ):
        for i in range(self.n_blocks):
            Q_L = self.upcast[i](Q_L, A_I, tok_idx=tok_idx)
            Q_L = self.atom_transformer[i](
                Q_L,
                C_L,
                P_LL,
                indices=indices,
                f=f,
                chunked_pairwise_embedder=chunked_pairwise_embedder,
                initializer_outputs=initializer_outputs,
            )

        # Downcast to sequence
        A_I = self.downcast(Q_L.detach(), A_I.detach(), S_I.detach(), tok_idx=tok_idx)

        o = {}
        return A_I, Q_L, o
