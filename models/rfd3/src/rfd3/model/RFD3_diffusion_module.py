import functools
import logging
import os
from contextlib import ExitStack
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from rfd3.model.layers.block_utils import (
    bucketize_scaled_distogram,
    create_attention_indices,
)
from rfd3.model.layers.blocks import (
    CompactStreamingDecoder,
    Downcast,
    LinearEmbedWithPool,
    LinearSequenceHead,
    LocalAtomTransformer,
    LocalTokenTransformer,
)
from rfd3.model.layers.encoders import (
    DiffusionTokenEncoder,
)
from rfd3.model.layers.layer_utils import RMSNorm, linearNoBias

from foundry.model.layers.blocks import (
    FourierEmbedding,
)

logger = logging.getLogger(__name__)


class RFD3DiffusionModule(nn.Module):
    """
    RFDiffusion3 扩散模块 (Algorithm 5: Diffusion forward pass with recycling)
    RFDiffusion3 Diffusion Module

    这是RFD3的核心扩散模块,实现了SI中的Algorithm 5。
    采用UNet风格的架构,在token和atom两个层级处理特征。
    This is the core diffusion module of RFD3, implementing Algorithm 5 from SI.
    Uses a UNet-style architecture for processing features across tokens and atoms.

    主要组件 / Main Components:
    - Encoder: 局部atom transformer用于atom级特征编码
    - Encoder: Local atom transformer for atom-level feature encoding
    - Diffusion Token Encoder: 嵌入噪声尺度和循环distogram
    - Diffusion Token Encoder: Embeds noise scale and recycled distogram
    - Diffusion Transformer: 在token级进行稀疏注意力
    - Diffusion Transformer: Sparse attention at token level
    - Decoder: 解码为结构更新
    - Decoder: Decodes to structure updates

    参数 / Parameters:
        c_atom: Atom级别特征维度 / Atom-level feature dimension
        c_atompair: Atom配对特征维度 / Atom pairwise feature dimension
        c_token: Token级别特征维度 / Token-level feature dimension
        c_s: 单轨迹特征维度 / Single track feature dimension
        c_z: 配对轨迹特征维度 / Pair track feature dimension
        c_t_embed: 时间嵌入维度 / Time embedding dimension
        sigma_data: EDM数据方差 / EDM data variance (default: 16)
        f_pred: 预测类型 ("edm", "unconditioned", "noise_pred") / Prediction type
        n_attn_seq_neighbours: 序列局部注意力邻居数 (默认32) / Sequence-local attention neighbors
        n_attn_keys: 结构局部注意力键数 (默认128) / Structure-local attention keys
        n_recycle: 循环次数 (默认2) / Number of recycling iterations
    """
    def __init__(
        self,
        *,
        c_atom: int,
        c_atompair: int,
        c_token: int,
        c_s: int,
        c_z: int,
        c_t_embed: int,
        sigma_data: float,
        f_pred: str,
        n_attn_seq_neighbours: int,
        n_attn_keys: int,
        n_recycle: int,
        atom_attention_encoder: Dict[str, Any],
        diffusion_token_encoder: Dict[str, Any],
        diffusion_transformer: Dict[str, Any],
        atom_attention_decoder: Dict[str, Any],
        # upcast,
        downcast: Dict[str, Any],
        use_local_token_attention: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        self.sigma_data = sigma_data
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.c_s = c_s
        self.c_z = c_z
        self.f_pred = f_pred
        self.n_attn_seq_neighbours = n_attn_seq_neighbours
        self.n_attn_keys = n_attn_keys
        self.use_local_token_attention = use_local_token_attention

        # ===== 辅助模块 / Auxiliary Modules =====
        # Algorithm 5 步骤1: 坐标缩放 / Step 1: Scale positions
        self.process_r = linearNoBias(3, c_atom)  # [3] -> [c_atom]
        # Algorithm 5 步骤15: 坐标更新投影 / Step 15: Coordinate update projection
        self.to_r_update = nn.Sequential(RMSNorm((c_atom,)), linearNoBias(c_atom, 3))
        # 序列预测头 / Sequence prediction head
        self.sequence_head = LinearSequenceHead(c_token=c_token)

        # 循环和distogram参数 / Recycling and distogram parameters
        self.n_recycle = n_recycle  # 默认2次循环 / Default 2 recycling iterations
        self.n_bins = 65  # Distogram的离散化bin数 / Number of distogram bins
        self.bucketize_fn = functools.partial(
            bucketize_scaled_distogram,
            min_dist=1,   # 最小距离1Å / Minimum distance 1Å
            max_dist=30,  # 最大距离30Å / Maximum distance 30Å
            sigma_data=1,
            n_bins=self.n_bins,
        )

        # ===== 时间处理 (Algorithm 16 & 17) / Time Processing =====
        # Algorithm 16: FourierEmbedding - 用于时间编码
        # Algorithm 16: FourierEmbedding - for time encoding
        self.fourier_embedding = nn.ModuleList(
            [FourierEmbedding(c_t_embed), FourierEmbedding(c_t_embed)]  # [0]: atom级, [1]: token级
        )
        # Algorithm 17: Processing of noise conditioning features
        # 步骤1: 处理Fourier特征并投影 / Step 1: Process Fourier features and project
        self.process_n = nn.ModuleList(
            [
                nn.Sequential(RMSNorm(c_t_embed), linearNoBias(c_t_embed, c_atom)),  # [0]: 用于atom
                nn.Sequential(RMSNorm(c_t_embed), linearNoBias(c_t_embed, c_s)),     # [1]: 用于token
            ]
        )
        # ===== Algorithm 9: Downcast (池化操作) / Pooling Operations =====
        # 将atom特征池化到token特征 / Pool atom features to token features
        self.downcast_c = Downcast(c_atom=c_atom, c_token=c_s, c_s=None, **downcast)
        self.downcast_q = Downcast(c_atom=c_atom, c_token=c_token, c_s=c_s, **downcast)
        self.process_a = LinearEmbedWithPool(c_token)  # Algorithm 5 步骤2
        self.process_c = nn.Sequential(RMSNorm(c_atom), linearNoBias(c_atom, c_atom))  # 步骤7

        # ===== UNet风格架构 / UNet-style Architecture =====
        # 在token和atom两个层级处理特征 / Process features across tokens and atoms

        # Algorithm 5 步骤8: Local-atom transformer (编码器)
        # Algorithm 5 Step 8: Local-atom transformer (encoder)
        self.encoder = LocalAtomTransformer(
            c_atom=c_atom, c_s=c_atom, c_atompair=c_atompair, **atom_attention_encoder
        )

        # Algorithm 5 步骤12: Diffusion token encoder
        # 嵌入噪声尺度和循环distogram / Embed noise scale and recycled distogram
        self.diffusion_token_encoder = DiffusionTokenEncoder(
            c_s=c_s,
            c_token=c_token,
            c_z=c_z,
            c_atompair=c_atompair,
            **diffusion_token_encoder,
        )

        # Algorithm 5 步骤13: Sparse attention at token level
        # Token级稀疏注意力 / Token-level sparse attention
        self.diffusion_transformer = LocalTokenTransformer(
            c_token=c_token,
            c_tokenpair=c_z,
            c_s=c_s,
            **diffusion_transformer,
        )

        # Algorithm 5 步骤14: Up-projection and decode to structure
        # 上投影并解码为结构 / Up-projection and decode to structure
        self.decoder = CompactStreamingDecoder(
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token,
            c_s=c_s,
            c_tokenpair=c_z,
            **atom_attention_decoder,
        )

    def scale_positions_in(self, X_noisy_L: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        输入坐标缩放 (EDM预条件化) / Input coordinate scaling (EDM preconditioning)

        根据EDM框架对噪声坐标进行缩放,使其适合神经网络输入。
        Scale noisy coordinates according to EDM framework for neural network input.

        参数 / Args:
            X_noisy_L: [B, L, 3] 噪声坐标 / Noisy coordinates
            t: [B] 或 [B, L] 噪声水平 / Noise level

        返回 / Returns:
            R_noisy_L: [B, L, 3] 缩放后的坐标 / Scaled coordinates
        """
        # 扩展时间维度以匹配坐标形状 / Expand time dimension to match coordinate shape
        if t.ndim == 1:
            t = t[..., None, None]  # [B] -> [B, 1, 1]
        elif t.ndim == 2:
            t = t[..., None]  # [B, L] -> [B, L, 1]

        # EDM预条件化公式: c_in(σ) * X / Preconditioning formula
        if self.f_pred == "edm":
            # c_in(σ) = 1 / sqrt(σ^2 + σ_data^2)
            R_noisy_L = X_noisy_L / torch.sqrt(t**2 + self.sigma_data**2)
        elif self.f_pred == "unconditioned":
            # 无条件:清零输入 / Unconditional: zero input
            R_noisy_L = torch.zeros_like(X_noisy_L)
        elif self.f_pred == "noise_pred":
            # 噪声预测:直接使用噪声坐标 / Noise prediction: use noisy coordinates directly
            R_noisy_L = X_noisy_L
        else:
            raise Exception(f"{self.f_pred=} unrecognized")
        return R_noisy_L

    def scale_positions_out(
        self, R_update_L: torch.Tensor, X_noisy_L: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        输出坐标缩放 (EDM去预条件化) / Output coordinate scaling (EDM de-preconditioning)

        将神经网络输出转换为去噪后的坐标,遵循EDM框架。
        Convert neural network output to denoised coordinates following EDM framework.

        参数 / Args:
            R_update_L: [B, L, 3] 网络预测的更新 / Network predicted update
            X_noisy_L: [B, L, 3] 输入的噪声坐标 / Input noisy coordinates
            t: [B] 或 [B, L] 噪声水平 / Noise level

        返回 / Returns:
            X_out_L: [B, L, 3] 去噪后的坐标 / Denoised coordinates
        """
        # 扩展时间维度 / Expand time dimension
        if t.ndim == 1:
            t = t[..., None, None]  # [B] -> [B, 1, 1]
        elif t.ndim == 2:
            t = t[..., None]  # [B, L] -> [B, L, 1]

        # EDM去预条件化公式 / EDM de-preconditioning formula
        if self.f_pred == "edm":
            # c_skip(σ) * X + c_out(σ) * F_θ
            # c_skip = σ_data^2 / (σ^2 + σ_data^2)
            # c_out = σ * σ_data / sqrt(σ^2 + σ_data^2)
            X_out_L = (self.sigma_data**2 / (self.sigma_data**2 + t**2)) * X_noisy_L + (
                self.sigma_data * t / (self.sigma_data**2 + t**2) ** 0.5
            ) * R_update_L
        elif self.f_pred == "unconditioned":
            # 无条件:直接使用预测 / Unconditional: use prediction directly
            X_out_L = R_update_L
        elif self.f_pred == "noise_pred":
            # 噪声预测: X - ε / Noise prediction: X - ε
            X_out_L = X_noisy_L + R_update_L
        else:
            raise Exception(f"{self.f_pred=} unrecognized")
        return X_out_L

    def process_time_(self, t_L: torch.Tensor, i: int) -> torch.Tensor:
        """
        时间条件化特征处理 (Algorithm 17) / Time conditioning feature processing

        将噪声水平 t 编码为特征,用于条件化扩散过程。
        Encode noise level t into features for conditioning the diffusion process.

        参数 / Args:
            t_L: [B, L] 每个atom/token的噪声水平 / Noise level per atom/token
            i: 0=atom级, 1=token级 / 0=atom-level, 1=token-level

        返回 / Returns:
            C_L: [B, L, c_atom] 或 [B, L, c_s] 时间条件化特征 / Time conditioning features
        """
        # Algorithm 17 步骤1: 对数时间编码 + Fourier特征
        # Log-time encoding + Fourier features
        # log(t/σ_data) / 4 转换为频率范围
        C_L = self.process_n[i](
            self.fourier_embedding[i](
                1 / 4 * torch.log(torch.clamp(t_L, min=1e-20) / self.sigma_data)
            )
        )
        # Algorithm 17 步骤2: 屏蔽零时间(固定motif区域)
        # Mask out zero-time features (fixed motif regions)
        C_L = C_L * (t_L > 0).float()[..., None]  # [B, L, c_atom/c_s]
        return C_L

    def forward(
        self,
        X_noisy_L: torch.Tensor,
        t: torch.Tensor,
        f: Dict[str, Any],
        # Features from initialization
        Q_L_init: torch.Tensor,
        C_L: torch.Tensor,
        P_LL: torch.Tensor,
        S_I: torch.Tensor,
        Z_II: torch.Tensor,
        n_recycle: Optional[int] = None,
        # Chunked memory optimization parameters
        chunked_pairwise_embedder: Optional[Any] = None,
        initializer_outputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        Algorithm 5: Diffusion forward pass with recycling (./docs/rf3_si.pdf)
        扩散前向传播
        Diffusion forward pass with recycling

        给定噪声坐标和编码特征,计算去噪后的位置。
        Computes denoised positions given encoded features and noisy coordinates.

        参数 / Args:
            X_noisy_L: [B, L, 3] 噪声坐标 / Noisy coordinates
            t: [B] 噪声水平 / Noise level
            f: 特征字典 / Feature dictionary
            Q_L_init: [L, c_atom] 初始atom特征 / Initial atom features (from TokenInitializer)
            C_L: [L, c_atom] 条件化atom特征 / Conditioned atom features
            P_LL: [L, L, c_atompair] Atom配对特征 / Atom pairwise features
            S_I: [I, c_s] Token单轨迹特征 / Token single features
            Z_II: [I, I, c_z] Token配对特征 / Token pair features
            n_recycle: 循环次数 / Number of recycling iterations

        返回 / Returns:
            outputs: 包含去噪坐标和序列预测的字典 / Dictionary with denoised coordinates and sequence predictions
        """
        # Algorithm 5 - line 1: Collect inputs and create attention indices
        # ===== 步骤1: 收集输入和创建注意力索引 / Step 1: Collect inputs and create attention indices =====
        tok_idx = f["atom_to_token_map"]  # [L] atom到token的映射 / Atom to token mapping
        L = len(tok_idx)  # 原子总数 / Total number of atoms
        I = tok_idx.max() + 1  # Token总数 / Total number of tokens

        # 创建SL2稀疏注意力索引 (序列局部 + 结构局部)
        # Create SL2 sparse attention indices (sequence-local + structure-local)
        f["attn_indices"] = create_attention_indices(
            X_L=X_noisy_L,
            f=f,
            n_attn_keys=self.n_attn_keys,  # 结构局部键数 (默认128)
            n_attn_seq_neighbours=self.n_attn_seq_neighbours,  # 序列局部邻居 (默认32)
        )

        # Algorithm 5 - line 2: Expand time tensors and mask fixed regions
        # ===== 步骤2-3: 扩展时间张量并屏蔽固定区域 / Step 2-3: Expand time tensors and mask fixed regions =====
        # t_L: [B, L] 每个atom的噪声水平,motif区域为0
        t_L = t.unsqueeze(-1).expand(-1, L) * (
            ~f["is_motif_atom_with_fixed_coord"]
        ).float().unsqueeze(0)
        # t_I: [B, I] 每个token的噪声水平,完全固定的token为0
        t_I = t.unsqueeze(-1).expand(-1, I) * (
            ~f["is_motif_token_with_fully_fixed_coord"]
        ).float().unsqueeze(0)

        # Algorithm 5 - line 3: Scale positions (EDM preconditioning)
        # ===== 步骤4: 坐标缩放 (EDM预条件化) / Step 4: Scale positions (EDM preconditioning) =====
        R_L_uniform = self.scale_positions_in(X_noisy_L, t)  # [B, L, 3] 均匀缩放用于distogram
        R_noisy_L = self.scale_positions_in(X_noisy_L, t_L)  # [B, L, 3] 每atom缩放用于特征

        # Algorithm 5 - line 4: Pool initial representation to token level (Downcast)
        # ===== 步骤5: 池化初始表示到token级 (Algorithm 9: Downcast) / Step 5: Pool initial representation to token level =====
        A_I = self.process_a(R_noisy_L, tok_idx=tok_idx)  # [B, I, c_token] 从坐标池化的token特征
        S_I = self.downcast_c(C_L, S_I, tok_idx=tok_idx)  # [I, c_s] 从atom特征池化的token特征

        # Algorithm 5 - line 5: Add position and time embeddings
        # ===== 步骤6-7: 添加批次级特征 (时间条件化) / Step 6-7: Add batch-wise features (time conditioning) =====
        # Algorithm 5 步骤1: 坐标投影 + 初始化特征
        Q_L = Q_L_init.unsqueeze(0) + self.process_r(R_noisy_L)  # [B, L, c_atom]

        # Algorithm 5 - line 6: Add time conditioning (Algorithm 17)
        # Algorithm 17: 添加时间条件化特征
        C_L = C_L.unsqueeze(0) + self.process_time_(t_L, i=0)  # [B, L, c_atom] atom级
        S_I = S_I.unsqueeze(0) + self.process_time_(t_I, i=1)  # [B, I, c_s] token级
        C_L = C_L + self.process_c(C_L)  # [B, L, c_atom] 额外的MLP处理

        # Algorithm 5 - line 7: Local-atom self-attention encoder
        # ===== 步骤8: Local-Atom Self Attention (编码器) / Step 8: Local-Atom Self Attention (encoder) =====
        # Algorithm 5 步骤8: 局部atom transformer
        if chunked_pairwise_embedder is not None:
            # 分块模式:传递分块嵌入器用于内存优化 / Chunked mode: pass chunked embedder for memory optimization
            Q_L = self.encoder(
                Q_L,
                C_L,
                P_LL=None,
                indices=f["attn_indices"],
                f=f,  # 传递特征字典用于分块计算
                chunked_pairwise_embedder=chunked_pairwise_embedder,
                initializer_outputs=initializer_outputs,
            )
        else:
            # 标准模式:使用完整的P_LL / Standard mode: use full P_LL
            Q_L = self.encoder(Q_L, C_L, P_LL, indices=f["attn_indices"])

        # Algorithm 5 - line 8: Pool atom features to token level (Downcast)
        # ===== 步骤9: 池化到token级准备transformer / Step 9: Pool to token level for transformer =====
        # Algorithm 9: Downcast - 将atom特征池化为token特征
        A_I = self.downcast_q(Q_L, A_I=A_I, S_I=S_I, tok_idx=tok_idx)  # [B, I, c_token]

        # Algorithm 5 - line 9: for r ∈ [1, ..., n_recycle] do (Recycling loop)
        # ===== 步骤10-17: 循环处理 (Recycling Loop) / Step 10-17: Recycling loop =====
        # Algorithm 5 步骤10-17: 带distogram循环的迭代细化
        recycled_features = self.forward_with_recycle(
            n_recycle,
            X_noisy_L=X_noisy_L,
            R_L_uniform=R_L_uniform,
            t_L=t_L,
            f=f,
            Q_L=Q_L,
            C_L=C_L,
            P_LL=P_LL,
            A_I=A_I,
            S_I=S_I,
            Z_II=Z_II,
            chunked_pairwise_embedder=chunked_pairwise_embedder,
            initializer_outputs=initializer_outputs,
        )

        # Algorithm 5 - line 17: end for
        # Algorithm 5 - line 18: return x̂0
        # ===== 收集输出 / Collect outputs =====
        outputs = {
            "X_L": recycled_features["X_L"],  # [B, L, 3] 去噪后的坐标 / Denoised positions
            "sequence_indices_I": recycled_features["sequence_indices_I"],  # 序列索引 / Sequence indices
            "sequence_logits_I": recycled_features["sequence_logits_I"],  # 序列logits / Sequence logits
        }
        return outputs

    def forward_with_recycle(
        self,
        n_recycle: Optional[int],
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        循环前向传播包装器 (Algorithm 5 步骤10-17) / Recycling forward pass wrapper

        迭代细化结构预测,使用前一次循环的distogram作为条件。
        Iteratively refines structure prediction using previous cycle's distogram as conditioning.

        参数 / Args:
            n_recycle: 循环次数,训练时必须提供,推理时使用self.n_recycle (默认2)
                      Number of recycling iterations, must be provided during training

        返回 / Returns:
            recycled_features: 最终循环的输出 / Output from final recycling iteration
        """
        # 推理时使用默认循环次数,训练时必须明确指定
        # Use default n_recycle during inference, must be explicit during training
        if not self.training:
            n_recycle = self.n_recycle
        else:
            assert n_recycle is not None

        recycled_features = {}  # 存储上一次循环的输出 / Store previous cycle outputs
        for i in range(n_recycle):
            with ExitStack() as stack:
                # 只在最后一次循环保留梯度 / Only keep gradients in final cycle
                last = not (i < n_recycle - 1)
                if not last:
                    # 中间循环使用no_grad节省内存 / Use no_grad for intermediate cycles to save memory
                    stack.enter_context(torch.no_grad())

                # 清除autocast缓存(PyTorch bug的解决方法)
                # Clear autocast cache (workaround for PyTorch bug)
                # See: https://github.com/pytorch/pytorch/issues/65766
                if torch.is_grad_enabled():
                    torch.clear_autocast_cache()

                # 运行单次循环迭代 / Run single recycling iteration
                # 将前一次的distogram (D_II_self) 和坐标 (X_L) 作为条件
                recycled_features = self.process_(
                    D_II_self=recycled_features.get("D_II_self"),  # [B, I, I, n_bins] 或 None
                    X_L_self=recycled_features.get("X_L"),  # [B, L, 3] 或 None
                    **kwargs,
                )

        return recycled_features

    def process_(
        self,
        D_II_self: Optional[torch.Tensor],
        X_L_self: Optional[torch.Tensor],
        *,
        R_L_uniform: torch.Tensor,
        X_noisy_L: torch.Tensor,
        t_L: torch.Tensor,
        f: Dict[str, Any],
        Q_L: torch.Tensor,
        C_L: torch.Tensor,
        P_LL: torch.Tensor,
        A_I: torch.Tensor,
        S_I: torch.Tensor,
        Z_II: torch.Tensor,
        chunked_pairwise_embedder: Optional[Any] = None,
        initializer_outputs: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        单次循环迭代处理 (Algorithm 5 步骤12-16) / Single recycling iteration processing

        执行一次完整的transformer前向传播:编码器 -> transformer -> 解码器 -> 更新。
        Performs one complete transformer forward pass: encoder -> transformer -> decoder -> update.

        参数 / Args:
            D_II_self: [B, I, I, n_bins] 或 None - 前一次循环的distogram / Previous cycle's distogram
            X_L_self: [B, L, 3] 或 None - 前一次循环的坐标 / Previous cycle's coordinates
            R_L_uniform: [B, L, 3] 均匀缩放的坐标 / Uniformly scaled coordinates
            X_noisy_L: [B, L, 3] 原始噪声坐标 / Original noisy coordinates
            t_L: [B, L] 时间条件 / Time conditioning
            f: 特征字典 / Feature dictionary
            Q_L: [B, L, c_atom] Atom查询特征 / Atom query features
            C_L: [B, L, c_atom] Atom条件特征 / Atom conditioning features
            P_LL: [L, L, c_atompair] Atom配对特征 / Atom pairwise features
            A_I: [B, I, c_token] Token特征 / Token features
            S_I: [B, I, c_s] Token单轨迹特征 / Token single features
            Z_II: [I, I, c_z] Token配对特征 / Token pair features

        返回 / Returns:
            包含更新坐标、distogram和序列预测的字典 / Dictionary with updated coordinates, distogram, and sequence predictions
        """
        # Algorithm 5 - line 10: DiffusionTokenEncoder (Algorithm 12)
        # ===== 步骤12: DiffusionTokenEncoder - 嵌入噪声尺度和循环distogram =====
        # Step 12: DiffusionTokenEncoder - Embed noise scale and recycled distogram
        # Algorithm 12: 将当前坐标的distogram和前一次循环的distogram嵌入到Z_II中
        S_I, Z_II = self.diffusion_token_encoder(
            f=f,
            R_L=R_L_uniform,  # [B, L, 3] 用于计算当前distogram
            D_II_self=D_II_self,  # [B, I, I, n_bins] 前一次循环的self-conditioning distogram
            S_init_I=S_I,  # [B, I, c_s] 初始token特征
            Z_init_II=Z_II,  # [I, I, c_z] 初始token配对特征
            C_L=C_L,  # [B, L, c_atom] Atom条件特征
            P_LL=P_LL,  # [L, L, c_atompair] Atom配对特征
        )

        # Algorithm 5 - line 11: DiffusionTransformer (Algorithm 6: LocalTokenTransformer)
        # ===== 步骤13: DiffusionTransformer - Token级稀疏注意力 =====
        # Step 13: DiffusionTransformer - Token-level sparse attention
        # Algorithm 6: LocalTokenTransformer with SL2 sparse attention
        A_I = self.diffusion_transformer(
            A_I,  # [B, I, c_token] Token特征
            S_I,  # [B, I, c_s] 单轨迹特征
            Z_II,  # [B, I, I, c_z] 配对特征(用作注意力偏置)
            f=f,
            X_L=(
                # 使用当前坐标或前一次循环的坐标(仅C-alpha)用于结构局部注意力
                X_noisy_L[..., f["is_ca"], :]
                if X_L_self is None
                else X_L_self[..., f["is_ca"], :]
            ),
            full=not (os.environ.get("RFD3_LOW_MEMORY_MODE", None) == "1"),  # 低内存模式标志
        )

        # Algorithm 5 - line 12: Decoder (CompactStreamingDecoder with Upcast)
        # ===== 步骤14: Decoder - 上投影并解码为结构 =====
        # Step 14: Decoder - Up-projection and decode to structure
        # CompactStreamingDecoder: Token -> Atom特征,包含Algorithm 10 (Upcast)

        if chunked_pairwise_embedder is not None:
            # 分块模式:传递嵌入器,不使用预计算的P_LL / Chunked mode: pass embedder, no pre-computed P_LL
            A_I, Q_L, o = self.decoder(
                A_I,
                S_I,
                Z_II,
                Q_L,
                C_L,
                P_LL=None,  # 分块模式不使用 / Not used in chunked mode
                tok_idx=f["atom_to_token_map"],
                indices=f["attn_indices"],
                f=f,  # 传递特征字典用于按需计算 / Pass f for on-demand computation
                chunked_pairwise_embedder=chunked_pairwise_embedder,
                initializer_outputs=initializer_outputs,
            )
        else:
            # 标准模式:使用完整的P_LL / Standard mode: use full P_LL
            A_I, Q_L, o = self.decoder(
                A_I,
                S_I,
                Z_II,
                Q_L,
                C_L,
                P_LL=P_LL,  # [L, L, c_atompair] 预计算的atom配对特征
                tok_idx=f["atom_to_token_map"],
                indices=f["attn_indices"],
            )

        # Algorithm 5 - line 13: Project atom features to coordinate update
        # ===== 步骤15-16: 坐标更新和去预条件化 =====
        # Step 15-16: Coordinate update and de-preconditioning
        # Algorithm 5 步骤15: 投影atom特征到3D坐标更新
        R_update_L = self.to_r_update(Q_L)  # [B, L, 3] 预测的坐标更新

        # Algorithm 5 - line 14: De-preconditioning (EDM scaling inverse)
        # 步骤16: EDM去预条件化,得到去噪后的坐标
        X_out_L = self.scale_positions_out(R_update_L, X_noisy_L, t_L)  # [B, L, 3]

        # Algorithm 5 - line 15: Compute sequence logits and distogram for recycling
        # ===== 辅助输出:序列预测和distogram =====
        # Auxiliary outputs: sequence prediction and distogram
        # 序列预测头:从token特征预测残基类型
        sequence_logits_I, sequence_indices_I = self.sequence_head(A_I=A_I)

        # Algorithm 5 - line 16: Bucketize distogram for next recycling iteration
        # 计算distogram用于下一次循环的self-conditioning
        # 使用detach()防止梯度回传到前一次循环
        D_II_self = self.bucketize_fn(X_out_L[..., f["is_ca"], :].detach())  # [B, I, I, n_bins]

        return {
            "X_L": X_out_L,  # [B, L, 3] 更新后的坐标
            "D_II_self": D_II_self,  # [B, I, I, n_bins] 用于下一次循环的distogram
            "sequence_logits_I": sequence_logits_I,  # [B, I, 21] 序列预测logits
            "sequence_indices_I": sequence_indices_I,  # [B, I] 序列索引
        } | o  # 合并解码器的额外输出
