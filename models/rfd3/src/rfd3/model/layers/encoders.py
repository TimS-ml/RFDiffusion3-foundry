import functools
import logging

import torch
import torch.nn as nn
from rfd3.model.layers.block_utils import (
    bucketize_scaled_distogram,
    pairwise_mean_pool,
)
from rfd3.model.layers.blocks import (
    Downcast,
    LocalAtomTransformer,
    OneDFeatureEmbedder,
    PositionPairDistEmbedder,
    RelativePositionEncodingWithIndexRemoval,
    SinusoidalDistEmbed,
)
from rfd3.model.layers.chunked_pairwise import (
    ChunkedPairwiseEmbedder,
    ChunkedPositionPairDistEmbedder,
    ChunkedSinusoidalDistEmbed,
)
from rfd3.model.layers.layer_utils import (
    RMSNorm,
    Transition,
    linearNoBias,
)
from rfd3.model.layers.pairformer_layers import PairformerBlock

from foundry.common import exists
from foundry.training.checkpoint import activation_checkpointing

logger = logging.getLogger(__name__)


class TokenInitializer(nn.Module):
    """
    Token初始化器 (Algorithm 3 & 4: Token and Atom initializer)
    Token Initializer

    RFD3的初始化模块,实现SI中的Algorithm 3和4。
    负责从原始特征生成初始的token级和atom级表示。
    Initialization module for RFD3, implementing Algorithms 3 and 4 from SI.
    Responsible for generating initial token-level and atom-level representations from raw features.

    主要功能 / Main Functions:
    1. Token初始化 (Algorithm 3):
       - 嵌入1D特征 (残基类型、配对特征等)
       - 相对位置编码
       - Pairformer处理配对特征
       - 生成 S_I (token单特征) 和 Z_II (token配对特征)

    2. Atom初始化 (Algorithm 4):
       - 嵌入atom级1D特征
       - Motif位置和参考坐标编码
       - 生成 Q_L (atom初始特征), C_L (atom条件特征), P_LL (atom配对特征)

    参数 / Parameters:
        c_s: Token单轨迹特征维度 / Token single track feature dimension
        c_z: Token配对特征维度 / Token pair feature dimension
        c_atom: Atom特征维度 / Atom feature dimension
        c_atompair: Atom配对特征维度 / Atom pairwise feature dimension
        use_chunked_pll: 是否使用分块P_LL计算(内存优化) / Whether to use chunked P_LL computation
    """

    def __init__(
        self,
        c_s,
        c_z,
        c_atom,
        c_atompair,
        relative_position_encoding,
        n_pairformer_blocks,
        pairformer_block,
        downcast,
        token_1d_features,
        atom_1d_features,
        atom_transformer,
        use_chunked_pll=False,  # New parameter for memory optimization
    ):
        super().__init__()

        # 存储分块模式标志 / Store chunked mode flag
        self.use_chunked_pll = use_chunked_pll

        # ===== Algorithm 3: Token Initializer - 1D特征嵌入 / 1D Feature Embedding =====
        # Algorithm 15: OneDFeatureEmbedder - 嵌入原始1D特征
        self.atom_1d_embedder_1 = OneDFeatureEmbedder(atom_1d_features, c_s)  # 用于token初始化
        self.atom_1d_embedder_2 = OneDFeatureEmbedder(atom_1d_features, c_atom)  # 用于atom初始化
        self.token_1d_embedder = OneDFeatureEmbedder(token_1d_features, c_s)

        # Algorithm 9: Downcast - 从atom特征池化到token特征
        self.downcast_atom = Downcast(c_atom=c_s, c_token=c_s, c_s=None, **downcast)
        # Algorithm 3 步骤5: Transition层用于特征混合
        self.transition_post_token = Transition(c=c_s, n=2)
        self.transition_post_atom = Transition(c=c_s, n=2)
        self.process_s_init = nn.Sequential(
            RMSNorm(c_s),
            linearNoBias(c_s, c_s),
        )

        # ===== Algorithm 3 步骤6-8: 配对特征初始化 / Pair Feature Initialization =====
        # 步骤6-7: 从单特征投影到配对特征 (outer sum)
        self.to_z_init_i = linearNoBias(c_s, c_z)  # S_I -> Z_II (i维度)
        self.to_z_init_j = linearNoBias(c_s, c_z)  # S_I -> Z_II (j维度)
        # 步骤8: 相对位置编码
        self.relative_position_encoding = RelativePositionEncodingWithIndexRemoval(
            c_z=c_z, **relative_position_encoding
        )
        self.relative_position_encoding2 = RelativePositionEncodingWithIndexRemoval(
            c_z=c_z, **relative_position_encoding
        )
        # Token间的化学键编码
        self.process_token_bonds = linearNoBias(1, c_z)

        # ===== Algorithm 3 步骤9-12: 配对特征处理 / Pair Feature Processing =====
        # 步骤13-19: 混合多个配对特征源并后处理
        self.process_z_init = nn.Sequential(
            RMSNorm(c_z * 2),  # 处理拼接的配对特征
            linearNoBias(c_z * 2, c_z),
        )
        self.transition_1 = nn.ModuleList(
            [
                Transition(c=c_z, n=2),
                Transition(c=c_z, n=2),
            ]
        )
        # Algorithm 14: PositionPairDistEmbedder - 参考坐标的距离编码
        self.ref_pos_embedder_tok = PositionPairDistEmbedder(c_z, embed_frame=False)

        # ===== Algorithm 3 步骤10-12: Pairformer块 / Pairformer Blocks =====
        # Algorithm 7: TransformerBlock - 全注意力transformer处理token特征
        self.transformer_stack = nn.ModuleList(
            [
                PairformerBlock(c_s=c_s, c_z=c_z, **pairformer_block)
                for _ in range(n_pairformer_blocks)
            ]
        )

        # ===== Algorithm 4: Atom Initializer - Atom级特征处理 / Atom-level Feature Processing =====
        # 步骤2: 从token特征投影到atom特征 / Project from token features to atom features
        self.process_s_trunk = nn.Sequential(RMSNorm(c_s), linearNoBias(c_s, c_atom))

        # 步骤5-7: Atom配对特征的MLP处理 / MLP processing for atom pairwise features
        # 步骤5: 处理单atom特征(l维度) / Process single atom features (l dimension)
        self.process_single_l = nn.Sequential(
            nn.ReLU(), linearNoBias(c_atom, c_atompair)
        )
        # 步骤5: 处理单atom特征(m维度) / Process single atom features (m dimension)
        self.process_single_m = nn.Sequential(
            nn.ReLU(), linearNoBias(c_atom, c_atompair)
        )
        # 步骤6: 从token配对特征投影 / Project from token pair features
        self.process_z = nn.Sequential(RMSNorm(c_z), linearNoBias(c_z, c_atompair))

        # ===== Algorithm 4 步骤3-4: 位置和距离嵌入器 / Position and Distance Embedders =====
        # 总是创建这些MLP - 在分块和标准模式间共享 / Always create these MLPs - shared between modes
        # Algorithm 13: SinusoidalDistEmbed - Motif位置的正弦距离嵌入
        self.motif_pos_embedder = SinusoidalDistEmbed(c_atompair=c_atompair)
        # Algorithm 14: PositionPairDistEmbedder - 参考位置的配对距离嵌入
        self.ref_pos_embedder = PositionPairDistEmbedder(c_atompair, embed_frame=False)
        # 步骤7: 深度MLP用于混合配对特征 / Deep MLP for mixing pairwise features
        self.pair_mlp = nn.Sequential(
            nn.ReLU(),
            linearNoBias(c_atompair, c_atompair),
            nn.ReLU(),
            linearNoBias(c_atompair, c_atompair),
            nn.ReLU(),
            linearNoBias(c_atompair, c_atompair),
        )

        # ===== 分块P_LL计算(内存优化) / Chunked P_LL Computation (Memory Optimization) =====
        # Atom配对特征处理 - 支持标准模式和分块模式
        if self.use_chunked_pll:
            # 初始化分块嵌入器并共享已训练的MLP! / Initialize chunked embedders and share trained MLPs!
            self.chunked_pairwise_embedder = ChunkedPairwiseEmbedder(
                c_atompair=c_atompair,
                motif_pos_embedder=ChunkedSinusoidalDistEmbed(c_atompair=c_atompair),
                ref_pos_embedder=ChunkedPositionPairDistEmbedder(
                    c_atompair, embed_frame=False
                ),
                process_single_l=self.process_single_l,  # 共享训练参数! / Share trained parameters!
                process_single_m=self.process_single_m,  # 共享训练参数!
                process_z=self.process_z,  # 共享训练参数!
                pair_mlp=self.pair_mlp,  # 共享训练参数!
            )
        # 池化P_LL到token级别 / Pool P_LL to token level
        self.process_pll = linearNoBias(c_atompair, c_atompair)
        self.project_pll = linearNoBias(c_atompair, c_z)

        # ===== 可选的Atom Transformer / Optional Atom Transformer =====
        # 使用序列局部注意力混合atom条件特征
        # Mix atom conditioning features via sequence-local attention
        if atom_transformer["n_blocks"] > 0:
            self.atom_transformer = LocalAtomTransformer(
                c_atom=c_atom, c_s=None, c_atompair=c_atompair, **atom_transformer
            )
        else:
            self.atom_transformer = None

        # Post-processing
        # self.process_s_post = nn.Sequential(
        #     RMSNorm(c_s),
        #     linearNoBias(c_s, c_s),
        # )
        # self.process_z_post = nn.Sequential(
        #     RMSNorm(c_z),
        #     linearNoBias(c_z, c_z),
        # )

    def forward(self, f):
        """
        生成初始atom和token表示 (Algorithm 3 & 4)
        Generate initial atom and token representations

        给定输入特征字典,生成token级和atom级的初始化表示。
        Given input feature dictionary, generate initial token-level and atom-level representations.

        参数 / Args:
            f: 特征字典,包含:
               - atom_to_token_map: [L] atom到token的映射
               - restype: [I] 残基类型
               - ref_pos: [L, 3] 参考坐标
               - motif_pos: [L, 3] Motif坐标
               - 其他1D特征...

        返回 / Returns:
            包含以下键的字典:
            - Q_L_init: [L, c_atom] 初始atom查询特征
            - C_L: [L, c_atom] Atom条件特征
            - P_LL: [L, L, c_atompair] 或 分块嵌入器 - Atom配对特征
            - S_I: [I, c_s] Token单特征
            - Z_II: [I, I, c_z] Token配对特征
        """
        tok_idx = f["atom_to_token_map"]  # [L] atom到token的映射
        L = len(tok_idx)  # 原子总数 / Total number of atoms
        f["ref_atom_name_chars"] = f["ref_atom_name_chars"].reshape(L, -1)
        I = len(f["restype"])  # Token总数 / Total number of tokens

        def init_tokens():
            """
            Algorithm 3: Token初始化器 / Token initializer
            生成初始的token级单特征(S_I)和配对特征(Z_II)
            """
            # ===== 步骤1-4: 嵌入1D特征 / Step 1-4: Embed 1D features =====
            # Algorithm 15: 嵌入token级1D特征(残基类型等)
            S_I = self.token_1d_embedder(f, I)  # [I, c_s]
            # 步骤5: Transition层混合特征
            S_I = S_I + self.transition_post_token(S_I)

            # 嵌入atom级1D特征并下采样到token级
            # Algorithm 9: Downcast - 从atom池化到token
            S_I = self.downcast_atom(
                Q_L=self.atom_1d_embedder_1(f, L), A_I=S_I, tok_idx=tok_idx
            )
            S_I = S_I + self.transition_post_atom(S_I)
            S_I = self.process_s_init(S_I)  # [I, c_s]

            # ===== 步骤6-8: 初始化配对特征Z_II / Step 6-8: Initialize pair features Z_II =====
            # 步骤6-7: 从单特征生成配对特征 (outer sum: S_I_i + S_I_j)
            Z_init_II = self.to_z_init_i(S_I).unsqueeze(-3) + self.to_z_init_j(
                S_I
            ).unsqueeze(-2)  # [I, I, c_z]
            # 步骤8: 添加相对位置编码
            Z_init_II = Z_init_II + self.relative_position_encoding(f)
            # 添加token间的化学键信息
            Z_init_II = Z_init_II + self.process_token_bonds(
                f["token_bonds"].unsqueeze(-1).float()
            )

            # ===== 步骤9: 嵌入配体的参考坐标 / Step 9: Embed reference coordinates of ligands =====
            # Algorithm 14: PositionPairDistEmbedder
            token_id = f["ref_space_uid"][f["is_ca"]]  # C-alpha的token ID
            # 创建mask:仅对同一token内的原子对计算距离
            valid_mask = (token_id.unsqueeze(-1) == token_id.unsqueeze(-2)).unsqueeze(
                -1
            )
            Z_init_II = Z_init_II + self.ref_pos_embedder_tok(
                f["ref_pos"][f["is_ca"]], valid_mask
            )

            # ===== 步骤10-12: Pairformer transformer栈 / Step 10-12: Pairformer transformer stack =====
            # Algorithm 7: TransformerBlock - 使用全注意力处理配对特征
            for block in self.transformer_stack:
                S_I, Z_init_II = block(S_I, Z_init_II)

            # ===== 步骤13-19: 配对特征后处理 / Step 13-19: Post-process pair features =====
            # 拼接第二个相对位置编码并混合
            Z_init_II = torch.cat(
                [
                    Z_init_II,
                    self.relative_position_encoding2(f),
                ],
                dim=-1,
            )  # [I, I, c_z * 2]
            Z_init_II = self.process_z_init(Z_init_II)  # [I, I, c_z]
            # 两个Transition层进一步混合
            for b in range(2):
                Z_init_II = Z_init_II + self.transition_1[b](Z_init_II)

            return {"S_init_I": S_I, "Z_init_II": Z_init_II}

        @activation_checkpointing
        def init_atoms(S_init_I, Z_init_II):
            """
            Algorithm 4: Atom初始化器 / Atom initializer
            生成atom级特征: Q_L_init, C_L, P_LL
            """
            # ===== 步骤1: 嵌入atom级1D特征 / Step 1: Embed atom-level 1D features =====
            # Algorithm 15: OneDFeatureEmbedder for atom features
            Q_L_init = self.atom_1d_embedder_2(f, L)  # [L, c_atom]

            # ===== 步骤2: 从token特征投影 / Step 2: Project from token features =====
            C_L = Q_L_init + self.process_s_trunk(S_init_I)[..., tok_idx, :]  # [L, c_atom]

            if self.use_chunked_pll:
                # ===== 分块模式:返回嵌入器供后续稀疏计算 / Chunked mode: return embedder for later sparse computation =====
                return {
                    "Q_L_init": Q_L_init,
                    "C_L": C_L,
                    "chunked_pairwise_embedder": self.chunked_pairwise_embedder,
                    "S_I": S_init_I,
                    "Z_II": Z_init_II,
                }
            else:
                # ===== 标准模式:完整P_LL计算 / Standard mode: full P_LL computation =====

                # ===== 步骤3: 嵌入Motif坐标 / Step 3: Embed motif coordinates =====
                # Algorithm 13: SinusoidalDistEmbed - 正弦距离嵌入
                # 仅对固定坐标的motif原子对计算距离
                valid_mask = (
                    f["is_motif_atom_with_fixed_coord"].unsqueeze(-1)
                    & f["is_motif_atom_with_fixed_coord"].unsqueeze(-2)
                ).unsqueeze(-1)  # [L, L, 1]
                P_LL = self.motif_pos_embedder(
                    f["motif_pos"], valid_mask
                )  # [L, L, c_atompair]

                # ===== 步骤4: 嵌入参考位置 / Step 4: Embed reference positions =====
                # Algorithm 14: PositionPairDistEmbedder
                # 仅对同一token内的原子对计算距离
                atoms_in_same_token = (
                    f["ref_space_uid"].unsqueeze(-1) == f["ref_space_uid"].unsqueeze(-2)
                ).unsqueeze(-1)
                # 仅对给定序列的原子考虑ref_pos (否则ref_pos为0,计算无意义)
                atoms_has_seq = (
                    f["is_motif_atom_with_fixed_seq"].unsqueeze(-1)
                    & f["is_motif_atom_with_fixed_seq"].unsqueeze(-2)
                ).unsqueeze(-1)
                valid_mask = atoms_in_same_token & atoms_has_seq
                P_LL = P_LL + self.ref_pos_embedder(f["ref_pos"], valid_mask)

                # ===== 步骤5-7: Atom配对特征的MLP处理 / Step 5-7: MLP processing for atom pairwise features =====
                # 步骤5: 添加单atom特征的外积 (outer sum: C_L_l + C_L_m)
                P_LL = P_LL + (
                    self.process_single_l(C_L).unsqueeze(-2)
                    + self.process_single_m(C_L).unsqueeze(-3)
                )
                # 步骤6: 添加从token配对特征投影的信息
                # 将Z_II [I, I, c_z] 通过tok_idx索引到atom级 [L, L, c_z]
                P_LL = (
                    P_LL
                    + self.process_z(Z_init_II)[..., tok_idx, :, :][..., tok_idx, :]
                )
                # 步骤7: 深度MLP混合所有配对特征
                P_LL = P_LL + self.pair_mlp(P_LL)
                P_LL = P_LL.contiguous()  # [L, L, c_atompair]

                # ===== 池化P_LL回token级以提供atom级分辨率 / Pool P_LL to token level =====
                # 将atom配对特征池化为token配对特征,增强Z_II
                pooled_atom_level_features = pairwise_mean_pool(
                    pairwise_atom_features=self.process_pll(P_LL).unsqueeze(0),
                    atom_to_token_map=tok_idx,
                    I=int(tok_idx.max().item()) + 1,
                    dtype=P_LL.dtype,
                ).squeeze(0)  # [I, I, c_atompair]
                Z_init_II = Z_init_II + self.project_pll(pooled_atom_level_features)

                # ===== 可选: Atom transformer混合条件特征 / Optional: Atom transformer =====
                # 使用序列局部注意力进一步混合atom特征
                if exists(self.atom_transformer):
                    C_L = self.atom_transformer(
                        C_L.unsqueeze(0), None, P_LL, indices=None, f=f, X_L=None
                    ).squeeze(0)

                return {
                    "Q_L_init": Q_L_init,  # [L, c_atom] 初始atom查询特征
                    "C_L": C_L,  # [L, c_atom] Atom条件特征
                    "P_LL": P_LL,  # [L, L, c_atompair] Atom配对特征
                    "S_I": S_init_I,  # [I, c_s] Token单特征
                    "Z_II": Z_init_II,  # [I, I, c_z] Token配对特征(增强后)
                }

        tokens = init_tokens()
        return init_atoms(**tokens)


class DiffusionTokenEncoder(nn.Module):
    """
    扩散Token编码器 (Algorithm 12: Diffusion token encoder)
    Diffusion Token Encoder

    在每次扩散循环中嵌入噪声尺度和循环distogram。
    Embeds noise scale and recycled distogram at each diffusion cycle.

    主要功能 / Main Functions:
    - 将当前噪声坐标的distogram编码到Z_II中
    - 将前一次循环的self-conditioning distogram编码到Z_II中
    - 通过Pairformer块混合token单特征和配对特征

    参数 / Parameters:
        c_s: Token单轨迹特征维度 / Token single track dimension
        c_z: Token配对特征维度 / Token pair dimension
        sigma_data: EDM数据方差 / EDM data variance
        use_distogram: 是否使用当前distogram / Whether to use current distogram
        use_self: 是否使用self-conditioning distogram / Whether to use self-conditioning distogram
    """
    def __init__(
        self,
        c_s,
        c_z,
        c_token,
        c_atompair,
        sigma_data,
        n_pairformer_blocks,
        pairformer_block,
        use_distogram,
        use_self,
        use_sinusoidal_distogram_embedder=True,
        **_,
    ):
        super().__init__()

        # ===== Algorithm 12 步骤1-3: Token单特征处理 / Step 1-3: Token single feature processing =====
        self.transition_1 = nn.ModuleList(
            [
                Transition(c=c_s, n=2),
                Transition(c=c_s, n=2),
            ]
        )

        # ===== Algorithm 12 步骤4-8: Distogram嵌入和配对特征处理 / Step 4-8: Distogram embedding and pair feature processing =====
        self.n_bins_distogram = 65  # Distogram离散化bin数 (1-30Å) / Number of distogram bins
        n_bins_noise = self.n_bins_distogram
        self.use_self = use_self  # 是否使用self-conditioning distogram
        self.use_distogram = use_distogram  # 是否使用当前噪声distogram
        self.use_sinusoidal_distogram_embedder = use_sinusoidal_distogram_embedder

        # 步骤4: 离散化或嵌入distogram / Bucketize or embed distogram
        if self.use_distogram:
            if self.use_sinusoidal_distogram_embedder:
                # Algorithm 13: 使用正弦嵌入 / Use sinusoidal embedding
                self.dist_embedder = SinusoidalDistEmbed(c_atompair=c_z)
                n_bins_noise = c_z
            else:
                # 使用离散化 / Use bucketization
                self.bucketize_fn = functools.partial(
                    bucketize_scaled_distogram,
                    min_dist=1,   # 最小距离1Å
                    max_dist=30,  # 最大距离30Å
                    sigma_data=sigma_data,
                    n_bins=self.n_bins_distogram,
                )

        # 计算拼接后的配对特征维度 / Calculate concatenated pair feature dimension
        # Z_II + distogram + self_distogram
        cat_c_z = (
            c_z
            + int(self.use_distogram) * n_bins_noise  # 当前distogram
            + int(self.use_self) * self.n_bins_distogram  # Self-conditioning distogram
        )
        # 步骤5-8: 混合拼接的配对特征 / Mix concatenated pair features
        self.process_z = nn.Sequential(
            RMSNorm(cat_c_z),
            linearNoBias(cat_c_z, c_z),
        )

        self.transition_2 = nn.ModuleList(
            [
                Transition(c=c_z, n=2),
                Transition(c=c_z, n=2),
            ]
        )

        # ===== Algorithm 12 步骤9-11: Pairformer块 / Step 9-11: Pairformer blocks =====
        # Algorithm 7: TransformerBlock - 混合单特征和配对特征
        self.pairformer_stack = nn.ModuleList(
            [
                PairformerBlock(c_s=c_s, c_z=c_z, **pairformer_block)
                for _ in range(n_pairformer_blocks)
            ]
        )

    def forward(self, f, R_L, S_init_I, Z_init_II, C_L, P_LL, **kwargs):
        """
        扩散Token编码器前向传播 (Algorithm 12)
        Diffusion token encoder forward pass

        嵌入噪声尺度和循环distogram到token配对特征中。
        Embeds noise scale and recycled distogram into token pair features.

        参数 / Args:
            f: 特征字典 / Feature dictionary
            R_L: [B, L, 3] 当前噪声坐标 / Current noisy coordinates
            S_init_I: [I, c_s] 初始token单特征 / Initial token single features
            Z_init_II: [I, I, c_z] 初始token配对特征 / Initial token pair features
            C_L: [B, L, c_atom] Atom条件特征 / Atom conditioning features
            P_LL: [L, L, c_atompair] Atom配对特征 / Atom pairwise features
            **kwargs: 包含D_II_self (前一次循环的distogram) / Contains D_II_self (previous cycle's distogram)

        返回 / Returns:
            S_I: [B, I, c_s] 更新后的token单特征 / Updated token single features
            Z_II: [B, I, I, c_z] 更新后的token配对特征 / Updated token pair features
        """
        B = R_L.shape[0]

        @activation_checkpointing
        def token_embed(S_init_I, Z_init_II):
            """
            Algorithm 12的核心实现 / Core implementation of Algorithm 12
            """
            # ===== 步骤1-3: 处理token单特征 / Step 1-3: Process token single features =====
            S_I = S_init_I
            for b in range(2):
                S_I = S_I + self.transition_1[b](S_I)

            # ===== 步骤4-8: 准备配对特征 / Step 4-8: Prepare pair features =====
            # 扩展到batch维度 / Expand to batch dimension
            Z_II = Z_init_II.unsqueeze(0).expand(B, -1, -1, -1)  # [B, I, I, c_z]

            # 收集要拼接的配对特征 / Collect pair features to concatenate
            Z_II_list = [Z_II]

            # 步骤4: 嵌入当前噪声坐标的distogram / Step 4: Embed current noisy coordinate distogram
            if self.use_distogram:
                if self.use_sinusoidal_distogram_embedder:
                    # Algorithm 13: 正弦距离嵌入 / Sinusoidal distance embedding
                    mask = f["is_motif_atom_with_fixed_coord"][f["is_ca"]]
                    # 移除对角线外不同时间的距离(无意义)
                    # Remove off-diagonal distances across time (meaningless)
                    mask = (mask[None, :] != mask[:, None]).unsqueeze(-1)
                    D_LL = self.dist_embedder(R_L[..., f["is_ca"], :], ~mask)
                else:
                    # 离散化distogram / Bucketize distogram
                    D_LL = self.bucketize_fn(
                        R_L[..., f["is_ca"], :]
                    )  # [B, I, I, n_bins]
                Z_II_list.append(D_LL)

            # 步骤4: 添加self-conditioning distogram (前一次循环的输出)
            # Add self-conditioning distogram (previous cycle's output)
            if self.use_self:
                D_II_self = kwargs.get("D_II_self")
                if D_II_self is None:
                    # 第一次循环,使用零初始化 / First cycle, use zeros
                    D_II_self = torch.zeros(
                        Z_II.shape[:-1] + (self.n_bins_distogram,),
                        device=Z_II.device,
                        dtype=Z_II.dtype,
                    )
                Z_II_list.append(D_II_self)

            # 步骤5-8: 拼接并混合配对特征 / Step 5-8: Concatenate and mix pair features
            Z_II = torch.cat(Z_II_list, dim=-1)  # [B, I, I, c_z + n_bins + n_bins]

            # 投影回c_z维度 / Project back to c_z dimension
            Z_II = self.process_z(Z_II)  # [B, I, I, c_z]

            # 两个Transition层进一步混合 / Two Transition layers for further mixing
            for b in range(2):
                Z_II = Z_II + self.transition_2[b](Z_II)

            # ===== 步骤9-11: Pairformer混合单特征和配对特征 / Step 9-11: Pairformer to mix single and pair features =====
            # Algorithm 7: TransformerBlock
            for block in self.pairformer_stack:
                S_I, Z_II = block(S_I, Z_II)

            return S_I, Z_II

        return token_embed(S_init_I, Z_init_II)
