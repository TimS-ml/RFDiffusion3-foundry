# RFdiffusion3 SI算法实现指南

本文档提供了论文补充材料(SI)第2节 "RFdiffusion3 Architecture and Inference" 中所有算法的代码实现位置映射。

## 目录结构

主要代码位于 `./models/rfd3/src/rfd3/model/` 目录下:
- `RFD3.py` - 主模型类
- `RFD3_diffusion_module.py` - 扩散模块
- `inference_sampler.py` - 推理采样器
- `layers/` - 各种网络层实现
  - `encoders.py` - Token和Atom初始化器
  - `blocks.py` - Transformer块和其他构建模块
  - `attention.py` - 注意力机制
  - `pairformer_layers.py` - Pairformer层
  - `layer_utils.py` - 基础层工具

---

## 算法映射

### Section 2.1: Main Inference Loop

#### **Algorithm 1: Inference diffusion loop** (SI p.56)

**实现位置:**
- 文件: `./models/rfd3/src/rfd3/model/inference_sampler.py`
- 类: `ConditionalDiffusionSampler`
- 方法: `sample_diffusion_like_af3()`
- 关键代码段: 行 ~200-300

**算法步骤说明:**
1. **Token and atom embedding (步骤1-2)**: 在 `RFD3.py` 的 `forward()` 方法中调用 `token_initializer`
2. **Initialize structure (步骤3-11)**: 在 `inference_sampler.py` 的主循环中
3. **Modulation of ODE/SDE (步骤4-5)**: 噪声调制参数计算
4. **Noise injection (步骤6-7)**: 向扩散坐标添加噪声
5. **DiffusionModule (步骤8)**: 调用 `RFD3DiffusionModule`
6. **Coordinate update (步骤9-10)**: 更新原子坐标

---

### Section 2.2: Inference Loop for Symmetric Design

#### **Algorithm 2: Symmetric inference diffusion loop** (SI p.57)

**实现位置:**
- 文件: `./models/rfd3/src/rfd3/inference/symmetry/symmetry_utils.py`
- 函数: `apply_symmetry_to_xyz_atomwise()`
- 辅助文件: `./models/rfd3/src/rfd3/model/inference_sampler.py`
- 配置: `SampleDiffusionConfig.kind = "symmetry"`

**关键特性:**
- 对称性应用 (步骤9-10): 使用旋转矩阵 `R={Rn}` 对非对称单元进行对称化
- 可配置的对称停止比例 `σsym` (默认0.9)

---

### Section 2.3: Token Initializers

#### **Algorithm 3: Token initializer** (SI p.59)

**实现位置:**
- 文件: `./models/rfd3/src/rfd3/model/layers/encoders.py`
- 类: `TokenInitializer`
- 方法: `forward()`

**关键组件:**
- **步骤1-4**: `OneDFeatureEmbedder` 用于1D特征嵌入
- **步骤5**: `Transition` 层
- **步骤6-8**: 配对特征初始化 (`to_z_init_i`, `to_z_init_j`, `relative_position_encoding`)
- **步骤9**: `PositionPairDistEmbedder` 用于距离编码
- **步骤10-12**: `PairformerBlock` 在 `transformer_stack` 中
- **步骤13-19**: 配对特征后处理

#### **Algorithm 4: Atom initializer** (SI p.60)

**实现位置:**
- 文件: `./models/rfd3/src/rfd3/model/layers/encoders.py`
- 类: `TokenInitializer` (包含atom初始化逻辑)
- 方法: `forward()` 的后半部分

**关键组件:**
- **步骤1**: `atom_1d_embedder_2` - Atom级1D特征嵌入
- **步骤2**: 从token特征投影: `process_s_trunk`
- **步骤3**: `SinusoidalDistEmbed` - Motif位置嵌入
- **步骤4**: `PositionPairDistEmbedder` - 参考位置嵌入
- **步骤5-7**: Atom配对特征的MLP处理: `process_single_l`, `process_single_m`, `process_z`, `pair_mlp`

---

### Section 2.4: DiffusionModule

#### **Algorithm 5: Diffusion forward pass with recycling** (SI p.61)

**实现位置:**
- 文件: `./models/rfd3/src/rfd3/model/RFD3_diffusion_module.py`
- 类: `RFD3DiffusionModule`
- 方法: `forward()`

**关键步骤:**
- **步骤1**: 坐标缩放: `process_r`
- **步骤2-3**: `Downcast` 池化到序列级别
- **步骤4-7**: 时间/批次嵌入: `fourier_embedding`, `process_n`
- **步骤8**: Local atom transformer: `encoder` (LocalAtomTransformer)
- **步骤9**: Downcast: `downcast_q`
- **步骤10-17**: 循环处理 (nrecycle=2):
  - **步骤12**: `DiffusionTokenEncoder` - 嵌入噪声尺度和循环distogram
  - **步骤13**: `transformer` (LocalTokenTransformer) - Token级稀疏注意力
  - **步骤14**: `decoder` (CompactStreamingDecoder) - 上投影并解码为结构
  - **步骤15-16**: 坐标更新: `to_r_update`

---

### Section 2.4.1 & 2.4.2: Transformers and Attention

#### **Algorithm 6: Local token transformer** (SI p.64)

**实现位置:**
- 文件: `./models/rfd3/src/rfd3/model/layers/blocks.py`
- 类: `LocalTokenTransformer`
- 方法: `forward()`

**关键特性:**
- **步骤1**: `create_attention_indices()` - 创建SL2稀疏注意力索引
- **步骤2-8**: 循环通过多个transformer块
- **步骤4**: `Upcast` - 如果提供了cskip
- **步骤6**: `SparseAttentionPairBias` - 带配对偏置的稀疏注意力
- **步骤7**: `ConditionedTransitionBlock` - 条件化的transition块

#### **Algorithm 7: TransformerBlock** (SI p.64)

**实现位置:**
- 文件: `./models/rfd3/src/rfd3/model/layers/pairformer_layers.py`
- 类: `PairformerBlock`
- 方法: `forward()`

**特性:**
- 全注意力版本的transformer
- 用于处理配对特征到单轨迹

#### **Algorithm 8: Sparse attention with pair bias** (SI p.65)

**实现位置:**
- 文件: `./models/rfd3/src/rfd3/model/layers/attention.py`
- 类: `LocalAttentionPairBias`
- 方法: `forward()`

**关键步骤:**
- **步骤1-5**: AdaLN或RMSNorm标准化
- **步骤6-9**: Q, K, V, gate投影
- **步骤10**: `sparse_pairbias_attention()` - 带配对偏置的稀疏注意力
- **步骤12-14**: 可选的门控与条件信号

---

### Section 2.5: Cross Attention Pooling

#### **Algorithm 9: Downcast** (SI p.66)

**实现位置:**
- 文件: `./models/rfd3/src/rfd3/model/layers/blocks.py`
- 类: `Downcast`
- 方法: `forward()`

**关键操作:**
- **步骤1**: `group_atoms()` - 按token ID分组原子
- **步骤2**: `GatedCrossAttention` - Q=ai, KV=qia
- **步骤4**: 添加token特征 (si)

#### **Algorithm 10: Upcast** (SI p.66)

**实现位置:**
- 文件: `./models/rfd3/src/rfd3/model/layers/blocks.py`
- 类: `Upcast` (在 `CompactStreamingDecoder` 中使用)
- 相关代码在多个transformer块中

**关键操作:**
- **步骤1**: `reshape()` - 分割token并按token ID分组原子
- **步骤3**: `GatedCrossAttention` - Q=qia, KV=a_split

#### **Algorithm 11: Gated cross attention** (SI p.67)

**实现位置:**
- 文件: `./models/rfd3/src/rfd3/model/layers/attention.py`
- 类: `GatedCrossAttention`
- 方法: `forward()`

**关键步骤:**
- **步骤1-2**: RMSNorm标准化
- **步骤3-6**: Q, K, V, gate投影
- **步骤7-13**: Scaled dot-product attention with valid masking
- **步骤14**: 合并头部并投影

---

### Section 2.6: Embedders

#### **Algorithm 12: Diffusion token encoder** (SI p.68)

**实现位置:**
- 文件: `./models/rfd3/src/rfd3/model/layers/encoders.py`
- 类: `DiffusionTokenEncoder`
- 方法: `forward()`

**关键步骤:**
- **步骤1-3**: Transition层处理
- **步骤4**: `bucketize_scaled_distogram()` - 将noisy和self坐标的distogram离散化
- **步骤9-11**: `PairformerBlock` 循环

#### **Algorithm 13: Sinusoidal Distance Embedding** (SI p.69)

**实现位置:**
- 文件: `./models/rfd3/src/rfd3/model/layers/blocks.py`
- 类: `SinusoidalDistEmbed`
- 方法: `forward()`

**关键步骤:**
- **步骤1**: 计算配对距离
- **步骤2-4**: 正弦嵌入: `ω_k`, `θ_lmk`, `e_lm = [sin(θ) || cos(θ)]`
- **步骤5-7**: 应用并嵌入mask

#### **Algorithm 14: Position Pair Distance Embedding** (SI p.69)

**实现位置:**
- 文件: `./models/rfd3/src/rfd3/model/layers/blocks.py`
- 类: `PositionPairDistEmbedder`
- 方法: `forward()`

**关键步骤:**
- **步骤1**: 计算配对距离
- **步骤2**: 编码逆配对距离: `1/(1 + ||d_lm||^2)`
- **步骤3-4**: 嵌入mask

#### **Algorithm 15: One-dimension Feature Embedder** (SI p.69)

**实现位置:**
- 文件: `./models/rfd3/src/rfd3/model/layers/blocks.py`
- 类: `OneDFeatureEmbedder`
- 方法: `forward()`

**特性:**
- 对token级和atom级特征使用相同的实现
- 对每个1D特征嵌入并求和结果

---

### Section 2.7: Classifier Free Guidance

#### **Algorithm 16: FourierEmbedding** (SI p.70)

**实现位置:**
- 文件: `./src/foundry/model/layers/blocks.py` (foundry包)
- 类: `FourierEmbedding`

**用法:** 在 `RFD3_diffusion_module.py` 中的时间嵌入

#### **Algorithm 17: Processing of noise conditioning features** (SI p.70)

**实现位置:**
- 文件: `./models/rfd3/src/rfd3/model/RFD3_diffusion_module.py`
- 在 `forward()` 方法中的时间处理部分

**关键步骤:**
- **步骤1**: `fourier_embedding` - Fourier特征
- **步骤2**: 乘以时间mask `(tl > 0)`

**分类器自由引导实现:**
- 文件: `./models/rfd3/src/rfd3/model/cfg_utils.py`
- 函数: `strip_f()`, `strip_X()`
- 配置: `use_classifier_free_guidance=True`, `cfg_scale`

---

## 其他重要文件

### 训练相关
- `./models/rfd3/src/rfd3/trainer/` - 训练器实现
- `./models/rfd3/src/rfd3/transforms/` - 数据转换和条件化

### 推理相关
- `./models/rfd3/src/rfd3/run_inference.py` - 推理脚本
- `./models/rfd3/src/rfd3/utils/inference.py` - 推理工具函数

### 对称性支持
- `./models/rfd3/src/rfd3/inference/symmetry/` - 对称性推理
- `./models/rfd3/src/rfd3/transforms/symmetry.py` - 对称性转换

### 配置
- `./models/rfd3/configs/` - Hydra配置文件

---

## 注意事项

1. **坐标表示**: 所有坐标使用atom级别表示 `[B, L, 3]`, 其中L是原子数
2. **Token vs Atom**:
   - Token (I): 残基或小分子作为单个token
   - Atom (L): 每个原子独立表示
3. **稀疏注意力 (SL2)**:
   - 序列局部: `n_attn_seq_neighbours = 32`
   - 结构局部: `n_attn_keys = 128`
4. **循环 (Recycling)**: 默认 `n_recycle = 2`
5. **低内存模式**: 设置 `RFD3_LOW_MEMORY_MODE=1` 使用分块的配对特征处理

---

## 快速索引

| 算法 | 主要实现类/函数 | 文件路径 |
|------|----------------|----------|
| Algorithm 1 | `ConditionalDiffusionSampler.sample_diffusion_like_af3()` | `inference_sampler.py` |
| Algorithm 2 | `apply_symmetry_to_xyz_atomwise()` | `inference/symmetry/symmetry_utils.py` |
| Algorithm 3 | `TokenInitializer` | `layers/encoders.py` |
| Algorithm 4 | `TokenInitializer` (atom部分) | `layers/encoders.py` |
| Algorithm 5 | `RFD3DiffusionModule.forward()` | `RFD3_diffusion_module.py` |
| Algorithm 6 | `LocalTokenTransformer` | `layers/blocks.py` |
| Algorithm 7 | `PairformerBlock` | `layers/pairformer_layers.py` |
| Algorithm 8 | `LocalAttentionPairBias` | `layers/attention.py` |
| Algorithm 9 | `Downcast` | `layers/blocks.py` |
| Algorithm 10 | `Upcast` in `CompactStreamingDecoder` | `layers/blocks.py` |
| Algorithm 11 | `GatedCrossAttention` | `layers/attention.py` |
| Algorithm 12 | `DiffusionTokenEncoder` | `layers/encoders.py` |
| Algorithm 13 | `SinusoidalDistEmbed` | `layers/blocks.py` |
| Algorithm 14 | `PositionPairDistEmbedder` | `layers/blocks.py` |
| Algorithm 15 | `OneDFeatureEmbedder` | `layers/blocks.py` |
| Algorithm 16 | `FourierEmbedding` | `foundry/model/layers/blocks.py` |
| Algorithm 17 | Time processing in `RFD3DiffusionModule` | `RFD3_diffusion_module.py` |

---

**生成日期**: 2025-12-23
**基于论文**: RFdiffusion3 Supplementary Information, Section 2
