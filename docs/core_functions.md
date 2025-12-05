# RFDiffusion3 Foundry - Core Functions Documentation

This document provides technical documentation for the core functions and components in the RFDiffusion3 Foundry framework.

## Table of Contents

1. [Overview](#overview)
2. [Foundry Framework Core](#foundry-framework-core)
3. [RFD3 Model Architecture](#rfd3-model-architecture)
4. [Transform Pipeline](#transform-pipeline)
5. [Utility Functions](#utility-functions)

---

## Overview

The RFDiffusion3 Foundry is a modular ML framework for protein design using diffusion models. It consists of:

- **Foundry Framework**: Shared infrastructure for training and inference
- **RFD3 Model**: Diffusion-based generative protein design model
- **RF3 Model**: Structure prediction model (AlphaFold3-like)
- **MPNN Model**: Inverse folding/sequence design model

This document focuses on the foundry framework and RFD3 core functions.

---

## Foundry Framework Core

The foundry framework (`/src/foundry/`) provides shared utilities used across all models.

### Common Utilities (`foundry.common`)

**Location**: `/src/foundry/common.py`

#### Helper Functions

```python
def exists(obj: Any) -> bool
```
Returns `True` if object is not None.

```python
def default(obj: Any, default: Any) -> Any
```
Returns `obj` if it exists, otherwise returns `default`.

```python
def concat_dicts(*dicts: dict) -> dict
```
Concatenates dictionaries with the same keys into a single dict with list values.

**Example**:
```python
>>> d1 = {"a": 1, "b": 2}
>>> d2 = {"a": 3, "b": 4}
>>> concat_dicts(d1, d2)
{'a': [1, 3], 'b': [2, 4]}
```

```python
def listmap(fn: Callable, lst: Iterable[Any]) -> list
```
Applies a function to each element of a list.

```python
def ensure_dtype(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor
```
Converts tensor to target dtype if needed.

### Alignment Utilities (`foundry.utils.alignment`)

**Location**: `/src/foundry/utils/alignment.py`

#### Weighted Rigid Alignment

```python
def weighted_rigid_align(
    X_L: torch.Tensor,        # [B, L, 3] predicted coordinates
    X_gt_L: torch.Tensor,     # [B, L, 3] ground truth coordinates
    X_exists_L: Optional[torch.Tensor] = None,  # [L] existence mask
    w_L: Optional[torch.Tensor] = None,         # [B, L] weights
) -> torch.Tensor  # [B, L, 3] aligned coordinates
```

Performs weighted rigid body alignment of ground truth onto predicted coordinates using Algorithm 28 from AlphaFold3. This enables SE(3)-invariant loss computation.

**Algorithm**:
1. Compute weighted centroids of both coordinate sets
2. Center both coordinate sets
3. Compute covariance matrix
4. Perform SVD to find optimal rotation
5. Apply rotation and translation

**Usage**: Used in training to align predicted structures to ground truth for loss calculation.

#### RMSD Calculation

```python
def get_rmsd(xyz1: torch.Tensor, xyz2: torch.Tensor, eps: float = 1e-4) -> torch.Tensor
```

Computes root mean square deviation between two coordinate sets.

#### Superposition

```python
def superimpose(
    xyz1: torch.Tensor,
    xyz2: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-4
) -> torch.Tensor
```

Superimposes xyz1 onto xyz2 using a boolean mask.

### Inference Engine Base (`foundry.inference_engines.base`)

**Location**: `/src/foundry/inference_engines/base.py`

#### BaseInferenceEngine

Base class for all model inference engines. Provides:

- **Checkpoint loading**: Automatic model weight loading
- **Transform pipeline setup**: Configuration of data preprocessing
- **Device management**: GPU/CPU handling
- **Distributed inference**: Multi-GPU support

**Key Methods**:
- `initialize()`: Loads model and sets up pipeline
- `run()`: Main inference entry point

---

## RFD3 Model Architecture

### Main Model Wrapper (`rfd3.model.RFD3`)

**Location**: `/models/rfd3/src/rfd3/model/RFD3.py`

#### RFD3 Class

```python
class RFD3(nn.Module):
    def __init__(
        self,
        *,
        c_s: int,              # Token embedding dimension
        c_z: int,              # Token pair embedding dimension
        c_atom: int,           # Atom embedding dimension
        c_atompair: int,       # Atom pair embedding dimension
        token_initializer: DictConfig | dict,
        diffusion_module: DictConfig | dict,
        inference_sampler: DictConfig | dict,
        **_: Any,
    ) -> None
```

Main model wrapper that orchestrates:
1. **Token initialization**: Converts input features to embeddings
2. **Diffusion module**: Core denoising network
3. **Inference sampler**: Handles multi-step diffusion rollout

**Components**:

- `token_initializer`: `TokenInitializer` - Creates initial token embeddings from input features
- `diffusion_module`: `RFD3DiffusionModule` - Main denoising network
- `inference_sampler`: `ConditionalDiffusionSampler` - Manages sampling process

**Forward Pass**:

```python
def forward(
    self,
    input: Dict[str, Any],
    coord_atom_lvl_to_be_noised: Optional[torch.Tensor] = None,
    n_cycle: Optional[int] = None,
    **_: Any,
) -> Dict[str, Any]
```

**Training mode**: Single denoising step at random timestep
**Inference mode**: Full diffusion rollout via `inference_sampler`

**Classifier-Free Guidance**: If enabled, creates stripped reference features for unconditional guidance.

### Diffusion Module (`rfd3.model.RFD3_diffusion_module`)

**Location**: `/models/rfd3/src/rfd3/model/RFD3_diffusion_module.py`

#### RFD3DiffusionModule Class

Core diffusion denoising network with recycling and attention mechanisms.

**Architecture**:
```
Input Positions (noisy)
    ↓
Position Scaling (EDM-based)
    ↓
Local Atom Attention Encoder
    ↓
Token Pooling
    ↓
Diffusion Token Encoder
    ↓
Token-Level Transformer
    ↓
Atom-Level Decoder
    ↓
Position Update
    ↓
Output Positions (denoised)
```

**Key Methods**:

##### Position Scaling

```python
def scale_positions_in(self, X_noisy_L: torch.Tensor, t: torch.Tensor) -> torch.Tensor
```

Scales input positions based on noise level `t` using EDM preconditioning:
- `edm`: \( R = X / \sqrt{t^2 + \sigma_{data}^2} \)
- `unconditioned`: \( R = 0 \)
- `noise_pred`: \( R = X \) (no scaling)

```python
def scale_positions_out(
    self, R_update_L: torch.Tensor, X_noisy_L: torch.Tensor, t: torch.Tensor
) -> torch.Tensor
```

Converts model output back to position space using inverse scaling.

##### Time Embedding

```python
def process_time_(self, t_L: torch.Tensor, i: int) -> torch.Tensor
```

Converts timestep to Fourier features:
\[ \text{feature} = \text{FFN}(\text{Fourier}(\frac{1}{4} \log(t / \sigma_{data}))) \]

Masked to zero where `t = 0` (motif atoms).

##### Forward with Recycling

```python
def forward_with_recycle(
    self,
    n_recycle: Optional[int],
    **kwargs: Any,
) -> Dict[str, torch.Tensor]
```

Performs multiple recycling iterations:
- **Training**: Random number of recycles (0 to n_recycle-1)
- **Inference**: Fixed `n_recycle` iterations
- Uses gradient checkpointing for memory efficiency
- Only last iteration computes gradients

**Outputs**:
- `X_L`: Denoised atom positions `[B, L, 3]`
- `sequence_indices_I`: Predicted sequence indices `[B, I]`
- `sequence_logits_I`: Sequence prediction logits `[B, I, num_residues]`

### Inference Sampler (`rfd3.model.inference_sampler`)

**Location**: `/models/rfd3/src/rfd3/model/inference_sampler.py`

#### SampleDiffusionConfig

Configuration dataclass for diffusion sampling:

```python
@dataclass(kw_only=True)
class SampleDiffusionConfig:
    # Sampling schedule
    num_timesteps: int = 200
    min_t: int = 0
    max_t: int = 1
    sigma_data: int = 16
    s_min: float = 4e-4
    s_max: int = 160
    p: int = 7

    # Noise schedule parameters
    gamma_0: float = 0.6
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    step_scale: float = 1.5

    # Design-specific
    center_option: str = "all"
    s_trans: float = 1.0
    s_jitter_origin: float = 0.0
    fraction_of_steps_to_fix_motif: float = 0.0

    # Classifier-free guidance
    use_classifier_free_guidance: bool = False
    cfg_scale: float = 2.0
    cfg_t_max: Optional[float] = None
```

#### Noise Schedule Construction

```python
def _construct_inference_noise_schedule(
    self, device: torch.device, partial_t: Optional[float] = None
) -> torch.Tensor
```

Creates noise schedule following AF3 formulation:

\[ t_{hat} = \sigma_{data} \left( s_{max}^{1/p} + t \cdot (s_{min}^{1/p} - s_{max}^{1/p}) \right)^p \]

Where \( t \in [0, 1] \) is linearly spaced.

**Partial Diffusion**: If `partial_t` is specified, only uses timesteps \( \leq partial_t \).

---

## Transform Pipeline

Transforms are modular preprocessing steps applied to input data before model inference.

### Design Transforms (`rfd3.transforms.design_transforms`)

**Location**: `/models/rfd3/src/rfd3/transforms/design_transforms.py`

#### SubsampleToTypes

```python
class SubsampleToTypes(Transform):
    def __init__(self, allowed_types: list[str] | str = ["is_protein"]) -> None
```

Filters atom array to include only specified molecular types (protein, DNA, RNA, ligand).

#### CreateDesignReferenceFeatures

```python
class CreateDesignReferenceFeatures(Transform):
    def __init__(
        self,
        generate_conformers: bool,
        provide_reference_conformer_when_unmasked: bool,
        ground_truth_conformer_policy: str,
        **kwargs: Any,
    ) -> None
```

Creates reference features for design:
- **Conformer generation**: For ligands and non-standard residues
- **Reference positions**: Ground truth positions for motif atoms
- **Element features**: Atomic numbers for unindexed components
- **Motif encoding**: One-hot encoding of motif types

**Output Features**:
- `ref_atom_name_chars`: Encoded atom names `[n_atoms, 4]`
- `ref_pos`: Reference positions `[n_atoms, 3]`
- `ref_mask`: Validity mask `[n_atoms]`
- `ref_element`: Atomic numbers `[n_atoms]`
- `ref_charge`: Partial charges `[n_atoms]`
- `motif_pos`: Ground truth motif positions `[n_atoms, 3]`
- `ref_motif_token_type`: Motif type encoding `[n_tokens, 3]`

#### AddIsXFeats

```python
class AddIsXFeats(Transform):
    def __init__(
        self,
        X: list[str],
        central_atom: str,
        extra_atom_level_feats: list[str] = [],
        extra_token_level_feats: list[str] = [],
    ) -> None
```

Adds boolean feature masks:
- `is_backbone`: Backbone atoms
- `is_sidechain`: Sidechain atoms
- `is_virtual`: Virtual padding atoms
- `is_central`: Token representative atoms
- `is_ca`: Alpha carbon atoms
- `is_motif_atom_with_fixed_coord`: Fixed motif atoms
- `is_motif_atom_unindexed`: Unindexed (atomized) motif

#### MotifCenterRandomAugmentation

```python
class MotifCenterRandomAugmentation(Transform):
    def __init__(
        self,
        batch_size: int,
        sigma_perturb: float,
        center_option: str,
    ) -> None
```

Training-time augmentation:
1. Centers coordinates on motif or diffused region
2. Adds random translation offset
3. Applies random SO(3) rotation

**Purpose**: Makes model SE(3)-equivariant and robust to global transformations.

#### AugmentNoise

```python
class AugmentNoise(Transform):
    def __init__(self, sigma_perturb_com: float, batch_size: int) -> None
```

Augments noise with time-dependent COM perturbation:
- Adds center-of-mass offset between motif and diffused regions
- Scaled by \( (t/t_{max})^3 \) for smooth annealing
- Zeros out noise for fixed motif atoms

---

## Utility Functions

### Inference Utilities (`rfd3.utils.inference`)

**Location**: `/models/rfd3/src/rfd3/utils/inference.py`

#### File Loading

```python
def inference_load_(
    file: PathLike,
    *,
    assembly_id: str = "1",
    cif_parser_args: dict | None = None
) -> dict
```

Loads structure file (PDB/CIF) for inference:
- Parses assembly
- Extracts chain and ligand information
- Loads conditioning annotations
- Converts boolean annotations

**Returns**:
```python
{
    "atom_array": AtomArray,      # Structure
    "chain_info": dict,            # Chain metadata
    "ligand_info": dict,           # Ligand information
    "metadata": dict,              # File metadata
}
```

#### Coordinate Centering

```python
def set_com(
    atom_array: AtomArray,
    ori_token: Optional[list] = None,
    infer_ori_strategy: Optional[str] = None
) -> AtomArray
```

Centers structure coordinates on specified origin:

**Priority**:
1. `ori_token` (explicit coordinates)
2. ORI residue in input structure
3. `infer_ori_strategy` ("hotspots" or "com")
4. Motif center of mass (if motif exists)
5. Zero (no offset)

**Infer Strategies**:
- `"hotspots"`: Centers 10Å above atom-level hotspots
- `"com"`: Centers on structure center of mass

#### Idealized CB Generation

```python
def generate_idealized_cb_position(
    N: np.array,
    Ca: np.array,
    C: np.array
) -> np.array
```

Generates idealized Cβ coordinates given backbone N, Cα, C atoms:

1. Constructs local frame from backbone
2. Places Cβ at idealized position (-0.529, -0.774, -1.205) relative to Cα
3. Based on ideal alanine geometry

**Use case**: Repairing incomplete backbone structures for motif scaffolding.

---

## Summary

This document covers the core architectural components of RFDiffusion3:

1. **Foundry utilities**: Common helpers, alignment, and inference base
2. **RFD3 model**: Main wrapper, diffusion module, and sampling
3. **Transform pipeline**: Modular preprocessing for design tasks
4. **Utilities**: File I/O, coordinate manipulation, and feature generation

For inference flow details, see [inference_logic.md](./inference_logic.md).

For usage examples, see the [examples directory](../docs/releases/rfd3/examples/).

---

**Last Updated**: 2025-12-04
**Version**: RFDiffusion3 Foundry v0.1.0
