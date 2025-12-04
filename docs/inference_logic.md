# RFDiffusion3 - Inference Logic Documentation

This document provides a detailed technical explanation of the RFDiffusion3 inference pipeline, from CLI input to structure output.

## Table of Contents

1. [Overview](#overview)
2. [Inference Flow](#inference-flow)
3. [Input Specification](#input-specification)
4. [Data Pipeline](#data-pipeline)
5. [Diffusion Sampling](#diffusion-sampling)
6. [Output Generation](#output-generation)
7. [Advanced Features](#advanced-features)

---

## Overview

RFDiffusion3 uses a diffusion-based generative model to design protein structures. The inference process involves:

1. **Input parsing**: Load and validate design specifications
2. **Tokenization**: Convert structure to model-compatible format
3. **Diffusion sampling**: Iteratively denoise from random noise to structure
4. **Output writing**: Save generated structures as CIF files

**Key Concept**: The model learns to reverse a noise-corruption process, starting from Gaussian noise and gradually denoising to produce protein coordinates.

---

## Inference Flow

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. CLI Entry Point                                              │
│    rfd3 design [inputs] [out_dir] [options]                     │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 v
┌─────────────────────────────────────────────────────────────────┐
│ 2. Parse Design Specifications                                  │
│    • Load JSON/YAML/structure files                             │
│    • Validate input parameters                                  │
│    • Create DesignInputSpecification objects                    │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 v
┌─────────────────────────────────────────────────────────────────┐
│ 3. Initialize Inference Engine                                  │
│    • Load model checkpoint                                      │
│    • Set up transform pipeline                                  │
│    • Configure sampler                                          │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 v
┌─────────────────────────────────────────────────────────────────┐
│ 4. Data Pipeline                                                │
│    • Apply transforms (featurization, tokenization)             │
│    • Create feature dictionary                                  │
│    • Prepare batch for model                                    │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 v
┌─────────────────────────────────────────────────────────────────┐
│ 5. Diffusion Sampling                                           │
│    • Initialize random noise                                    │
│    • Iteratively denoise for N timesteps                        │
│    • Apply motif constraints                                    │
│    • Optional: Classifier-free guidance                         │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 v
┌─────────────────────────────────────────────────────────────────┐
│ 6. Post-Processing & Output                                     │
│    • Remove virtual atoms                                       │
│    • Assign sequences (optional)                                │
│    • Write CIF files                                            │
│    • Save metadata JSON                                         │
└─────────────────────────────────────────────────────────────────┘
```

### Detailed Code Flow

#### Entry Point: CLI

**File**: `/models/rfd3/src/rfd3/cli.py`

```python
@app.command()
def design(
    inputs: Optional[str] = None,
    out_dir: str = "./outputs",
    num_designs: int = 1,
    # ... other options
):
    """Main CLI entry for RFD3 design"""
```

**Flow**:
1. Parse command-line arguments
2. Compose Hydra configuration
3. Call `run_inference()` from `run_inference.py`

#### Hydra Main: Run Inference

**File**: `/models/rfd3/src/rfd3/run_inference.py`

```python
@hydra.main(config_path="configs", config_name="inference")
def run_inference(cfg: DictConfig) -> None:
    """
    Main inference entry point using Hydra configuration.

    Flow:
        1. Instantiate RFD3InferenceEngine
        2. Call engine.run() with inputs
    """
    engine = instantiate_inference_engine(cfg)
    engine.run(
        inputs=cfg.inputs,
        out_dir=cfg.out_dir,
        n_batches=cfg.num_designs,
    )
```

#### Inference Engine

**File**: `/models/rfd3/src/rfd3/engine.py`

```python
class RFD3InferenceEngine(BaseInferenceEngine):
    def run(
        self,
        *,
        inputs: str | PathLike | AtomArray | DesignInputSpecification,
        n_batches: int | None = None,
        out_dir: str | PathLike | None = None,
    ):
        """
        Main inference execution.

        Steps:
            1. Canonicalize inputs → Dict[example_id, specification]
            2. Multiply specifications for n_batches
            3. Initialize model
            4. Run multi-batch inference
            5. Write outputs
        """
```

**Key Method: `_model_forward`**

```python
def _model_forward(self, pipeline_output) -> List[RFD3Output]:
    """
    Single forward pass through the model.

    Args:
        pipeline_output: Featurized data from transform pipeline

    Returns:
        List of RFD3Output objects with:
            - atom_array: Final structure
            - metadata: Inference parameters and metrics
            - trajectory_stacks: Optional denoising trajectory
    """
    with torch.no_grad():
        output_val = self.trainer.validation_step(
            batch=pipeline_output,
            batch_idx=0,
            compute_metrics=False,
        )
    # Convert outputs to RFD3Output objects
    ...
```

---

## Input Specification

### DesignInputSpecification

**File**: `/models/rfd3/src/rfd3/inference/input_parsing.py`

Pydantic model for validating design inputs:

```python
class DesignInputSpecification(BaseModel):
    # Input structure (optional)
    input: Optional[str] = None
    assembly_id: str = "1"

    # Motif scaffolding
    motif: Optional[str] = None          # Chain/residue selection
    fixed_atoms: Optional[Dict] = None   # Atom-level fixation

    # Design targets
    length: Optional[str | int] = None   # Target length
    chains: Optional[str] = None         # Chain specification

    # Conditioning
    hotspots: Optional[str] = None       # Residue-level hotspots
    secondary_structure: Optional[str] = None

    # Symmetry
    symmetry: Optional[str] = None       # e.g., "C3", "D2"

    # Advanced
    ori_token: Optional[List[float]] = None
    infer_ori_strategy: Optional[str] = None
    partial_T: Optional[float] = None    # Partial diffusion
```

### Input Processing

**Function**: `process_input()` in `/models/rfd3/src/rfd3/engine.py`

```python
def process_input(
    inputs: str | list | None,
    json_keys_subset: Optional[str | list] = None,
    global_prefix: Optional[str] = None,
    specification_overrides: Optional[dict] = None,
    validate: bool = True,
) -> Dict[str, dict]:
    """
    Converts various input formats to specification dictionaries.

    Input formats:
        1. JSON/YAML file with multiple examples
        2. Structure file (PDB/CIF)
        3. None (unconditional generation)
        4. List of any above

    Returns:
        {
            'example_id_1': {**specification_args},
            'example_id_2': {**specification_args},
            ...
        }
    """
```

**Example JSON Input**:

```json
{
    "design_1": {
        "input": "motif.pdb",
        "motif": "A",
        "length": 100,
        "hotspots": "A10-15,A45-50"
    },
    "design_2": {
        "input": "binder.cif",
        "motif": "B",
        "length": "50-80"
    }
}
```

---

## Data Pipeline

### Transform Sequence

Transforms are applied sequentially to convert raw input into model features:

```python
# Simplified pipeline
transforms = [
    # 1. Structure loading
    LoadStructureFromFile(),

    # 2. Type assignment
    AssignTypes(),               # is_protein, is_ligand, etc.

    # 3. Motif conditioning
    UnindexFlaggedTokens(),      # Handle atomized motifs

    # 4. Virtual atom padding
    PadTokensWithVirtualAtoms(), # Ensure fixed token size

    # 5. Reference features
    CreateDesignReferenceFeatures(),

    # 6. Feature encoding
    AddIsXFeats(),               # Boolean masks
    FeaturizeAtoms(),            # RASA, H-bonds

    # 7. Tokenization
    TokenizeStructure(),         # Convert to model format

    # 8. Batching for diffusion
    BatchStructuresForDiffusionNoising(),
]
```

### Tokenization

**Key Concept**: Proteins are represented at two granularities:

1. **Token level** (residue/ligand level): Used for pairwise features and transformers
2. **Atom level**: Used for coordinate prediction

**Tokenization Process**:

```
Atom Array (biotite)
    ↓
Group by residue/ligand
    ↓
Create mappings:
    - atom_to_token_map: [n_atoms] → token indices
    - token_starts: [n_tokens] → first atom of each token
    ↓
Features:
    - s_I: Token embeddings [n_tokens, c_s]
    - z_II: Token pair embeddings [n_tokens, n_tokens, c_z]
    - q_L: Atom embeddings [n_atoms, c_atom]
    - p_LL: Atom pair embeddings [n_atoms, n_atoms, c_atompair]
```

**Motif Handling**:

- **Indexed motifs**: Standard residues with fixed coordinates/sequence
- **Unindexed motifs**: Atomized ligands/non-standard residues
  - Each atom becomes its own token
  - Allows arbitrary molecular structures

### Feature Dictionary

**Output of Pipeline**: `pipeline_output` dict

```python
{
    # Original structure
    "atom_array": AtomArray,
    "example_id": str,

    # Model inputs
    "f": {  # Feature dict
        "atom_to_token_map": Tensor[n_atoms],
        "is_ca": Tensor[n_atoms],
        "is_motif_atom_with_fixed_coord": Tensor[n_atoms],
        # ... many more features
    },

    # Coordinates to noise
    "coord_atom_lvl_to_be_noised": Tensor[n_atoms, 3],

    # Ground truth (training only)
    "ground_truth": {
        "X_L": Tensor[n_atoms, 3],
        "sequence_gt_I": Tensor[n_tokens],
    },
}
```

---

## Diffusion Sampling

### Diffusion Process Overview

**Training**: Learn to denoise coordinates at various noise levels
- Sample timestep \( t \sim \text{Uniform}(0, t_{max}) \)
- Add noise: \( X_{noisy} = X_{gt} + \epsilon \cdot \sigma(t) \)
- Predict denoised: \( \hat{X}_{denoised} = f_{\theta}(X_{noisy}, t, features) \)
- Loss: \( \mathcal{L} = \|X_{gt} - \hat{X}_{denoised}\|^2 \)

**Inference**: Reverse the process from noise to structure
- Start: \( X_0 \sim \mathcal{N}(0, \sigma_{max}^2 I) \)
- Denoise iteratively for \( N \) steps
- End: \( X_N \approx \) clean structure

### Sampling Algorithm

**File**: `/models/rfd3/src/rfd3/model/inference_sampler.py`

**Method**: `sample_diffusion_like_af3()`

#### Step 1: Initialize Noise

```python
# Sample from Gaussian
X_noisy = torch.randn(batch_size, n_atoms, 3) * sigma_max

# Zero out motif atoms (they stay fixed)
X_noisy[..., is_motif_atom_with_fixed_coord, :] = 0
```

#### Step 2: Construct Noise Schedule

```python
t_schedule = construct_inference_noise_schedule()
# Returns: [t_max, t_{max-1}, ..., t_1, t_0]
# where t_i decreases from ~160 to ~0.0004
```

**Schedule formula**:
\[ t_i = \sigma_{data} \left( s_{max}^{1/p} + \frac{i}{N} (s_{min}^{1/p} - s_{max}^{1/p}) \right)^p \]

Default values:
- \( \sigma_{data} = 16 \)
- \( s_{max} = 160 \), \( s_{min} = 0.0004 \)
- \( p = 7 \)
- \( N = 200 \) steps

#### Step 3: Iterative Denoising

```python
for step, (t_curr, t_next) in enumerate(zip(t_schedule[:-1], t_schedule[1:])):
    # 3a. Model forward pass
    model_output = diffusion_module(
        X_noisy_L=X_noisy,
        t=t_curr,
        f=features,
        **initializer_outputs,
    )
    X_denoised = model_output["X_L"]

    # 3b. Compute update direction (AF3 second-order solver)
    gamma = compute_gamma(t_curr)
    d = (X_noisy - X_denoised) / (t_curr + gamma)

    # 3c. Euler step
    dt = t_next - t_curr
    X_noisy = X_noisy + d * dt

    # 3d. Add noise (stochastic sampling)
    if t_next > 0:
        noise = torch.randn_like(X_noisy) * noise_scale * sqrt(dt)
        X_noisy = X_noisy + noise

    # 3e. Enforce motif constraints
    X_noisy[..., is_motif_atom_with_fixed_coord, :] = motif_coords

    # 3f. Re-center coordinates
    X_noisy = recenter_coords(X_noisy, center_option)
```

#### Gamma Schedule (Noise Scaling)

```python
def compute_gamma(t: float) -> float:
    """
    Annealing schedule for noise injection.

    Returns larger values at high t (more noise early)
    and smaller values at low t (less noise late).
    """
    if t > s_max:
        return gamma_0
    else:
        return max(gamma_min, gamma_0 * (t / s_max))
```

### Motif Constraints

**Fixed Coordinates**: Motif atoms are never noised and are enforced at every step:

```python
# Before each denoising step
X_noisy[..., is_motif_atom_with_fixed_coord, :] = X_motif_gt
```

**Partial Fixation**: Option to allow motif to move during early diffusion steps:

```python
if step < num_timesteps * fraction_of_steps_to_fix_motif:
    # Don't enforce motif yet
    pass
else:
    # Enforce motif
    X_noisy[..., is_motif_atom_with_fixed_coord, :] = X_motif_gt
```

### Classifier-Free Guidance (CFG)

**Purpose**: Enhance conditioning signal strength

**Method**:
1. Run model twice per step:
   - **Conditional**: With full features \( \epsilon_{cond} = f(x_t, t, c) \)
   - **Unconditional**: With stripped features \( \epsilon_{uncond} = f(x_t, t, \emptyset) \)

2. Extrapolate:
   \[ \epsilon_{guided} = \epsilon_{uncond} + w \cdot (\epsilon_{cond} - \epsilon_{uncond}) \]
   where \( w \) is `cfg_scale` (typically 2.0).

3. Use \( \epsilon_{guided} \) for denoising

**Stripped Features**: Remove conditioning signals like hotspots, secondary structure, etc.

```python
if use_classifier_free_guidance and t > cfg_t_max:
    # Run unconditional
    output_uncond = diffusion_module(X_noisy, t, f_stripped, ...)

    # Run conditional
    output_cond = diffusion_module(X_noisy, t, f_full, ...)

    # Combine
    X_denoised = output_uncond["X_L"] + cfg_scale * (
        output_cond["X_L"] - output_uncond["X_L"]
    )
```

---

## Output Generation

### Post-Processing

After sampling completes:

1. **Remove virtual atoms**: Filter out padding atoms
   ```python
   is_not_virtual = atom_array.element != VIRTUAL_ATOM_ELEMENT_NAME
   atom_array = atom_array[is_not_virtual]
   ```

2. **Remove guideposts**: Filter out ORI and other guide tokens
   ```python
   is_not_guidepost = ~np.isin(atom_array.res_name, ["ORI", "HOH"])
   atom_array = atom_array[is_not_guidepost]
   ```

3. **Assign sequence** (optional): Use model's sequence prediction head
   ```python
   sequence_indices = model_output["sequence_indices_I"]
   atom_array = assign_sequence_to_structure(atom_array, sequence_indices)
   ```

### Output Structure

**File**: RFD3Output dataclass

```python
@dataclass
class RFD3Output:
    atom_array: AtomArray           # Final designed structure
    metadata: dict                  # Inference config + metrics
    example_id: str                 # Unique identifier
    denoised_trajectory_stack: Optional[AtomArrayStack] = None
    noisy_trajectory_stack: Optional[AtomArrayStack] = None
```

**Metadata Contents**:

```python
{
    "ckpt_path": "/path/to/checkpoint",
    "seed": 42,
    "num_timesteps": 200,
    "cfg_scale": 2.0,
    "input_specification": {...},
    "design_metrics": {
        "motif_rmsd": 0.15,  # If applicable
        # ... other metrics
    }
}
```

### File Writing

**CIF Output**: `example_id_model_0.cif.gz`

```python
to_cif_file(
    atom_array,
    output_path,
    file_type="cif.gz",
    include_entity_poly=False,
    extra_fields=SAVED_CONDITIONING_ANNOTATIONS,
)
```

**JSON Metadata**: `example_id_model_0.json`

```python
with open(f"{output_path}.json", "w") as f:
    json.dump(metadata, f, indent=4)
```

**Trajectory Output** (if `dump_trajectories=True`):
- `example_id_denoised_model_0.cif.gz`: Denoising trajectory
- `example_id_noisy_model_0.cif.gz`: Noised trajectory

Shows structure at different diffusion timesteps for visualization.

---

## Advanced Features

### Symmetry Design

**File**: `/models/rfd3/src/rfd3/inference/symmetry/`

For symmetric assemblies (e.g., C3, D2):

1. **Input**: Only provide asymmetric unit
2. **Symmetry expansion**: Replicate unit according to symmetry group
3. **Constrained sampling**: Ensure all copies maintain symmetry
4. **Output**: Full symmetric assembly

**Example**:
```python
specification = {
    "input": "monomer.pdb",
    "symmetry": "C3",  # 3-fold rotational symmetry
    "length": 100,
}
```

### Partial Diffusion

**Use case**: Refine existing structure rather than generate from scratch

**Parameter**: `partial_T`

```python
specification = {
    "input": "initial_design.pdb",
    "partial_T": 50.0,  # Start from t=50 instead of t_max
}
```

**Effect**: Skips high-noise timesteps, starts denoising from intermediate noise level.

### Hotspot Conditioning

**Purpose**: Bias design to have specific residues at specified locations

**Input**:
```python
specification = {
    "input": "target.pdb",
    "motif": "A",
    "hotspots": "A10-15,A45-50",  # Residues for binding
}
```

**Mechanism**: Adds atom-level features indicating hotspot regions, model learns to design around them.

### Secondary Structure Conditioning

**Purpose**: Control secondary structure elements

**Input**:
```python
specification = {
    "length": 100,
    "secondary_structure": "HHHHHHHHH" + "L"*10 + "EEEEE",
    # H = helix, E = strand, L = loop
}
```

**Encoding**: Per-token features indicating desired SS type.

### Low Memory Mode

**Environment Variable**: `RFD3_LOW_MEMORY_MODE=1`

**Effect**: Enables chunked computation of pairwise features to reduce GPU memory:

```python
# Standard mode
P_LL = compute_pairwise_features(q_L)  # [n_atoms, n_atoms, c_pair]
output = attention(q_L, P_LL)

# Chunked mode (low memory)
output = chunked_attention(q_L, pairwise_embedder_fn)
# Computes P_LL in chunks on-the-fly
```

**Trade-off**: Lower memory usage, slightly slower.

---

## Example: End-to-End Inference

### Command

```bash
rfd3 design \
    --inputs design.json \
    --out_dir ./outputs \
    --num_designs 10 \
    --inference_sampler.num_timesteps 200 \
    --inference_sampler.cfg_scale 2.0 \
    --seed 42
```

### design.json

```json
{
    "binder_design": {
        "input": "target.pdb",
        "motif": "A",
        "length": "60-80",
        "hotspots": "A10-15,A45-50",
        "infer_ori_strategy": "hotspots"
    }
}
```

### Execution Flow

1. **Parse inputs**: Load `design.json` → `{"binder_design": {...}}`
2. **Multiply**: Create 10 copies: `binder_design_0` through `binder_design_9`
3. **Load target**: Parse `target.pdb` → AtomArray
4. **Extract motif**: Select chain A as fixed motif
5. **Infer origin**: Center on hotspot residues + 10Å offset
6. **Sample length**: Randomly sample 60-80 (e.g., 72)
7. **Initialize**: Create 72-residue backbone with random coords
8. **Transform pipeline**: Featurize and tokenize
9. **Diffusion sampling**: Run 200-step denoising
10. **Write outputs**:
    - `outputs/binder_design_0_model_0.cif.gz`
    - `outputs/binder_design_0_model_0.json`
    - ... (repeat for all 10 designs)

### Output Inspection

```python
import json
from biotite.structure.io.pdb import PDBFile

# Load structure
file = PDBFile.read("outputs/binder_design_0_model_0.cif.gz")
structure = file.get_structure()

# Load metadata
with open("outputs/binder_design_0_model_0.json") as f:
    metadata = json.load(f)

print(f"Designed {structure.shape[0]} atoms")
print(f"Sampling used {metadata['num_timesteps']} timesteps")
print(f"Motif RMSD: {metadata.get('motif_rmsd', 'N/A')}")
```

---

## Summary

RFDiffusion3 inference consists of:

1. **Input parsing**: Flexible specification format (JSON/structure files)
2. **Featurization**: Transform pipeline converts raw data to model features
3. **Diffusion sampling**: Iterative denoising with motif constraints
4. **Output**: CIF structures with metadata

**Key innovations**:
- **Motif scaffolding**: Fixed regions guide design
- **Classifier-free guidance**: Enhanced conditioning
- **Symmetry support**: Native symmetric assembly design
- **Flexible conditioning**: Hotspots, secondary structure, partial diffusion

For implementation details, see [core_functions.md](./core_functions.md).

For usage examples, see the [RFD3 documentation](../docs/releases/rfd3/).

---

**Last Updated**: 2025-12-04
**Version**: RFDiffusion3 Foundry v0.1.0
