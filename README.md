# Make Tracking Easy: Neural Motion Retargeting for Humanoid Whole-body Control (NMR)

[[Paper](#)] [[Website](#)] [[HuggingFace Demo](https://huggingface.co/spaces/RayZhao/NMR)]

## News

- **2026.03.24**: Release NMR paper and website.
- **2026.03.26**: Release HuggingFace live demo.
- **2026.04**: Release deployable inference code and checkpoint.

## TODOs

- [x] 2026.03.24: Release NMR paper and website.
- [x] 2026.03.26: Release HuggingFace live demo: https://huggingface.co/spaces/RayZhao/NMR
- [x] Release deployable inference code.
- [ ] Release CEPR dataset (SMPL and robot).
- [ ] Release training code.

---

## Quick Start

### 1. Install Dependencies

We recommend using conda:

```bash
conda create -n nmr python=3.10
conda activate nmr
```

Install PyTorch (adjust CUDA version as needed):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Install Pinocchio (robot forward kinematics, needed for visualization):

```bash
conda install -c conda-forge pinocchio
```

Install remaining dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the Gradio Web Demo

```bash
python app.py
```

On first run, the model checkpoint (~518 MB) and SMPL-X body model (~104 MB) will be **automatically downloaded** from HuggingFace Hub. Subsequent runs load from cache.

Upload any AMASS `.npz` file (or use the provided examples in `examples/`) to get:
- Interactive 3D skeleton animation
- Downloadable `.pkl` result file

### 3. Command-line Inference

```bash
python inference.py --src examples/sample_motion.npz --output-dir output/
```

**Batch processing** (a directory of NPZ/PKL files):

```bash
python inference.py --src /path/to/motions/ --output-dir output/
```

**Disable low-pass filter** (raw network output):

```bash
python inference.py --src examples/sample_motion.npz --output-dir output/ --no-filter
```

#### Input formats

| Format | Fields | Coordinate |
|--------|--------|-----------|
| AMASS `.npz` | `trans`, `root_orient`, `pose_body` | Z-up (auto-converted) |
| Standard `.npz` | `transl`, `global_orient`, `body_pose` | Y-up |
| `.pkl` | same as above | Z-up or Y-up |

High frame-rate sequences (>30 FPS) are automatically downsampled to 30 FPS.

#### Output format

A `.pkl` file containing a dictionary:

```python
{
    'dof':           np.ndarray (T, 29),   # joint angles [rad]
    'root_trans':    np.ndarray (T, 3),    # root XYZ position [m]
    'root_rot_quat': np.ndarray (T, 4),   # root orientation quaternion (w, x, y, z)
}
```

### 4. Python API

```python
from inference import load_all, infer_single

# Load all models (auto-downloads weights on first call)
model, smplx_model, betas, smplx_mean, smplx_std, g1_mean, g1_std, device = load_all()

# Run inference
result, timing = infer_single(
    "examples/sample_motion.npz",
    model, smplx_model, betas,
    smplx_mean, smplx_std, g1_mean, g1_std, device
)
# result: dict with 'dof', 'root_trans', 'root_rot_quat'
```

---

## Model Architecture

NMR uses a two-stage pipeline:

```
SMPL-X motion (T, 140)
        ↓
   SMPL-X VQ-VAE Encoder
        ↓ (T/4, 512)
   LLaMA Transformer (forward, non-autoregressive)
        ↓ (T/4, 512)
   G1 VQ-VAE Decoder
        ↓
G1 robot motion (T, 217)
        ↓
post-processing (Butterworth low-pass filter)
        ↓
{dof (T,29), root_trans (T,3), root_rot_quat (T,4)}
```

**Stage 1 — VQ-VAE Tokenizer**: Encodes SMPL-X human motion into a compact latent space using FSQ quantization (codebook size 1920, temporal downsampling ×4).

**Stage 2 — Transformer**: A 70M-parameter LLaMA-style model that maps human motion embeddings to G1 robot motion embeddings in a one-to-one forward pass (non-autoregressive).

For full architecture details, see the paper.

---

## Checkpoint

Model weights are hosted on HuggingFace Hub at [`RayZhao/NMR`](https://huggingface.co/RayZhao/NMR) and will be downloaded automatically on first use.

If you prefer to download manually:

```bash
huggingface-cli download RayZhao/NMR weights/epoch_30.pth --local-dir .
huggingface-cli download RayZhao/NMR assets/SMPLX_NEUTRAL.npz --local-dir .
```

---

## Citation

```bibtex
@article{nmr2026,
  title={Make Tracking Easy: Neural Motion Retargeting for Humanoid Whole-body Control},
  author={...},
  year={2026}
}
```

## License

[To be added]
