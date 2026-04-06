# 🧵 Lace Fabric Synthesis — GAN + Stable Diffusion

A 2-stage image generation pipeline for high-quality lace fabric synthesis:

- **Stage 1 — GAN**: StyleGAN-inspired architecture learns global lace structure (patterns, mesh holes, shapes)
- **Stage 2 — Diffusion**: Stable Diffusion img2img refines coarse GAN output into photorealistic lace with thread-level detail

---

## 📁 Project Structure

```
lace-gan-diffusion/
├── config.yaml                  # Central configuration (edit this first)
├── requirements.txt
├── preprocess.py                # Dataset preprocessing
│
├── stage1_gan/
│   ├── dataset.py               # LaceDataset + DataLoader
│   ├── model.py                 # Generator (StyleGAN-inspired) + Discriminator
│   ├── losses.py                # Adversarial + Gradient Penalty + Perceptual loss
│   └── train.py                 # Full training loop (fp16, checkpoints, TensorBoard)
│
├── stage2_diffusion/
│   ├── refiner.py               # SD img2img pipeline (memory-optimized)
│   └── prompts.py               # Curated lace text prompts
│
├── inference/
│   ├── pipeline.py              # Stage 1 → Stage 2 end-to-end script
│   └── evaluate.py              # FID score + CLIP score + visual comparison
│
├── data/
│   ├── raw/                     # ← Put your lace images here
│   └── processed/               # Auto-generated after preprocessing
│
├── checkpoints/gan/             # Auto-generated during training
└── outputs/
    ├── gan_generated/           # Stage 1 outputs
    ├── refined/                 # Stage 2 outputs
    └── final/                   # Pipeline final outputs
```

---

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Server with CUDA**: Make sure PyTorch is installed with the correct CUDA version:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### 2. Prepare Dataset

Put your lace fabric images into `data/raw/`. Then run:

```bash
# Resize to 256×256 (default)
python preprocess.py --size 256

# Or 512×512 for higher quality (needs more VRAM)
python preprocess.py --size 512 --keep_aspect
```

**Where to find lace datasets:**
- [DTD (Describable Textures Dataset)](https://www.robots.ox.ac.uk/~vgg/data/dtd/) — filter for "lacy" category
- [Kaggle: Fabric Texture Dataset](https://www.kaggle.com/datasets)
- Google Images / Pinterest → filter by lace fabric

---

## 🚀 Running

### Step 1 — Train the GAN

```bash
python stage1_gan/train.py --config config.yaml
```

**Resume after interruption:**
```bash
python stage1_gan/train.py --config config.yaml --resume
```

**Monitor training:**
```bash
tensorboard --logdir logs/gan
```

Training saves:
- `checkpoints/gan/latest.pth` — always latest
- `checkpoints/gan/best_generator.pth` — best G loss
- `checkpoints/gan/epoch_XXXX.pth` — every `save_every` epochs
- `outputs/gan_samples/epoch_XXXX.png` — visual progress every `sample_every` epochs

---

### Step 2 — Run the Full Pipeline (GAN → Diffusion)

```bash
# Default style
python inference/pipeline.py --config config.yaml --num_images 20

# Vintage lace style
python inference/pipeline.py --config config.yaml --num_images 20 --style vintage

# Stage 1 only (no Stable Diffusion)
python inference/pipeline.py --config config.yaml --skip_diffusion

# Use a specific checkpoint
python inference/pipeline.py --config config.yaml \
    --checkpoint checkpoints/gan/epoch_0150.pth \
    --style bridal
```

**Available styles:** `default`, `vintage`, `modern`, `geometric`, `floral`, `bridal`

---

### Step 3 — Evaluate

```bash
python inference/evaluate.py \
    --real_dir   data/processed \
    --gan_dir    outputs/gan_generated \
    --refined_dir outputs/final \
    --prompt "white lace fabric with intricate floral pattern, ultra detailed"
```

Output:
```
┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
  Metric               GAN    Refined
  ─────────────────────────────────
  FID ↓              185.3     92.7
  CLIP Score ↑        21.4     28.1
┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
  ✅ FID improvement: 92.6
```

A side-by-side image grid is saved to `outputs/comparison.png`.

---

## 🎛️ Key Parameters (config.yaml)

| Parameter | Default | Effect |
|---|---|---|
| `data.image_size` | 256 | Training resolution |
| `data.batch_size` | 8 | Reduce to 4 if OOM |
| `stage1_gan.epochs` | 200 | More = better quality |
| `stage1_gan.loss_type` | hinge | `hinge` / `bce` / `lsgan` |
| `stage1_gan.lambda_gp` | 10.0 | Gradient penalty strength |
| `stage2_diffusion.strength` | 0.6 | 0.4=subtle, 0.7=heavy rework |
| `stage2_diffusion.guidance_scale` | 8.5 | Prompt adherence (7–15) |
| `stage2_diffusion.num_inference_steps` | 30 | More steps = better quality |
| `stage2_diffusion.enable_cpu_offload` | false | Set `true` for <8GB VRAM |

---

## 💡 Sample Prompts

```python
# White floral lace
"white lace fabric with intricate floral pattern, delicate threads, fine mesh holes, 8k"

# Vintage lace
"antique ivory lace, vintage scalloped border, rose motifs, macro photography, museum quality"

# Modern geometric
"modern black lace with geometric mesh pattern, studio lighting, sharp details, high contrast"

# Bridal
"bridal white lace, elegant scallop edges, luxury wedding fabric, close-up macro, 4k"
```

---

## 🧠 Why GAN + Diffusion?

```
Random Noise → [GAN] ──────────────────→ [Stable Diffusion] → Final Image
                 ↑                               ↑
          Learns structure              Refines details
          (mesh, patterns)              (threads, holes, texture)
```

| | GAN | Diffusion | GAN + Diffusion |
|---|---|---|---|
| Global structure | ✅ Strong | ❌ Inconsistent | ✅ |
| Fine detail | ❌ Blurry | ✅ Excellent | ✅ |
| Speed | ✅ Fast | ⚠️ Slower | ⚠️ Moderate |
| Training needed | ✅ Yes | ❌ Pre-trained | ✅ Only GAN |

---

## 🚀 Possible Improvements

### 1. ControlNet
Replace simple img2img with ControlNet conditioned on edge maps extracted from GAN output.
More structural control, better consistency.

```python
# Concept
canny_edges = extract_canny(gan_image)
final = controlnet_pipeline(edges=canny_edges, prompt="lace fabric…")
```

### 2. LoRA Fine-tuning
Fine-tune Stable Diffusion on your lace dataset using LoRA adapters.
The model learns your specific lace domain — much more accurate detail.

```bash
# Using diffusers training scripts
python train_lora.py --dataset data/processed --output lora/lace_v1
```

### 3. Textual Inversion
Train a `<lace-v1>` token embedding that captures your dataset's specific style.
Use it in any SD prompt: `"<lace-v1> with floral motifs"`.

---

## 📋 Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0 + CUDA
- **Recommended VRAM**: 12-24 GB (server)
- **Minimum VRAM**: 8 GB (with `enable_cpu_offload: true`)
- RAM: ≥ 16 GB
# lace-gan-diffusion
# lace-gan-diffusion
# lace-gan-diffusion
