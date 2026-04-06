# 🧵 Sinh Tổng Hợp Vải Ren — GAN + Stable Diffusion

> **Đề tài:** Synthetic Fabric Texture Generation Using GAN and Stable Diffusion  
> **Môn học:** Machine Learning  
> **Nhóm thực hiện:** haluu07  
> **Server:** 2× NVIDIA RTX A4000 (16GB VRAM) | Ubuntu 24.04 | CUDA 12.1

---

## 📌 Tóm tắt

Dự án xây dựng pipeline 2 giai đoạn để tổng hợp ảnh vải ren chất lượng cao từ nhiễu ngẫu nhiên:

| Giai đoạn | Mô hình | Đầu vào | Đầu ra |
|---|---|---|---|
| Stage 1 | GAN (LSGAN) | Noise vector z (512 chiều) | Ảnh vải ren 256×256 |
| Stage 2 | Stable Diffusion 1.5 + LoRA | Ảnh GAN + Text prompt | Ảnh tinh chỉnh 256×256 |

**Dataset:** 960 ảnh vải ren thu thập từ internet, tiền xử lý về 256×256.

---

## 🗺️ Sơ đồ tổng quan hệ thống

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PIPELINE TỔNG QUAN                          │
│                                                                     │
│   ┌──────────┐     ┌─────────────────┐     ┌───────────────────┐   │
│   │ Noise z  │────▶│   STAGE 1: GAN  │────▶│  STAGE 2: SD+LoRA │   │
│   │ 512-dim  │     │  (Generator G)  │     │   (img2img refine)│   │
│   └──────────┘     └─────────────────┘     └───────────────────┘   │
│                            │                          │             │
│                            ▼                          ▼             │
│                    ┌──────────────┐          ┌───────────────┐      │
│                    │ Ảnh thô GAN  │          │ Ảnh chất lượng│      │
│                    │  256×256     │    +     │   cao 256×256 │      │
│                    └──────────────┘          └───────────────┘      │
│                                                       ▲             │
│                                              ┌────────┴──────┐      │
│                                              │  Text Prompt  │      │
│                                              │ "white lace…" │      │
│                                              └───────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Thuật toán chi tiết

### STAGE 1 — GAN (Generative Adversarial Network)

#### Sơ đồ kiến trúc GAN

```
                    ┌──────────────────────────────────────────┐
                    │            GENERATOR (G)                  │
                    │                                            │
  Noise z ─────────▶ Linear(512→4×4×1024)                      │
  (512-dim)         │      │ Reshape (1024, 4, 4)               │
                    │      ▼                                     │
                    │ ConvT(1024→512) + BN + ReLU  → 8×8        │
                    │      ▼                                     │
                    │ ConvT(512→256)  + BN + ReLU  → 16×16      │
                    │      ▼                                     │
                    │ ConvT(256→128)  + BN + ReLU  → 32×32      │
                    │      ▼                                     │
                    │ ConvT(128→64)   + BN + ReLU  → 64×64      │
                    │      ▼                                     │
                    │ ConvT(64→32)    + BN + ReLU  → 128×128    │
                    │      ▼                                     │
                    │ ConvT(32→3)     + Tanh        → 256×256   │
                    └──────────────────────────────────────────┘
                                        │
                              Ảnh giả 256×256
                                        │
          ┌─────────────────────────────┼──────────────────────┐
          │                             │                       │
          ▼                             ▼                       │
  Ảnh thật (real)               Ảnh giả (fake)                 │
  từ dataset                    từ Generator                    │
          │                             │                       │
          └─────────────────────────────┘                       │
                          │                                     │
                    ┌─────▼────────────────────────────────┐    │
                    │          DISCRIMINATOR (D)            │    │
                    │                                       │    │
                    │ Conv(3→64)     + LReLU  → 128×128    │    │
                    │ Conv(64→128)   + BN + LReLU → 64×64  │    │
                    │ Conv(128→256)  + BN + LReLU → 32×32  │    │
                    │ Conv(256→512)  + BN + LReLU → 16×16  │    │
                    │ Conv(512→1024) + BN + LReLU → 8×8    │    │
                    │ Flatten → Linear → Sigmoid            │    │
                    └──────────────────────────────────────┘    │
                                        │                       │
                              Xác suất [0,1]                    │
                                        │                       │
                    ┌───────────────────┴───────────────────┐   │
                    │          LOSS FUNCTION (LSGAN)        │   │
                    │                                       │   │
                    │ L_D = E[(D(x)-1)²] + E[(D(G(z)))²]  │   │
                    │ L_G = E[(D(G(z))-1)²]               │   │
                    └───────────────────────────────────────┘   │
                              │                     │            │
                         Cập nhật D           Cập nhật G ───────┘
```

#### Tại sao chọn LSGAN?

| Loss Type | Vấn đề | Dataset nhỏ |
|---|---|---|
| Vanilla GAN | Vanishing gradient → G không học được | ❌ |
| Wasserstein GAN | Phức tạp, cần Lipschitz constraint | ⚠️ |
| **LSGAN** | Ổn định, gradient luôn có giá trị | ✅ |

#### Hyperparameters Stage 1

| Tham số | Giá trị | Lý do |
|---|---|---|
| Latent dim | 512 | Đủ biểu diễn đa dạng, không quá lớn |
| Learning rate G | 3×10⁻⁵ | Thấp để tránh mode collapse |
| Learning rate D | 3×10⁻⁵ | Bằng G để cân bằng |
| β₁ Adam | 0.5 | Giảm momentum → GAN ổn định hơn |
| β₂ Adam | 0.999 | Standard |
| Batch size | 4 | Giới hạn VRAM 16GB |
| Epochs | 800 | Đủ để hội tụ |
| n_critic | 1 | D:G = 1:1 (LSGAN không cần nhiều D steps) |

---

### STAGE 2 — Stable Diffusion + LoRA

#### Sơ đồ kiến trúc Stable Diffusion

```
  Text Prompt
"white lace fabric..."
        │
        ▼
┌──────────────────┐
│  CLIP Text       │     ← Tokenize + encode text thành vector
│  Encoder         │
│  (frozen)        │
└──────────┬───────┘
           │ Text embeddings
           │ [77 tokens × 768 dim]
           │
           │         ┌─────────────────────────────┐
           │         │        VAE ENCODER           │
           │         │  Ảnh GAN 256×256×3           │
           │         │         │ Compress 8×         │
           │         │  Latent 32×32×4              │
           │         │         │ + noise (strength)  │
           │         │  Noisy latent 32×32×4        │
           │         └────────────┬────────────────┘
           │                      │
           ▼                      ▼
┌──────────────────────────────────────────────────┐
│                   UNet (Denoiser)                 │
│                                                   │
│  ┌──────────────────────────────────────────┐    │
│  │           ENCODER BLOCKS                  │    │
│  │  ResBlock + CrossAttention → 32×32       │    │
│  │  ResBlock + CrossAttention → 16×16       │    │
│  │  ResBlock + CrossAttention → 8×8         │    │
│  └──────────────────┬───────────────────────┘    │
│                     │                             │
│  ┌──────────────────▼───────────────────────┐    │
│  │              BOTTLENECK                   │    │
│  │  ResBlock + CrossAttention (8×8)         │    │
│  └──────────────────┬───────────────────────┘    │
│                     │                             │
│  ┌──────────────────▼───────────────────────┐    │
│  │           DECODER BLOCKS                  │    │
│  │  ResBlock + CrossAttention → 16×16       │    │
│  │  ResBlock + CrossAttention → 32×32       │    │
│  │  ResBlock + CrossAttention → 32×32       │    │
│  └──────────────────┬───────────────────────┘    │
│                     │                             │
│            Noise prediction                       │
└────────────────────┬─────────────────────────────┘
              ▲      │
              │      │ Khử noise (30 steps)
    Text ─────┘      │
  embeddings         ▼
           ┌─────────────────┐
           │   VAE DECODER   │      ← Giải nén latent → pixel
           │  32×32×4 → 256×256×3  │
           └────────┬────────┘
                    │
                    ▼
           Ảnh tinh chỉnh 256×256
```

#### Cross-Attention: Cơ chế text điều khiển ảnh

```
  Text tokens (K, V):  [white] [lace] [fabric] [floral] [pattern]
                          │       │       │        │         │
                       ┌──┴───────┴───────┴────────┴─────────┴──┐
  Latent pixels (Q): ──│         Attention(Q, K, V)              │
                       │  Score = softmax(QKᵀ/√d) × V            │
                       └─────────────────────────────────────────┘
                                          │
Vùng texture nghiêng về "lace" + "fabric"
Vùng họa tiết nghiêng về "floral" + "pattern"
```

#### LoRA — Fine-tune hiệu quả

```
TRƯỚC LoRA (frozen weights):
  Q = W_q × x        (W_q: 768×768 = 589,824 params — không train)

SAU LoRA:
  Q = W_q × x  +  (B × A) × x
                    ↑
              A: 768×16  (12,288 params)
              B: 16×768  (12,288 params)
              
              → Chỉ train 24,576 params thay vì 589,824
              → Tiết kiệm 96% bộ nhớ

  LoRA rank = 16, alpha = 32
  LoRA được inject vào: Q, K, V, Out projection của mỗi Attention layer
```

#### img2img với Strength

```
strength = 0.75 → thêm noise vào 75% timesteps

  Ảnh GAN       Pure noise
  x₀ ──────────────────▶ x_T
       t=0         t=T

  Diffusion bắt đầu từ t = T × strength = T × 0.75
                                   ↑
                     Bắt đầu khử noise từ đây
  
  Kết quả: 25% giữ cấu trúc từ GAN, 75% tinh chỉnh từ Diffusion
```

---

## 📊 Sơ đồ luồng dữ liệu (Data Flow)

```
┌────────────────┐
│    Dataset     │   960 ảnh vải ren (JPG/PNG)
│   (data/raw/)  │
└───────┬────────┘
        │ python preprocess.py
        ▼
┌────────────────┐
│   Processed    │   960 ảnh → resize 256×256
│(data/processed)│   Normalize [-1, 1]
└───────┬────────┘
        │
        ├──────────────────────────────────────────────────┐
        │                                                  │
        ▼                                                  ▼
┌────────────────┐                               ┌──────────────────┐
│  TRAIN GAN     │                               │  TRAIN LoRA      │
│  800 epochs    │                               │  100 epochs      │
│  LSGAN loss    │                               │  SD 1.5 + rank16 │
└───────┬────────┘                               └────────┬─────────┘
        │                                                  │
        ▼                                                  ▼
┌────────────────┐                               ┌──────────────────┐
│  checkpoints/  │                               │ checkpoints/lora │
│  gan/best.pth  │                               │  lace_lora/      │
└───────┬────────┘                               └────────┬─────────┘
        │                                                  │
        └──────────────────┬───────────────────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  INFERENCE      │
                  │  pipeline.py    │
                  └────────┬────────┘
                           │
               ┌───────────┴───────────┐
               │                       │
               ▼                       ▼
    ┌──────────────────┐    ┌──────────────────────┐
    │  Stage 1 output  │    │   Stage 2 output     │
    │ outputs/gan_gen/ │───▶│   outputs/final/     │
    │  (GAN only)      │    │ (GAN + SD tinh chỉnh)│
    └──────────────────┘    └──────────────────────┘
               │                       │
               └───────────┬───────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   EVALUATE      │
                  │  FID + CLIP     │
                  └─────────────────┘
```

---

## 📁 Cấu trúc dự án chi tiết

```
lace-gan-diffusion/
│
├── 📄 config.yaml              # Cấu hình trung tâm (LR, epochs, paths...)
├── 📄 requirements.txt         # Danh sách thư viện
├── 📄 preprocess.py            # Resize + normalize dataset
├── 📄 train_lora.py            # Fine-tune LoRA trên SD 1.5
├── 📄 app.py                   # Giao diện web Gradio (demo)
├── 📄 plot_clip_score.py       # Vẽ biểu đồ so sánh CLIP score
├── 📄 download_dataset.py      # Script tải dataset tự động
│
├── 📁 stage1_gan/              # ═══ GIAI ĐOẠN 1: GAN ═══
│   ├── dataset.py              # LaceDataset: load + augment ảnh
│   ├── model.py                # Generator + Discriminator (LSGAN)
│   ├── losses.py               # LSGAN loss + Perceptual loss
│   └── train.py                # Training loop: checkpoint, sample, log
│
├── 📁 stage2_diffusion/        # ═══ GIAI ĐOẠN 2: DIFFUSION ═══
│   ├── refiner.py              # SD img2img pipeline (tối ưu VRAM)
│   └── prompts.py              # Style → text prompt mapping
│
├── 📁 inference/               # ═══ SINH ẢNH & ĐÁNH GIÁ ═══
│   ├── pipeline.py             # End-to-end: GAN → SD → output
│   ├── evaluate.py             # FID + CLIP + ảnh so sánh
│   └── compute_fid.py          # Tính FID độc lập
│
├── 📁 data/
│   ├── raw/                    # ← Đặt dataset vào đây
│   └── processed/              # Tự động tạo sau preprocess
│
├── 📁 checkpoints/
│   ├── gan/
│   │   ├── best_generator.pth  # Checkpoint G tốt nhất
│   │   ├── latest.pth          # Checkpoint mới nhất
│   │   └── epoch_XXXX.pth      # Mỗi save_every epoch
│   └── lora/
│       └── lace_lora/          # LoRA weights sau fine-tune
│
├── 📁 outputs/
│   ├── gan_generated/          # Ảnh từ GAN (Stage 1)
│   ├── refined/                # Ảnh từ Diffusion (Stage 2)
│   ├── final/                  # Ảnh cuối pipeline đầy đủ
│   └── gan_samples/            # Ảnh sample theo epoch (monitor)
│
└── 📁 logs/
    └── gan/                    # TensorBoard logs (loss curves)
```

---

## ⚙️ Cài đặt môi trường

### Yêu cầu hệ thống

| | Tối thiểu | Khuyến nghị (server) |
|---|---|---|
| Python | ≥ 3.9 | 3.10+ |
| PyTorch | ≥ 2.0 | 2.1+ CUDA |
| VRAM GPU | 8 GB | 16 GB (RTX A4000) |
| RAM | 16 GB | 32 GB |
| Disk | 20 GB | 100 GB |

### Cài thư viện

```bash
pip install -r requirements.txt

# Server với CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Server với CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 🚀 Hướng dẫn chạy từng bước

### Bước 0 — Chuẩn bị dataset

```bash
# Đặt ảnh vải ren vào data/raw/, sau đó:
python preprocess.py --size 256

# Kết quả: 960 ảnh → data/processed/ (256×256, normalized)
```

**Nguồn dataset:**
- [DTD - Describable Textures Dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/) — lọc "lacy"
- [Kaggle Fabric Texture](https://www.kaggle.com/datasets) — tìm "lace fabric"

### Bước 1 — Train GAN

```bash
python stage1_gan/train.py --config config.yaml

# Resume sau khi bị interrupt:
python stage1_gan/train.py --config config.yaml --resume

# Monitor với TensorBoard:
tensorboard --logdir logs/gan   # Mở http://localhost:6006
```

**Kết quả sau khi train:** `checkpoints/gan/best_generator.pth`

### Bước 2 — Fine-tune LoRA

```bash
python train_lora.py --config config.yaml
# Kết quả: checkpoints/lora/lace_lora/
```

### Bước 3 — Sinh ảnh (Inference)

```bash
# Full pipeline (GAN → Diffusion)
python inference/pipeline.py --config config.yaml --num_images 4 --style default

# Các style khác:
python inference/pipeline.py --config config.yaml --style vintage
python inference/pipeline.py --config config.yaml --style floral
python inference/pipeline.py --config config.yaml --style bridal

# Chỉ GAN (nhanh hơn, không cần SD):
python inference/pipeline.py --config config.yaml --skip_diffusion
```

**Styles có sẵn:**

| Style | Prompt tương ứng |
|---|---|
| `default` | white lace fabric, intricate floral pattern... |
| `vintage` | antique ivory lace, vintage scalloped border... |
| `modern` | modern black lace, geometric mesh pattern... |
| `geometric` | geometric lace pattern, symmetric mesh... |
| `floral` | delicate floral lace, rose motifs... |
| `bridal` | bridal white lace, elegant scallop edges... |

### Bước 4 — Đánh giá

```bash
python inference/evaluate.py \
    --real_dir data/processed \
    --gan_dir outputs/gan_generated \
    --refined_dir outputs/final \
    --prompt "white lace fabric with intricate floral pattern, ultra detailed"
```

### Bước 5 — Demo web

```bash
python app.py
# Truy cập: http://localhost:7860
# Public URL (ngrok): python app.py --share
```

---

## 📊 Kết quả thực nghiệm

### Metrics đánh giá

| Metric | Mô tả | GAN (Stage 1) | GAN + SD (Stage 2) |
|---|---|---|---|
| **FID** ↓ | Fréchet Inception Distance — khoảng cách phân phối với ảnh thật | 359.53 | 359.53* |
| **CLIP Score** ↑ | Độ tương đồng ảnh–text (thang 0–100) | 21.45 | 21.51 |

> ⚠️ **Lưu ý về FID:** FID cần ≥ 2048 ảnh để tin cậy thống kê — với dataset nhỏ (< 10 ảnh test), FID không phản ánh đúng chất lượng. Kết quả visual tốt hơn rõ thấy dù FID không đổi.

### Quá trình train GAN

```
Epoch 1–100:    G loss cao, D loss thấp → D học nhanh hơn G
Epoch 100–400:  G bắt đầu sinh ảnh có cấu trúc
Epoch 400–600:  Ổn định, ảnh có texture vải ren
Epoch 600–800:  Mode collapse nhẹ — chất lượng giảm đôi chút
Best checkpoint: ~epoch 400–500
```

### So sánh Visual

```
Input noise z    │  Stage 1 (GAN)      │  Stage 2 (GAN+SD)
─────────────────┼─────────────────────┼──────────────────
Random noise     │ Texture cơ bản      │ Chi tiết sắc nét
                 │ Đôi khi mờ         │ Họa tiết rõ ràng
                 │ ~0.1s/ảnh          │ ~30s/ảnh
```

---

## ⚙️ Cấu hình quan trọng (config.yaml)

```yaml
data:
  image_size: 256       # Độ phân giải training (256 hoặc 512)
  batch_size: 4         # Giảm nếu OOM

stage1_gan:
  latent_dim: 512       # Chiều noise vector
  epochs: 800           # Số epoch
  lr_g: 0.00003         # LR Generator
  lr_d: 0.00003         # LR Discriminator
  beta1: 0.5            # Adam β₁
  loss_type: "lsgan"    # lsgan | bce | hinge
  lambda_gp: 5.0        # Gradient penalty (chỉ dùng WGAN-GP)

stage2_diffusion:
  model_id: "runwayml/stable-diffusion-v1-5"
  strength: 0.75        # 0=giữ GAN, 1=bỏ hoàn toàn GAN
  guidance_scale: 9.0   # Mức tuân theo prompt (7–15)
  num_inference_steps: 30
  enable_cpu_offload: true  # Bật nếu VRAM < 10GB
```

---

## 💡 Hướng cải tiến trong tương lai

### 1. StyleGAN2-ADA (Anti Data Augmentation)
Thiết kế cho dataset nhỏ — giải quyết mode collapse ở Stage 1:
```bash
# Thay Generator/Discriminator hiện tại bằng StyleGAN2-ADA
# Hỗ trợ training với < 1000 ảnh
```

### 2. ControlNet thay img2img
Dùng canny edge từ GAN để guide Diffusion — kiểm soát cấu trúc tốt hơn:
```python
canny_edges = extract_canny(gan_image)
result = controlnet(image=canny_edges, prompt="lace fabric...")
```

### 3. SD XL Turbo (Distilled)
Giảm thời gian inference từ ~30s xuống ~3s:
```python
model_id = "stabilityai/sdxl-turbo"
# Chỉ cần 1–4 steps thay vì 30
```

### 4. FID đáng tin cậy
Sinh ≥ 2048 ảnh synthetic để tính FID có ý nghĩa thống kê.

---

## 📚 Tài liệu tham khảo

| Paper | Năm | Liên quan |
|---|---|---|
| Goodfellow et al. — *Generative Adversarial Networks* | 2014 | Nền tảng GAN |
| Mao et al. — *Least Squares GAN* | 2017 | Hàm loss Stage 1 |
| Ho et al. — *DDPM* | 2020 | Nền tảng Diffusion |
| Rombach et al. — *Latent Diffusion / Stable Diffusion* | 2022 | Stage 2 backbone |
| Hu et al. — *LoRA* | 2021 | Fine-tune Stage 2 |
| Radford et al. — *CLIP* | 2021 | Text encoder + evaluation metric |
| Karras et al. — *StyleGAN2-ADA* | 2020 | Hướng cải tiến Stage 1 |
