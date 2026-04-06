# 🧵 Sinh Tổng Hợp Vải Ren — GAN + Stable Diffusion

> **Đề tài:** Synthetic Fabric Texture Generation Using GAN and Stable Diffusion  
> **Môn học:** Machine Learning  
> **Nhóm thực hiện:** haluu07

Pipeline 2 giai đoạn để tổng hợp ảnh vải ren chất lượng cao:

- **Giai đoạn 1 — GAN**: Mạng đối sinh học cấu trúc vải ren từ noise ngẫu nhiên
- **Giai đoạn 2 — Stable Diffusion + LoRA**: Mô hình khuếch tán tinh chỉnh kết quả GAN thành ảnh photorealistic

---

## 🎯 Mục tiêu dự án

Vải ren là loại vật liệu có cấu trúc phức tạp (lỗ lưới, họa tiết hoa văn, sợi chỉ mảnh) — rất khó tổng hợp bằng một mô hình đơn lẻ. Dự án giải quyết bằng cách kết hợp:

1. **GAN** sinh ra cấu trúc tổng thể nhanh
2. **Stable Diffusion** tinh chỉnh chi tiết cấp độ sợi chỉ dựa trên text prompt

---

## 🧠 Thuật toán sử dụng

### Giai đoạn 1: GAN (Generative Adversarial Network)

#### Ý tưởng cơ bản

GAN bao gồm 2 mạng neural đối lập nhau cùng học song song:

```
Generator (G)                    Discriminator (D)
─────────────────────            ─────────────────────
Nhận: noise vector z             Nhận: ảnh (thật hoặc fake)
Sinh: ảnh vải ren giả            Phán đoán: thật hay giả?
Mục tiêu: qua mặt D             Mục tiêu: không bị qua mặt

    Noise z ──→ [G] ──→ Ảnh giả ──→ [D] ──→ Xác suất
                                ↑              ↑
                          Ảnh thật ────────────┘
```

Ví dụ trực quan: G giống thợ làm tiền giả, D giống cảnh sát — cả hai cùng giỏi lên qua mỗi epoch.

#### Kiến trúc Generator

```
Input: Noise z (512 chiều)
         │
    Linear → 4×4×1024
         │ ConvTranspose2d + BatchNorm + ReLU
    8×8×512
         │ ConvTranspose2d + BatchNorm + ReLU  
    16×16×256
         │ ConvTranspose2d + BatchNorm + ReLU
    32×32×128
         │ ConvTranspose2d + BatchNorm + ReLU
    64×64×64
         │ ConvTranspose2d + BatchNorm + ReLU
    128×128×32
         │ ConvTranspose2d
    256×256×3
         │ Tanh (normalize về [-1, 1])
    Output: Ảnh 256×256 RGB
```

#### Kiến trúc Discriminator

```
Input: Ảnh 256×256 RGB
         │ Conv2d + LeakyReLU(0.2)
    128×128×64
         │ Conv2d + BatchNorm + LeakyReLU
    64×64×128
         │ Conv2d + BatchNorm + LeakyReLU
    32×32×256
         │ Conv2d + BatchNorm + LeakyReLU
    16×16×512
         │ Conv2d + BatchNorm + LeakyReLU
    8×8×1024
         │ Flatten → Linear
    Output: Xác suất [0, 1]
```

#### Hàm Loss: LSGAN (Least Squares GAN)

Dự án sử dụng **LSGAN** thay vì vanilla GAN để tránh vanishing gradient:

```
Loss Discriminator:
  L_D = E[(D(x) - 1)²] + E[(D(G(z)))²]
         ─────────────   ──────────────
         Ảnh thật → 1    Ảnh giả → 0

Loss Generator:
  L_G = E[(D(G(z)) - 1)²]
         ─────────────────
        Cố gắng để D cho điểm 1 (thật)
```

**Tại sao chọn LSGAN?** Dataset nhỏ (960 ảnh) → vanilla GAN dễ bị NaN gradient, LSGAN ổn định hơn nhiều.

#### Optimizer

- **Adam** với β₁ = 0.5, β₂ = 0.999
- Learning rate: 3×10⁻⁵ (thấp để tránh mode collapse)
- Batch size: 4 (giới hạn VRAM)

---

### Giai đoạn 2: Stable Diffusion + LoRA

#### Ý tưởng: Diffusion Process

Stable Diffusion học cách **đảo ngược quá trình thêm nhiễu**:

```
FORWARD (thêm noise — training):
  Ảnh thật x₀ → x₁ → x₂ → ... → xₜ (pure noise)

REVERSE (khử noise — inference):
  Pure noise xₜ → xₜ₋₁ → ... → x₁ → x₀ (ảnh tạo ra)
```

Mỗi bước khử noise, mạng UNet dự đoán lượng noise cần loại bỏ, có điều kiện theo text prompt.

#### Latent Diffusion

Stable Diffusion không làm việc trực tiếp trên pixel — dùng không gian latent để tiết kiệm VRAM:

```
Ảnh gốc 512×512×3
       │ VAE Encoder (nén 8x)
  Latent 64×64×4   ← UNet làm việc ở đây
       │ VAE Decoder (giải nén)
  Ảnh cuối 512×512×3
```

#### UNet với Cross-Attention (Text Conditioning)

```
Text Prompt: "white lace fabric with floral pattern"
                    │
              CLIP Text Encoder
                    │
              [token₁, token₂, ..., tokenₙ]  ← Key (K), Value (V)
                    │
                    ↓ Cross-Attention
Latent Image ──→ Query (Q) ──→ Attention(Q,K,V) ──→ Feature cập nhật
```

Cơ chế Cross-Attention cho phép mỗi vùng của ảnh "hỏi" xem nó liên quan nhất đến từ nào trong prompt, rồi cập nhật feature tương ứng.

#### LoRA (Low-Rank Adaptation)

Fine-tune Stable Diffusion trên dataset vải ren mà không cần train toàn bộ mô hình:

```
Bình thường:  W' = W + ΔW    (ΔW có hàng tỉ parameters)

Với LoRA:     W' = W + A×B
              A: (d × r),  B: (r × d),  r = 16 (rank)
              
              Thay vì tune hàng tỉ params:
              Chỉ tune A và B → tiết kiệm 99% bộ nhớ
```

LoRA được inject vào các ma trận Q, K, V của Cross-Attention trong UNet, dạy mô hình rằng prompt "lace fabric" tương ứng với texture vải ren cụ thể trong dataset.

#### img2img Pipeline

Thay vì sinh từ noise hoàn toàn, dự án dùng **img2img**: bắt đầu từ ảnh GAN đã có cấu trúc:

```
Ảnh GAN (256×256)
       │ VAE Encode
  Latent GAN
       │ Thêm noise một phần (strength = 0.75)
  Latent nhiễu một phần
       │ UNet denoise (30 steps) + text guidance
  Latent refined
       │ VAE Decode
  Ảnh cuối (256×256)
```

**Strength = 0.75**: 75% là thông tin mới từ diffusion, 25% giữ lại cấu trúc từ GAN.

---

### Tại sao kết hợp GAN + Diffusion?

| Tiêu chí | GAN | Diffusion | GAN + Diffusion |
|---|---|---|---|
| Tốc độ inference | ✅ Rất nhanh (~0.1s) | ❌ Chậm (~30s) | ⚠️ Trung bình |
| Cấu trúc tổng thể | ✅ Ổn định | ❌ Không ổn định | ✅ |
| Chi tiết tinh tế | ❌ Mờ | ✅ Rất sắc nét | ✅ |
| Dataset nhỏ | ❌ Mode collapse | ✅ LoRA giải quyết | ✅ |
| Kiểm soát style | ❌ Khó | ✅ Text prompt | ✅ |

**Kết luận:** GAN cung cấp nền cấu trúc vải ren nhanh và ổn định → Diffusion tận dụng làm điểm khởi đầu để tinh chỉnh chi tiết với text prompt, tốt hơn so với dùng từng mô hình riêng lẻ.

---

## 📁 Cấu trúc dự án

```
lace-gan-diffusion/
├── config.yaml                  # File cấu hình trung tâm (chỉnh ở đây trước)
├── requirements.txt             # Danh sách thư viện cần cài
├── preprocess.py                # Tiền xử lý dataset (resize, normalize)
├── train_lora.py                # Script huấn luyện LoRA cho Stage 2
├── app.py                       # Giao diện web Gradio để demo
├── plot_clip_score.py           # Script vẽ biểu đồ so sánh CLIP score
│
├── stage1_gan/                  # Giai đoạn 1: GAN
│   ├── dataset.py               # LaceDataset + DataLoader + augmentation
│   ├── model.py                 # Generator + Discriminator
│   ├── losses.py                # LSGAN loss + Gradient Penalty + Perceptual loss
│   └── train.py                 # Training loop (checkpoint, logging, TensorBoard)
│
├── stage2_diffusion/            # Giai đoạn 2: Stable Diffusion
│   ├── refiner.py               # SD img2img pipeline (tối ưu bộ nhớ)
│   └── prompts.py               # Mapping style → text prompt
│
├── inference/                   # Inference & đánh giá
│   ├── pipeline.py              # Pipeline end-to-end: Stage 1 → Stage 2
│   ├── evaluate.py              # Tính FID score + CLIP score + ảnh so sánh
│   └── compute_fid.py           # Tính FID riêng lẻ
│
├── data/
│   ├── raw/                     # Ảnh gốc (đặt dataset vào đây)
│   └── processed/               # Ảnh sau tiền xử lý (tự động tạo)
│
├── checkpoints/
│   ├── gan/                     # Checkpoint GAN (tự động lưu khi train)
│   └── lora/                    # Checkpoint LoRA (tự động lưu khi train)
│
├── outputs/
│   ├── gan_generated/           # Ảnh Stage 1 (GAN only)
│   ├── refined/                 # Ảnh Stage 2 (Diffusion only)
│   └── final/                   # Ảnh cuối (full pipeline)
│
└── logs/                        # TensorBoard logs
    └── gan/
```

---

## ⚙️ Cài đặt môi trường

### 1. Cài thư viện

```bash
pip install -r requirements.txt
```

Với server có CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. Chuẩn bị dataset

Đặt ảnh vải ren vào `data/raw/`, sau đó chạy:

```bash
python preprocess.py --size 256
```

**Nguồn dataset:**
- [DTD - Describable Textures Dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/) — lọc danh mục "lacy"
- Kaggle: tìm "lace fabric texture"
- Google Images / Pinterest

---

## 🚀 Hướng dẫn chạy

### Bước 1 — Train GAN (Giai đoạn 1)

```bash
python stage1_gan/train.py --config config.yaml
```

Tiếp tục từ checkpoint:
```bash
python stage1_gan/train.py --config config.yaml --resume
```

Theo dõi quá trình train:
```bash
tensorboard --logdir logs/gan
```

### Bước 2 — Fine-tune LoRA (Giai đoạn 2)

```bash
python train_lora.py --config config.yaml
```

### Bước 3 — Chạy pipeline sinh ảnh

```bash
# Mặc định
python inference/pipeline.py --config config.yaml --num_images 4

# Chọn style
python inference/pipeline.py --config config.yaml --style vintage --num_images 4

# Chỉ GAN (không qua Diffusion)
python inference/pipeline.py --config config.yaml --skip_diffusion
```

**Các style hỗ trợ:** `default`, `vintage`, `modern`, `geometric`, `floral`, `bridal`

### Bước 4 — Đánh giá kết quả

```bash
python inference/evaluate.py \
    --real_dir data/processed \
    --gan_dir outputs/gan_generated \
    --refined_dir outputs/final \
    --prompt "white lace fabric with intricate floral pattern"
```

### Bước 5 — Chạy giao diện web demo

```bash
python app.py
# Truy cập: http://localhost:7860
```

---

## 📊 Kết quả thực nghiệm

| Metric | GAN (Stage 1) | GAN + Diffusion (Stage 2) |
|---|---|---|
| FID ↓ (thấp hơn = tốt hơn) | 359.53 | 359.53* |
| CLIP Score ↑ | 21.45 | 21.51 |

> *FID trên tập nhỏ (< 10 ảnh) không đủ tin cậy về mặt thống kê. Cần ≥ 2048 ảnh để FID có ý nghĩa.

**Nhận xét:** Mặc dù FID chưa cải thiện đáng kể (do dataset nhỏ và mode collapse ở Stage 1), ảnh từ pipeline 2 giai đoạn cho chất lượng visual tốt hơn rõ rệt — sắc nét hơn, texture chi tiết hơn.

---

## ⚙️ Cấu hình quan trọng (config.yaml)

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `data.image_size` | 256 | Độ phân giải training |
| `data.batch_size` | 4 | Giảm nếu hết VRAM |
| `stage1_gan.epochs` | 800 | Số epoch train GAN |
| `stage1_gan.loss_type` | lsgan | Hàm loss: lsgan / bce / hinge |
| `stage1_gan.lr_g` | 3e-5 | Learning rate Generator |
| `stage1_gan.lr_d` | 3e-5 | Learning rate Discriminator |
| `stage2_diffusion.strength` | 0.75 | Mức độ ảnh hưởng của Diffusion (0=giữ GAN, 1=bỏ GAN) |
| `stage2_diffusion.guidance_scale` | 9.0 | Mức độ tuân theo text prompt (7–15) |
| `stage2_diffusion.num_inference_steps` | 30 | Số bước khử noise (nhiều hơn = chất lượng cao hơn) |
| `stage2_diffusion.enable_cpu_offload` | true | Bật nếu VRAM < 8GB |

---

## 💡 Hướng cải tiến

### 1. StyleGAN2-ADA
Thay thế GAN hiện tại bằng StyleGAN2-ADA — được thiết kế riêng cho dataset nhỏ, giảm mode collapse đáng kể.

### 2. ControlNet
Dùng canny edge từ ảnh GAN để guide Diffusion — kiểm soát cấu trúc tốt hơn img2img thuần túy.

### 3. SD XL Turbo
Thay SD 1.5 bằng SD XL Turbo — chỉ cần 1–4 bước inference thay vì 30, nhanh hơn ~10x.

---

## 📋 Yêu cầu hệ thống

| | Tối thiểu | Khuyến nghị |
|---|---|---|
| Python | ≥ 3.9 | 3.10+ |
| PyTorch | ≥ 2.0 | 2.1+ |
| VRAM GPU | 8 GB | 16–24 GB |
| RAM | 16 GB | 32 GB |
| GPU | GTX 1080 Ti | RTX 3090 / A4000 |
| CUDA | 11.8 | 12.1 |

---

## 📚 Tài liệu tham khảo

- Goodfellow et al. (2014) — *Generative Adversarial Networks*
- Mao et al. (2017) — *Least Squares GAN*
- Ho et al. (2020) — *Denoising Diffusion Probabilistic Models (DDPM)*
- Rombach et al. (2022) — *High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)*
- Hu et al. (2021) — *LoRA: Low-Rank Adaptation of Large Language Models*
- Radford et al. (2021) — *CLIP: Learning Transferable Visual Models From Natural Language Supervision*
