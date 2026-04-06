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

```mermaid
flowchart LR
    A([🎲 Noise z\n512-dim]) --> B

    subgraph B["⚙️ Stage 1 — GAN"]
        direction TB
        G["🔧 Generator G"] --> D["🔍 Discriminator D"]
        REAL["🖼️ Ảnh thật\nDataset"] --> D
        D --> LOSS["📉 LSGAN Loss"]
        LOSS --> G
    end

    B --> C["🖼️ Ảnh vải ren thô\n256×256"]

    subgraph SD["✨ Stage 2 — Stable Diffusion + LoRA"]
        direction TB
        VAE_E["🗜️ VAE Encoder"] --> UNET["🧠 UNet Denoiser\n30 steps"]
        TEXT["📝 Text Prompt"] --> CLIP["🔤 CLIP Encoder"] --> UNET
        UNET --> VAE_D["🔍 VAE Decoder"]
    end

    C --> SD
    VAE_D --> OUT(["✅ Ảnh chất lượng cao\n256×256"])

    style A fill:#6366f1,color:#fff,stroke:none
    style C fill:#f59e0b,color:#fff,stroke:none
    style OUT fill:#10b981,color:#fff,stroke:none
    style B fill:#ede9fe,stroke:#7c3aed
    style SD fill:#ecfdf5,stroke:#059669
```

---

## 🧠 Thuật toán chi tiết

### STAGE 1 — GAN

#### Kiến trúc Generator

```mermaid
flowchart TD
    Z(["🎲 Noise z\n512-dim"]) --> L["Linear\n512 → 16384"]
    L --> R["Reshape\n1024 × 4 × 4"]

    R --> T1["ConvTranspose2d\n1024→512 | BN | ReLU\n8×8"]
    T1 --> T2["ConvTranspose2d\n512→256 | BN | ReLU\n16×16"]
    T2 --> T3["ConvTranspose2d\n256→128 | BN | ReLU\n32×32"]
    T3 --> T4["ConvTranspose2d\n128→64 | BN | ReLU\n64×64"]
    T4 --> T5["ConvTranspose2d\n64→32 | BN | ReLU\n128×128"]
    T5 --> T6["ConvTranspose2d\n32→3 | Tanh\n256×256"]

    T6 --> OUT(["🖼️ Ảnh giả 256×256×3"])

    style Z fill:#6366f1,color:#fff,stroke:none
    style OUT fill:#f59e0b,color:#fff,stroke:none
    style T1 fill:#ddd6fe,stroke:#7c3aed
    style T2 fill:#ddd6fe,stroke:#7c3aed
    style T3 fill:#ddd6fe,stroke:#7c3aed
    style T4 fill:#ddd6fe,stroke:#7c3aed
    style T5 fill:#ddd6fe,stroke:#7c3aed
    style T6 fill:#ddd6fe,stroke:#7c3aed
```

#### Kiến trúc Discriminator

```mermaid
flowchart TD
    IN(["🖼️ Ảnh 256×256×3\n(thật hoặc giả)"]) --> C1

    C1["Conv2d 3→64\nLeakyReLU(0.2)\n128×128"]
    C1 --> C2["Conv2d 64→128\nBN + LeakyReLU\n64×64"]
    C2 --> C3["Conv2d 128→256\nBN + LeakyReLU\n32×32"]
    C3 --> C4["Conv2d 256→512\nBN + LeakyReLU\n16×16"]
    C4 --> C5["Conv2d 512→1024\nBN + LeakyReLU\n8×8"]
    C5 --> FL["Flatten → Linear"]
    FL --> OUT(["🎯 Xác suất [0,1]\nThật / Giả"])

    style IN fill:#f59e0b,color:#fff,stroke:none
    style OUT fill:#ef4444,color:#fff,stroke:none
    style C1 fill:#fef3c7,stroke:#d97706
    style C2 fill:#fef3c7,stroke:#d97706
    style C3 fill:#fef3c7,stroke:#d97706
    style C4 fill:#fef3c7,stroke:#d97706
    style C5 fill:#fef3c7,stroke:#d97706
```

#### Quá trình đối kháng GAN

```mermaid
flowchart LR
    Z(["🎲 Noise z"]) --> G["Generator G"]
    G --> FAKE["🖼️ Ảnh giả"]
    REAL["🖼️ Ảnh thật\nDataset"] --> D
    FAKE --> D["Discriminator D"]

    D --> LD["📉 Loss D\nE[(D(x)−1)²]\n+ E[(D(G(z)))²]"]
    D --> LG["📉 Loss G\nE[(D(G(z))−1)²]"]

    LD -->|Cập nhật| D
    LG -->|Cập nhật| G

    style Z fill:#6366f1,color:#fff,stroke:none
    style REAL fill:#10b981,color:#fff,stroke:none
    style FAKE fill:#f59e0b,color:#fff,stroke:none
    style LD fill:#fee2e2,stroke:#dc2626
    style LG fill:#ede9fe,stroke:#7c3aed
```

---

### STAGE 2 — Stable Diffusion + LoRA

#### Kiến trúc tổng thể Stage 2

```mermaid
flowchart TD
    PROM(["📝 Text Prompt\n'white lace fabric...'"]) --> CLIP["🔤 CLIP Text Encoder\n(frozen)"]
    CLIP --> EMB["Text Embeddings\n77 tokens × 768 dim"]

    GAN(["🖼️ Ảnh GAN\n256×256"]) --> VAEE["🗜️ VAE Encoder"]
    VAEE --> LAT["Latent\n32×32×4"]
    LAT --> NOISE["➕ Thêm noise\nstrength=0.75"]

    subgraph UNET["🧠 UNet Denoiser (30 steps)"]
        direction TB
        ENC["Encoder Blocks\nResBlock + CrossAttention\n32→16→8"]
        MID["Bottleneck\nResBlock + CrossAttention\n8×8"]
        DEC["Decoder Blocks\nResBlock + CrossAttention\n8→16→32"]
        ENC --> MID --> DEC
    end

    NOISE --> UNET
    EMB -->|"Cross-Attention\n(LoRA injected here)"| UNET

    UNET --> VAED["🔍 VAE Decoder"]
    VAED --> OUT(["✅ Ảnh tinh chỉnh\n256×256"])

    style PROM fill:#3b82f6,color:#fff,stroke:none
    style GAN fill:#f59e0b,color:#fff,stroke:none
    style OUT fill:#10b981,color:#fff,stroke:none
    style UNET fill:#ecfdf5,stroke:#059669
    style EMB fill:#dbeafe,stroke:#3b82f6
```

#### Cross-Attention: Cơ chế text điều khiển ảnh

```mermaid
flowchart LR
    subgraph TEXT["📝 Text Tokens"]
        T1["white"] 
        T2["lace"]
        T3["fabric"]
        T4["floral"]
    end

    subgraph KV["Key & Value (từ text)"]
        K["K = W_k × text"]
        V["V = W_v × text"]
    end

    subgraph PIX["🖼️ Latent Pixels"]
        P1["Vùng texture"]
        P2["Vùng họa tiết"]
    end

    subgraph Q["Query (từ ảnh)"]
        Q1["Q = W_q × pixel"]
    end

    TEXT --> KV
    PIX --> Q

    Q1 --> ATT["Attention\nScore = softmax(QKᵀ/√d)×V"]
    K --> ATT
    V --> ATT

    ATT --> OUT["Feature cập nhật\ntheo text prompt"]

    style TEXT fill:#dbeafe,stroke:#3b82f6
    style PIX fill:#fef3c7,stroke:#d97706
    style ATT fill:#ecfdf5,stroke:#059669
    style OUT fill:#10b981,color:#fff,stroke:none
```

#### LoRA — Fine-tune hiệu quả

```mermaid
flowchart LR
    subgraph BEFORE["❌ Không có LoRA"]
        direction LR
        X1["Input x"] --> W1["W_q\n768×768\n589K params\nfrozen"]
        W1 --> Q1["Q"]
    end

    subgraph AFTER["✅ Có LoRA"]
        direction LR
        X2["Input x"] --> W2["W_q\nfrozen"]
        X2 --> A["A\n768×16\n12K params"]
        A --> B["B\n16×768\n12K params"]
        W2 --> ADD(("＋"))
        B --> ADD
        ADD --> Q2["Q"]
    end

    style BEFORE fill:#fee2e2,stroke:#dc2626
    style AFTER fill:#ecfdf5,stroke:#059669
    style A fill:#a7f3d0,stroke:#059669
    style B fill:#a7f3d0,stroke:#059669
```

---

## 📊 Luồng dữ liệu (Data Flow)

```mermaid
flowchart TD
    DS(["📁 Dataset\n960 ảnh vải ren"]) --> PRE["⚙️ preprocess.py\nResize 256×256\nNormalize [-1,1]"]
    PRE --> PROC["📁 data/processed/"]

    PROC --> TRGAN["🏋️ Train GAN\n800 epochs\nLSGAN loss"]
    PROC --> TRLORA["🏋️ Train LoRA\n100 epochs\nAdaLoRA rank=16"]

    TRGAN --> CK1["💾 checkpoints/gan/\nbest_generator.pth"]
    TRLORA --> CK2["💾 checkpoints/lora/\nlace_lora/"]

    CK1 --> INF["🚀 Inference\npipeline.py"]
    CK2 --> INF

    INF --> GAN_OUT["🖼️ outputs/gan_generated/\nGAN only"]
    INF --> FINAL["🖼️ outputs/final/\nGAN + SD"]

    GAN_OUT --> EVAL["📊 evaluate.py"]
    FINAL --> EVAL
    PROC --> EVAL

    EVAL --> FID["📉 FID Score"]
    EVAL --> CLIP["📈 CLIP Score"]
    EVAL --> VIZ["🖼️ Ảnh so sánh"]

    style DS fill:#6366f1,color:#fff,stroke:none
    style FINAL fill:#10b981,color:#fff,stroke:none
    style FID fill:#f59e0b,color:#fff,stroke:none
    style CLIP fill:#f59e0b,color:#fff,stroke:none
```

---

## 📁 Cấu trúc dự án

```
lace-gan-diffusion/
├── 📄 config.yaml              # Cấu hình trung tâm (LR, epochs, paths...)
├── 📄 requirements.txt         # Danh sách thư viện
├── 📄 preprocess.py            # Resize + normalize dataset
├── 📄 train_lora.py            # Fine-tune LoRA trên SD 1.5
├── 📄 app.py                   # Giao diện web Gradio (demo)
├── 📄 plot_clip_score.py       # Vẽ biểu đồ so sánh CLIP score
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
│   ├── gan/                    # GAN checkpoints
│   └── lora/lace_lora/         # LoRA weights
│
└── 📁 outputs/
    ├── gan_generated/          # Ảnh Stage 1 (GAN only)
    ├── final/                  # Ảnh Stage 2 (full pipeline)
    └── gan_samples/            # Sample ảnh mỗi epoch
```

---

## ⚙️ Cài đặt & Chạy

### Cài thư viện

```bash
pip install -r requirements.txt
# CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Chạy từng bước

```bash
# 1. Tiền xử lý dataset
python preprocess.py --size 256

# 2. Train GAN
python stage1_gan/train.py --config config.yaml

# 3. Fine-tune LoRA
python train_lora.py --config config.yaml

# 4. Sinh ảnh (full pipeline)
python inference/pipeline.py --config config.yaml --style vintage --num_images 4

# 5. Đánh giá
python inference/evaluate.py --real_dir data/processed --refined_dir outputs/final

# 6. Demo web
python app.py  # http://localhost:7860
```

---

## 📊 Kết quả thực nghiệm

| Metric | GAN (Stage 1) | GAN + SD (Stage 2) | Ý nghĩa |
|---|---|---|---|
| **FID ↓** | 359.53 | 359.53* | Khoảng cách phân phối với ảnh thật |
| **CLIP Score ↑** | 21.45 | 21.51 | Độ tương đồng ảnh–text |

> ⚠️ FID cần ≥ 2048 ảnh để có ý nghĩa thống kê. Dataset nhỏ (< 10 ảnh test) khiến FID không đáng tin cậy. Visual quality cải thiện rõ thấy ở Stage 2.

---

## ⚙️ Cấu hình quan trọng

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `data.image_size` | 256 | Độ phân giải training |
| `stage1_gan.epochs` | 800 | Số epoch train GAN |
| `stage1_gan.lr_g` | 3×10⁻⁵ | Learning rate Generator |
| `stage1_gan.loss_type` | lsgan | Hàm loss |
| `stage2_diffusion.strength` | 0.75 | Mức độ Diffusion (0=giữ GAN, 1=bỏ GAN) |
| `stage2_diffusion.guidance_scale` | 9.0 | Tuân theo prompt (7–15) |
| `stage2_diffusion.num_inference_steps` | 30 | Số bước khử noise |

---

## 💡 Hướng cải tiến

| Cải tiến | Lợi ích |
|---|---|
| **StyleGAN2-ADA** | Giảm mode collapse, tốt hơn cho dataset nhỏ |
| **ControlNet** | Kiểm soát cấu trúc tốt hơn img2img |
| **SD XL Turbo** | Giảm inference từ ~30s xuống ~3s |
| **FID đáng tin** | Sinh ≥ 2048 ảnh để FID có ý nghĩa |

---

## 📚 Tài liệu tham khảo

| Paper | Năm | Liên quan |
|---|---|---|
| Goodfellow et al. — *Generative Adversarial Networks* | 2014 | Nền tảng GAN |
| Mao et al. — *Least Squares GAN* | 2017 | Hàm loss Stage 1 |
| Ho et al. — *DDPM* | 2020 | Nền tảng Diffusion |
| Rombach et al. — *Latent Diffusion (Stable Diffusion)* | 2022 | Stage 2 backbone |
| Hu et al. — *LoRA* | 2021 | Fine-tune Stage 2 |
| Radford et al. — *CLIP* | 2021 | Text encoder + evaluation |
| Karras et al. — *StyleGAN2-ADA* | 2020 | Hướng cải tiến |
