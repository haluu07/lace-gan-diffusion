"""
app.py — Gradio Web Demo for Lace GAN + Stable Diffusion Pipeline.

Run:
    python app.py

Then open browser at:
    http://localhost:7860
    (or via SSH tunnel: ssh -L 7860:localhost:7860 qtech@<SERVER_IP>)
"""

import sys
import random
import argparse
from pathlib import Path

import yaml
import torch
import numpy as np
import gradio as gr
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent))

# ──────────────────────────────────────────────────────────────────
#  Load config & pipeline
# ──────────────────────────────────────────────────────────────────

CFG_PATH = "config.yaml"

def load_cfg():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)

cfg = load_cfg()

# Lazy-load pipeline to avoid OOM on startup
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from inference.pipeline import run_pipeline
        _pipeline = run_pipeline
    return _pipeline


# ──────────────────────────────────────────────────────────────────
#  Core generation function
# ──────────────────────────────────────────────────────────────────

def generate_lace(
    num_images: int,
    style: str,
    strength: float,
    guidance_scale: float,
    seed: int,
    skip_diffusion: bool,
):
    """Called by Gradio on every button click."""
    import subprocess

    # Determine best available checkpoint
    ckpt_dir = Path(cfg["stage1_gan"]["checkpoint_dir"])
    for name in ["epoch_0400.pth", "epoch_0500.pth", "epoch_0300.pth", "best_generator.pth", "latest.pth"]:
        ckpt = ckpt_dir / name
        if ckpt.exists():
            break

    # Write strength/guidance into config temporarily
    import copy, tempfile
    tmp_cfg = copy.deepcopy(cfg)
    tmp_cfg["stage2_diffusion"]["strength"]       = strength
    tmp_cfg["stage2_diffusion"]["guidance_scale"] = guidance_scale
    tmp_path = "config_tmp.yaml"
    with open(tmp_path, "w") as f:
        yaml.dump(tmp_cfg, f)

    cmd = [
        sys.executable, "inference/pipeline.py",
        "--config",     tmp_path,
        "--num_images", str(int(num_images)),
        "--style",      style,
        "--seed",       str(int(seed)),
        "--checkpoint", str(ckpt),
    ]
    if skip_diffusion:
        cmd.append("--skip_diffusion")

    import shutil

    # Đọc đúng path từ config — tránh hardcode sai folder
    if skip_diffusion:
        output_dir = Path(cfg["inference"]["output_gan_dir"])   # outputs/gan_generated
    else:
        output_dir = Path(cfg["inference"]["output_final_dir"]) # outputs/final

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={
            **__import__('os').environ,
            "CUDA_VISIBLE_DEVICES":        __import__('os').environ.get("CUDA_VISIBLE_DEVICES", "1"),
            "PYTORCH_CUDA_ALLOC_CONF":     "expandable_segments:True",
            "PYTORCH_NO_CUDA_MEMORY_CACHING": "1",
        }
    )
    if result.returncode != 0:
        err = result.stderr[-1200:] if result.stderr else "unknown error"
        return [], f"❌ Pipeline error:\n{err}"

    # Collect outputs
    imgs = sorted(output_dir.glob("*.png")) + sorted(output_dir.glob("*.jpg"))

    if not imgs:
        dbg = result.stderr[-600:] if result.stderr else "(no stderr)"
        return [], f"❌ No images found in {output_dir}/\n\nDebug:\n{dbg}"

    pil_imgs = [Image.open(p) for p in imgs]
    status = (
        f"✅ {len(pil_imgs)} image(s) | Style: {style} | "
        f"Strength: {strength} | Checkpoint: {ckpt.name} | Seed: {seed}"
    )
    return pil_imgs, status


# ──────────────────────────────────────────────────────────────────
#  Gradio UI
# ──────────────────────────────────────────────────────────────────

STYLES = ["default", "vintage", "modern", "geometric", "floral", "bridal"]

with gr.Blocks(
    title="Lace Fabric Generator",
    theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
    css="""
        .title { text-align: center; margin-bottom: 0.5em; }
        .subtitle { text-align: center; color: #888; margin-bottom: 1.5em; }
        footer { display: none !important; }
    """
) as demo:

    gr.HTML("""
        <h1 class='title'>🎨 Lace Fabric Generator</h1>
        <p class='subtitle'>Two-stage pipeline: GAN structure → Stable Diffusion + LoRA refinement</p>
    """)

    with gr.Row():
        # ── Left: Controls ──────────────────────────────────────
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### ⚙️ Settings")

            num_images = gr.Slider(
                label="Number of images", minimum=1, maximum=8,
                value=4, step=1
            )
            style = gr.Dropdown(
                label="Style", choices=STYLES, value="default"
            )
            strength = gr.Slider(
                label="SD Strength (0=keep GAN | 1=ignore GAN)",
                minimum=0.1, maximum=1.0, value=0.75, step=0.05
            )
            guidance = gr.Slider(
                label="Guidance Scale",
                minimum=4.0, maximum=15.0, value=9.0, step=0.5
            )
            seed = gr.Number(
                label="Seed (−1 = random)", value=42, precision=0
            )
            skip_sd = gr.Checkbox(
                label="Skip Diffusion (GAN only, faster)", value=False
            )

            with gr.Row():
                btn_gen = gr.Button("🎨 Generate", variant="primary")
                btn_rnd = gr.Button("🎲 Random Seed")

            status_box = gr.Textbox(label="Status", interactive=False, lines=2)

        # ── Right: Gallery ──────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### 🖼️ Generated Images")
            gallery = gr.Gallery(
                label="Results",
                show_label=False,
                elem_id="gallery",
                columns=4,
                height=480,
                object_fit="cover",
            )

    # Examples
    gr.Markdown("---")
    gr.Markdown("### 💡 Try these settings")
    gr.Examples(
        examples=[
            [4, "floral",    0.75, 9.0,  42,  False],
            [4, "vintage",   0.75, 9.0,  7,   False],
            [4, "geometric", 0.60, 8.5,  2024, False],
            [4, "bridal",    0.80, 10.0, 123, False],
            [4, "default",   0.10, 7.0,  42,  True],   # GAN only
        ],
        inputs=[num_images, style, strength, guidance, seed, skip_sd],
    )

    # ── Event handlers ──────────────────────────────────────────
    def random_seed():
        return random.randint(0, 9999)

    btn_rnd.click(fn=random_seed, outputs=seed)
    btn_gen.click(
        fn=generate_lace,
        inputs=[num_images, style, strength, guidance, seed, skip_sd],
        outputs=[gallery, status_box],
        show_progress="full",
    )


# ──────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",  default="0.0.0.0",  help="Bind host")
    parser.add_argument("--port",  default=7860, type=int)
    parser.add_argument("--share", action="store_true", help="Gradio public URL")
    args = parser.parse_args()

    print(f"\n{'━'*55}")
    print(f"  🎨  Lace Fabric Generator — Gradio Demo")
    print(f"  Local  : http://localhost:{args.port}")
    if args.share:
        print(f"  Public : (Gradio will print URL below)")
    print(f"{'━'*55}\n")

    demo.launch(
        server_name = args.host,
        server_port = args.port,
        share       = args.share,
        show_error  = True,
    )
