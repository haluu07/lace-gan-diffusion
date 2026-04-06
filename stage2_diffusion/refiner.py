"""
refiner.py — Stage 2: Stable Diffusion img2img refinement pipeline.

Takes a coarse GAN-generated lace image and uses SD img2img to:
  • Add thread-level texture and realistic lace holes
  • Fix GAN artifacts (blur, distortion)
  • Enhance high-frequency details guided by a text prompt

Memory optimizations bundled for 8-12 GB VRAM servers.
"""

import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from diffusers import (
    StableDiffusionImg2ImgPipeline,
    DPMSolverMultistepScheduler,
)


class LaceRefiner:
    """
    Wraps Stable Diffusion img2img for lace fabric refinement.

    Example:
        refiner = LaceRefiner()
        pil_refined = refiner.refine_image(
            image         = pil_gan_output,
            prompt        = "white lace fabric, intricate floral, detailed…",
            strength      = 0.6,
            guidance_scale= 8.5,
        )
    """

    def __init__(
        self,
        model_id: str   = "runwayml/stable-diffusion-v1-5",
        device: str     = None,
        enable_attention_slicing: bool = True,
        enable_vae_slicing:       bool = True,
        enable_cpu_offload:       bool = False,
    ):
        """
        Args:
            model_id                : HuggingFace model repo ID.
            device                  : 'cuda' | 'cpu' | None (auto).
            enable_attention_slicing: Saves ~30% VRAM, small speed cost.
            enable_vae_slicing      : Efficient batch VAE decoding.
            enable_cpu_offload      : Move model to CPU between steps
                                      (use for <8 GB VRAM — much slower).
        """
        self.device = device or ('cuda' if torch.cuda.is_available()
                                 else 'cpu')
        dtype = (torch.float16 if self.device == 'cuda'
                 else torch.float32)

        print(f"[Refiner] Loading '{model_id}' …")
        print(f"[Refiner] Device={self.device} | dtype={dtype}")

        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            # Disable safety checker to save VRAM
            safety_checker=None,
            requires_safety_checker=False,
        )

        # DPM++ 2M: fast, high quality, fewer steps needed
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            use_karras_sigmas=True,
        )

        # ── Memory optimizations ─────────────────────────────
        if enable_cpu_offload:
            # Sequential offload: only 1 sub-model on GPU at a time
            # Needs only ~500MB VRAM — slowest but most memory efficient
            self.pipe.enable_sequential_cpu_offload()
            self._cpu_offload = True
            print("[Refiner] ✓ Sequential CPU offload ON (~500MB VRAM)")
        else:
            self._cpu_offload = False
            self.pipe = self.pipe.to(self.device)
            if enable_attention_slicing:
                self.pipe.enable_attention_slicing()
                print("[Refiner] ✓ Attention slicing ON")
            if enable_vae_slicing:
                self.pipe.enable_vae_slicing()
                print("[Refiner] ✓ VAE slicing ON")

        if enable_attention_slicing and not enable_cpu_offload:
            pass  # already handled above
        elif enable_attention_slicing:
            self.pipe.enable_attention_slicing()

        if enable_vae_slicing and enable_cpu_offload:
            self.pipe.enable_vae_slicing()

        print("[Refiner] ✅ Ready\n")

    # ──────────────────────────────────────────────────────────────
    def refine_image(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str  = "",
        strength: float       = 0.6,
        guidance_scale: float = 8.5,
        num_inference_steps: int = 30,
        seed: int             = None,
    ) -> Image.Image:
        """
        Refine a single PIL image.

        Args:
            image               : Coarse GAN output (PIL RGB).
            prompt              : Target description.
            negative_prompt     : What to suppress.
            strength            : Noise level added to input.
                                  0.4 = subtle fix | 0.7 = heavy rework.
            guidance_scale      : CFG scale (7–15 typical).
            num_inference_steps : Denoising steps.
            seed                : Reproducibility seed.

        Returns:
            Refined PIL Image.
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')

        generator = None
        if seed is not None:
            # Use CPU generator when sequential offload is active
            gen_device = 'cpu' if getattr(self, '_cpu_offload', False) else self.device
            generator = torch.Generator(device=gen_device).manual_seed(seed)

        with torch.inference_mode():
            out = self.pipe(
                prompt              = prompt,
                image               = image,
                strength            = strength,
                guidance_scale      = guidance_scale,
                num_inference_steps = num_inference_steps,
                negative_prompt     = negative_prompt,
                generator           = generator,
            )
        return out.images[0]

    # ──────────────────────────────────────────────────────────────
    def refine_batch(
        self,
        image_paths: list,
        output_dir:  str,
        prompt: str,
        negative_prompt: str  = "",
        strength: float       = 0.6,
        guidance_scale: float = 8.5,
        num_inference_steps: int = 30,
    ) -> list:
        """
        Refine a list of image files and save results.

        Args:
            image_paths : List of str/Path to GAN-generated images.
            output_dir  : Directory to save refined images.
            (remaining) : Same as refine_image.

        Returns:
            List of Path objects pointing to refined images.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = []

        for i, img_path in enumerate(tqdm(image_paths, desc="[Refiner] Refining")):
            img_path = Path(img_path)
            pil      = Image.open(img_path).convert('RGB')
            refined  = self.refine_image(
                image               = pil,
                prompt              = prompt,
                negative_prompt     = negative_prompt,
                strength            = strength,
                guidance_scale      = guidance_scale,
                num_inference_steps = num_inference_steps,
                seed                = i,     # Unique seed per image
            )
            save_path = out_dir / f"refined_{img_path.stem}.png"
            refined.save(save_path, format='PNG')
            saved.append(save_path)

        print(f"[Refiner] ✅ {len(saved)} images saved → {out_dir}")
        return saved
