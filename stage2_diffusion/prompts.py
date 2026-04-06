"""
prompts.py — Curated text prompts for lace fabric generation.

Used by Stage 2 (Stable Diffusion img2img) to guide refinement.
Good prompts dramatically improve fine detail quality.
"""

# ── Positive prompt styles ────────────────────────────────────────
STYLE_PROMPTS = {
    "default": (
        "white lace fabric with intricate floral pattern, delicate threads, "
        "fine mesh holes, professional textile photography, ultra detailed, 8k"
    ),
    "vintage": (
        "antique ivory lace fabric, vintage scalloped border, delicate rose "
        "motifs, aged textile, macro photography, high detail, museum quality"
    ),
    "modern": (
        "modern black lace fabric with geometric abstract pattern, fine net "
        "structure, fashion photography, studio lighting, sharp details"
    ),
    "geometric": (
        "white lace with hexagonal mesh pattern, repeating geometric motifs, "
        "precise thread work, textile engineering, crisp ultra-detailed scan"
    ),
    "floral": (
        "cream lace fabric with large floral embroidery, lily and rose motifs, "
        "thread-level detail, luxury textile, sunlit natural background"
    ),
    "bridal": (
        "bridal white lace fabric, elegant scallop edges, pearls and flower "
        "motifs, satin ribbon trim, luxury wedding, close-up macro, 4k"
    ),
}

# ── Negative prompt (universal) ───────────────────────────────────
NEGATIVE_PROMPT = (
    "blurry, low quality, pixelated, distorted, artifact, noise, grain, "
    "watermark, text, logo, overexposed, flat, cartoonish, plastic, "
    "woven fabric, denim, cotton, no lace pattern"
)


def get_prompt(style: str = "default") -> tuple:
    """
    Return (positive_prompt, negative_prompt) for a given lace style.

    Args:
        style: One of 'default', 'vintage', 'modern',
               'geometric', 'floral', 'bridal'.

    Returns:
        Tuple of (positive_prompt: str, negative_prompt: str)
    """
    if style not in STYLE_PROMPTS:
        print(f"[Prompt] Unknown style '{style}', falling back to 'default'.")
        style = "default"
    return STYLE_PROMPTS[style], NEGATIVE_PROMPT


def list_styles() -> list:
    """Return all available style names."""
    return list(STYLE_PROMPTS.keys())


if __name__ == "__main__":
    print("Available lace styles:\n")
    for style in list_styles():
        pos, neg = get_prompt(style)
        print(f"  [{style}]\n  + {pos[:80]}…\n")
