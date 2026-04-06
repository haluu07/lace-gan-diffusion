"""
Vẽ lại biểu đồ CLIP Score với trục y bắt đầu từ 0.
Chạy: python plot_clip_score.py
Output: clip_score_comparison.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Dữ liệu ──────────────────────────────────────────────
labels      = ["GAN\n(Stage 1 only)", "Refined\n(GAN + SD LoRA)"]
scores      = [21.44, 21.50]
colors      = ["#388bfd", "#3fb950"]   # xanh dương, xanh lá
bar_width   = 0.45

# ── Style ─────────────────────────────────────────────────
C_BG  = "#0d1117"
C_TXT = "#e6edf3"
C_DIM = "#8b949e"
C_ANN = "#d29922"   # vàng cho annotation +0.06

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
})

fig, ax = plt.subplots(figsize=(6, 5.5))
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)

# ── Vẽ bar ───────────────────────────────────────────────
x = np.arange(len(labels))
bars = ax.bar(x, scores, width=bar_width,
              color=colors, edgecolor="none",
              zorder=3)

# Ghi số trên mỗi bar
for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.25,
            f"{score:.2f}",
            ha='center', va='bottom',
            fontsize=12, fontweight='bold', color=C_TXT)

# ── Annotation "+0.06" giữa 2 cột ────────────────────────
diff = scores[1] - scores[0]
mid_x  = (x[0] + x[1]) / 2
mid_y  = max(scores) + 1.2

ax.annotate("", xy=(x[1], scores[1] + 0.1),
            xytext=(x[0], scores[0] + 0.1),
            arrowprops=dict(arrowstyle="<->",
                            color=C_ANN, lw=1.4,
                            mutation_scale=12))
ax.text(mid_x, scores[1] + 0.55,
        f"+{diff:.2f} (+{diff/scores[0]*100:.2f}%)",
        ha='center', va='bottom',
        fontsize=9.5, color=C_ANN, style='italic')

# ── Trục & nhãn ──────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(labels, color=C_TXT, fontsize=11)
ax.set_ylabel("CLIP Score ↑ (higher = better)",
              color=C_TXT, fontsize=11)
ax.set_ylim(0, 26)          # bắt đầu từ 0, thêm headroom
ax.set_xlim(-0.6, 1.6)

ax.tick_params(axis='y', colors=C_DIM)
ax.tick_params(axis='x', bottom=False)
for spine in ax.spines.values():
    spine.set_visible(False)

ax.yaxis.grid(True, color="#21262d", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

ax.set_title("CLIP Score: GAN vs. Refined",
             color=C_TXT, fontsize=13, fontweight='bold', pad=12)

# Ghi chú nhỏ ở dưới
fig.text(0.5, 0.01,
         "Note: improvement is +0.06 (0.28%) — marginal gain.",
         ha='center', fontsize=8.5, color=C_DIM, style='italic')

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig("clip_score_comparison.png", dpi=180,
            bbox_inches='tight', facecolor=C_BG)
plt.close()
print("✅ Saved: clip_score_comparison.png")
