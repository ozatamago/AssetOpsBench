import matplotlib.pyplot as plt
import numpy as np
from textwrap import fill

# =========================
# Fine-grained failure-mode data
# =========================
labels = [
    "1.1 Disobey Task Specification",
    "1.2 Disobey Role Specification",
    "1.3 Step Repetition",
    "1.4 Loss of Conversation History",
    "1.5 Unaware of Termination Conditions",
    "2.1 Conversation Reset",
    "2.2 Fail to Ask for Clarification",
    "2.3 Task Derailment",
    "2.4 Information Withholding",
    "2.5 Ignored Other Agent's Input",
    "2.6 Action-Reasoning Mismatch",
    "3.1 Premature Termination",
    "3.2 No or Incorrect Verification",
    "3.3 Weak Verification",
]

base = [2.11, 0.00, 35.79, 2.11, 23.16, 0.00, 24.21, 4.21, 3.16, 0.00, 4.21, 25.26, 50.53, 12.63]
spin = [2.11, 0.00, 10.53, 0.00, 15.79, 0.00, 26.32, 7.37, 1.05, 0.00, 3.16, 25.26, 47.37, 9.47]
wo_sim = [5.26, 1.05, 17.89, 0.00, 18.95, 0.00, 24.21, 8.42, 1.05, 1.05, 4.21, 37.89, 49.47, 10.53]
wo_cri = [4.21, 0.00, 12.63, 1.05, 16.84, 0.00, 26.32, 5.26, 1.05, 0.00, 4.21, 25.26, 45.26, 10.53]

# =========================
# Output directory
# =========================
out_dir = "/Users/yusuke/Desktop/Program/codabench/AssetOpsBench/benchmark/cods_track1/track1_result/trajfm_outputs"

# =========================
# Colors
# =========================
base_color   = (120/255, 175/255, 220/255)
spin_color   = (220/255, 140/255, 140/255)
wo_sim_color = (220/255, 168/255, 105/255)
wo_cri_color = (140/255, 195/255, 130/255)

bar_alpha = 0.94

# =========================
# Font sizes
# =========================
label_fs = 13
tick_fs = 12
axis_fs = 14
legend_fs = 12
cat_xtick_fs = 14

# =========================
# 1) Full fine-grained rate figure
# =========================
wrapped_labels = [fill(lbl, width=34) for lbl in labels]

y = np.arange(len(labels))
h = 0.19

fig, ax = plt.subplots(figsize=(12.8, 9.8))

ax.barh(y - 1.5 * h, base,   height=h, label="BASE",
        color=base_color, alpha=bar_alpha, edgecolor="none")
ax.barh(y - 0.5 * h, spin,   height=h, label="SPIN",
        color=spin_color, alpha=bar_alpha, edgecolor="none")
ax.barh(y + 0.5 * h, wo_sim, height=h, label="SPIN_wo_sim",
        color=wo_sim_color, alpha=bar_alpha, edgecolor="none")
ax.barh(y + 1.5 * h, wo_cri, height=h, label="SPIN_wo_cri",
        color=wo_cri_color, alpha=bar_alpha, edgecolor="none")

ax.set_yticks(y)
ax.set_yticklabels(wrapped_labels, fontsize=label_fs)
ax.invert_yaxis()

ax.set_xlabel("Failure incidence rate (%)", fontsize=axis_fs)
ax.tick_params(axis="x", labelsize=tick_fs)
ax.set_xlim(0, 55)

ax.legend(loc="upper center", ncol=4, frameon=False, fontsize=legend_fs)
ax.grid(axis="x", alpha=0.22)

# Category separators
for sep in [4.5, 10.5]:
    ax.axhline(sep, linewidth=1.0, color="gray", alpha=0.6)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

full_png_path = f"{out_dir}/fma_full_rate.png"
full_pdf_path = f"{out_dir}/fma_full_rate.pdf"
fig.savefig(full_png_path, dpi=220, bbox_inches="tight")
fig.savefig(full_pdf_path, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {full_png_path}")
print(f"Saved: {full_pdf_path}")

# =========================
# 2) Category-level rate figure for main paper
# =========================
cat_labels = [
    "1.x Spec/State",
    "2.x Interaction",
    "3.x Verification/Completion",
]

base_cat = [63.16, 35.79, 88.42]
spin_cat = [28.42, 37.89, 82.11]
wo_sim_cat = [43.16, 38.95, 97.89]
wo_cri_cat = [34.74, 36.84, 81.05]

x = np.arange(len(cat_labels))
w = 0.18

fig, ax = plt.subplots(figsize=(9.0, 5.2))

ax.bar(x - 1.5 * w, base_cat,   width=w, label="BASE",
       color=base_color, alpha=bar_alpha, edgecolor="none")
ax.bar(x - 0.5 * w, spin_cat,   width=w, label="SPIN",
       color=spin_color, alpha=bar_alpha, edgecolor="none")
ax.bar(x + 0.5 * w, wo_sim_cat, width=w, label="SPIN_wo_sim",
       color=wo_sim_color, alpha=bar_alpha, edgecolor="none")
ax.bar(x + 1.5 * w, wo_cri_cat, width=w, label="SPIN_wo_cri",
       color=wo_cri_color, alpha=bar_alpha, edgecolor="none")

ax.set_xticks(x)
ax.set_xticklabels(cat_labels, fontsize=cat_xtick_fs)
ax.set_ylabel("Failure incidence rate (%)", fontsize=axis_fs)
ax.tick_params(axis="y", labelsize=tick_fs)
ax.set_ylim(0, 105)

ax.legend(loc="upper center", ncol=4, frameon=False, fontsize=legend_fs)
ax.grid(axis="y", alpha=0.22)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

cat_png_path = f"{out_dir}/fma_category_rate.png"
cat_pdf_path = f"{out_dir}/fma_category_rate.pdf"
fig.savefig(cat_png_path, dpi=220, bbox_inches="tight")
fig.savefig(cat_pdf_path, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {cat_png_path}")
print(f"Saved: {cat_pdf_path}")