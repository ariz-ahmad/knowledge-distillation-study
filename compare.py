import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json

with open("results.json") as f:
    results = json.load(f)

# ── Data ─────────────────────────────────────────────────────────────────────
# "Zero-shot" models never saw ImageNette's training images; the distilled
# student was fine-tuned on them (with distillation supervision), so its
# gain reflects both the distillation signal and that training exposure.
models = {
    "ResNet-50\n(Teacher, zero-shot)":        {"acc": results["teacher"]["accuracy"],
                                     "lat": results["teacher"]["latency_ms"],
                                     "size": results["teacher"]["size_mb"],
                                     "color": "#E06C75"},
    "MobileNetV2\n(zero-shot)":    {"acc": results["student_pretrained"]["accuracy"],
                                     "lat": results["student_pretrained"]["latency_ms"],
                                     "size": results["student_pretrained"]["size_mb"],
                                     "color": "#AAAAAA"},
    "MobileNetV2\n(distilled + fine-tuned)":     {"acc": results["student_distilled"]["accuracy"],
                                     "lat": results["student_distilled"]["latency_ms"],
                                     "size": results["student_distilled"]["size_mb"],
                                     "color": "#98C379"},
}

names  = list(models.keys())
accs   = [m["acc"]  for m in models.values()]
lats   = [m["lat"]  for m in models.values()]
sizes  = [m["size"] for m in models.values()]
colors = [m["color"] for m in models.values()]

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle(
    "Knowledge Distillation Study\n"
    "ResNet-50 Teacher → MobileNetV2 Student",
    fontsize=14, fontweight="bold", y=1.02
)

# ── Chart 1: Accuracy bar chart ───────────────────────────────────────────────
ax1 = axes[0]
bars = ax1.bar(names, accs, color=colors, edgecolor="white",
               linewidth=1.5, width=0.5)
ax1.set_ylabel("Accuracy (%)", fontsize=11)
ax1.set_title("Accuracy Comparison", fontsize=11)
ax1.set_ylim(60, 105)
ax1.spines[["top", "right"]].set_visible(False)
ax1.grid(axis="y", alpha=0.3, linestyle="--")
for bar, acc in zip(bars, accs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{acc:.1f}%", ha="center", va="bottom",
             fontsize=10, fontweight="bold")

# ── Chart 2: Accuracy vs Latency scatter ──────────────────────────────────────
ax2 = axes[1]
for name, m in models.items():
    if m["lat"] == 0:
        continue
    ax2.scatter(m["lat"], m["acc"], s=m["size"]*8,
                color=m["color"], edgecolors="white",
                linewidths=1.5, zorder=5)
    ax2.annotate(
        f"{name.replace(chr(10), ' ')}\n{m['acc']:.1f}% | {m['lat']:.2f}ms",
        (m["lat"], m["acc"]),
        textcoords="offset points", xytext=(10, -8),
        fontsize=8.5, color=m["color"], fontweight="bold"
    )

ax2.set_xlabel("Inference Latency (ms)", fontsize=11)
ax2.set_ylabel("Accuracy (%)", fontsize=11)
ax2.set_title("Accuracy vs Latency\n(bubble = model size)", fontsize=11)
ax2.set_xlim(2, 7)
ax2.set_ylim(60, 105)
ax2.grid(True, alpha=0.3, linestyle="--")
ax2.spines[["top", "right"]].set_visible(False)

# ── Chart 3: Size vs Accuracy ─────────────────────────────────────────────────
ax3 = axes[2]
for name, m in models.items():
    ax3.scatter(m["size"], m["acc"], s=200,
                color=m["color"], edgecolors="white",
                linewidths=1.5, zorder=5)
    ax3.annotate(
        f"{name.replace(chr(10), ' ')}\n{m['size']:.1f} MB",
        (m["size"], m["acc"]),
        textcoords="offset points", xytext=(6, -12),
        fontsize=8.5, color=m["color"], fontweight="bold"
    )

ax3.set_xlabel("Model Size (MB)", fontsize=11)
ax3.set_ylabel("Accuracy (%)", fontsize=11)
ax3.set_title("Accuracy vs Model Size", fontsize=11)
ax3.set_ylim(60, 105)
ax3.grid(True, alpha=0.3, linestyle="--")
ax3.spines[["top", "right"]].set_visible(False)

# ── Legend ────────────────────────────────────────────────────────────────────
patches = [mpatches.Patch(color=m["color"], label=n.replace("\n", " "))
           for n, m in models.items()]
fig.legend(handles=patches, loc="lower center", ncol=3,
           bbox_to_anchor=(0.5, -0.08), fontsize=10, framealpha=0.9)

# ── Summary box ───────────────────────────────────────────────────────────────
# NOTE: the distilled student was fine-tuned on ImageNette's training images;
# the teacher and the "no distill" baseline were evaluated zero-shot and never
# saw those images. So this is "fine-tuning via distillation closes the
# zero-shot gap while staying much smaller," not "distillation beats a bigger
# model at the same task exposure" — see README.md for the full caveat.
gap_closed = results["student_distilled"]["accuracy"] - results["student_pretrained"]["accuracy"]
size_ratio = results['teacher']['size_mb'] / results['student_distilled']['size_mb']
summary = (
    f"Key Findings (student fine-tuned via distillation vs. zero-shot baselines):\n"
    f"• Fine-tuning via distillation gained +{gap_closed:.2f}% accuracy over the zero-shot "
    f"MobileNetV2 baseline\n"
    f"• The fine-tuned student now exceeds the zero-shot teacher's accuracy on this "
    f"10-class task, at {size_ratio:.1f}x smaller — expected, since only the student saw "
    f"ImageNette's training images\n"
    f"• Latency: {results['student_distilled']['latency_ms']:.2f}ms for the student "
    f"vs {results['teacher']['latency_ms']:.2f}ms for the teacher"
)
fig.text(0.5, -0.18, summary, ha="center", fontsize=9,
         bbox=dict(boxstyle="round,pad=0.5",
                   facecolor="#F4F8FC", edgecolor="#1B4F8A", alpha=0.8))

plt.tight_layout()
plt.savefig("distillation_comparison.png", dpi=150, bbox_inches="tight")
print("✅ Chart saved to distillation_comparison.png")

# ── Print final table ─────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  {'Model':<30} {'Accuracy':>10} {'Latency':>10} {'Size':>10}")
print(f"{'-'*65}")
for name, m in models.items():
    clean = name.replace('\n', ' ')
    lat = f"{m['lat']:.2f}ms" if m['lat'] > 0 else "N/A"
    print(f"  {clean:<30} {m['acc']:>9.2f}% {lat:>10} {m['size']:>9.1f}MB")
print(f"{'='*65}")
print(f"\n  Gain over zero-shot student baseline : +{gap_closed:.2f}%")
print(f"  Size vs. teacher                     : {size_ratio:.1f}x smaller")
print(f"  (Teacher and 'zero-shot' student never saw ImageNette's training")
print(f"   images; the distilled student did — see README.md.)")
