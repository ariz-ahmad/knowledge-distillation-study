"""Establishes the two baselines the distillation run is measured against:
a frozen ResNet-50 (teacher) and an undistilled MobileNetV2 (student, pretrained
weights only). Both are evaluated zero-shot on ImageNette; see common.py for why
that matters when reading results.json.
"""

import json
import os

import torch
import torchvision.models as M

from common import evaluate, get_device, get_val_loader, measure_latency

if __name__ == '__main__':
    device = get_device()
    print(f"Using device: {device}")

    val_loader, idx_to_imagenet = get_val_loader("imagenette2-320/val")

    # ── Teacher: ResNet-50 ────────────────────────────────────────────────────
    print("\nLoading ResNet-50 teacher (pretrained on ImageNet)...")
    teacher = M.resnet50(weights=M.ResNet50_Weights.DEFAULT)
    teacher = teacher.to(device)
    teacher.eval()

    # Freeze all weights — teacher never updates
    for p in teacher.parameters():
        p.requires_grad_(False)
    print("✅ Teacher loaded and frozen")

    params = sum(p.numel() for p in teacher.parameters()) / 1e6
    print(f"   Parameters: {params:.1f}M")

    print("\nEvaluating teacher on ImageNette val set (zero-shot)...")
    teacher_acc = evaluate(teacher, val_loader, idx_to_imagenet, device)
    teacher_latency = measure_latency(teacher, device)

    torch.save(teacher.state_dict(), "resnet50_teacher.pth")
    teacher_size = os.path.getsize("resnet50_teacher.pth") / 1024 / 1024

    print(f"\n{'='*50}")
    print("  TEACHER (ResNet-50) BASELINE")
    print(f"{'='*50}")
    print(f"  Accuracy  : {teacher_acc*100:.2f}%")
    print(f"  Latency   : {teacher_latency:.2f}ms")
    print(f"  Size      : {teacher_size:.1f} MB")
    print(f"  Params    : {params:.1f}M")
    print(f"{'='*50}")

    # ── Student baseline: MobileNetV2, no distillation ────────────────────────
    print("\nEvaluating MobileNetV2 pretrained baseline (no distillation)...")
    student_base = M.mobilenet_v2(weights=M.MobileNet_V2_Weights.DEFAULT)
    student_base = student_base.to(device)
    student_base.eval()

    student_acc = evaluate(student_base, val_loader, idx_to_imagenet, device)
    student_latency = measure_latency(student_base, device)

    torch.save(student_base.state_dict(), "mobilenetv2_pretrained.pth")
    student_size = os.path.getsize("mobilenetv2_pretrained.pth") / 1024 / 1024
    student_params = sum(p.numel() for p in student_base.parameters()) / 1e6

    print(f"\n{'='*50}")
    print("  STUDENT PRETRAINED BASELINE (MobileNetV2)")
    print(f"{'='*50}")
    print(f"  Accuracy  : {student_acc*100:.2f}%")
    print(f"  Latency   : {student_latency:.2f}ms")
    print(f"  Size      : {student_size:.1f} MB")
    print(f"  Params    : {student_params:.1f}M")
    print(f"{'='*50}")

    print(f"\n  Teacher vs. zero-shot student gap: {(teacher_acc - student_acc)*100:.2f}%")
    print(f"  Distillation (next: distill.py) fine-tunes the student on ImageNette")
    print(f"  directly, so it is not evaluated zero-shot like these two baselines.")

    results = {
        "teacher": {
            "accuracy": round(teacher_acc * 100, 2),
            "latency_ms": round(teacher_latency, 2),
            "size_mb": round(teacher_size, 1),
            "params_m": round(params, 1)
        },
        "student_pretrained": {
            "accuracy": round(student_acc * 100, 2),
            "latency_ms": round(student_latency, 2),
            "size_mb": round(student_size, 1),
            "params_m": round(student_params, 1)
        }
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✅ Results saved to results.json")
