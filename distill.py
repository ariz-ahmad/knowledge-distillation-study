"""Distills a frozen ResNet-50 teacher into a MobileNetV2 student on ImageNette.

Both models keep their original 1000-way ImageNet classifier head — training
only fine-tunes the student (via distillation supervision) to be confident
about the 10 ImageNet classes that ImageNette's images actually belong to.
That means this run is NOT zero-shot like the two baselines in teacher.py:
the student sees ImageNette's training images. The accuracy gain over the
undistilled baseline reflects both the distillation signal and that
target-distribution exposure — see README.md for the full explanation.
"""

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as M

from common import evaluate, get_device, get_train_loader, get_val_loader, measure_latency
from losses import FeatureDistillationHook, distillation_loss

if __name__ == '__main__':
    device = get_device()
    print(f"Using device: {device}")

    train_loader, train_idx_to_imagenet = get_train_loader("imagenette2-320/train")
    val_loader, val_idx_to_imagenet = get_val_loader("imagenette2-320/val")
    print(f"✅ Train: {len(train_loader.dataset)} images")
    print(f"✅ Val:   {len(val_loader.dataset)} images")

    # ── Teacher: frozen ResNet-50 ─────────────────────────────────────────────
    print("\nLoading frozen ResNet-50 teacher...")
    teacher = M.resnet50(weights=M.ResNet50_Weights.DEFAULT)
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print("✅ Teacher loaded and frozen")

    # ── Student: MobileNetV2, fine-tuned from pretrained weights ──────────────
    print("Loading MobileNetV2 student (pretrained weights as starting point)...")
    student = M.mobilenet_v2(weights=M.MobileNet_V2_Weights.DEFAULT)
    student = student.to(device)
    print("✅ Student loaded")

    # ── Feature distillation adapter ──────────────────────────────────────────
    # Teacher last conv: 2048 channels | Student last conv: 1280 channels
    # 1×1 conv maps student features → teacher feature space
    adapter = nn.Conv2d(1280, 2048, kernel_size=1).to(device)

    # Register hooks to capture feature maps
    hooks = FeatureDistillationHook()
    teacher.layer4.register_forward_hook(hooks.hook("teacher_feat"))
    student.features[-1].register_forward_hook(hooks.hook("student_feat"))
    print("✅ Feature distillation hooks registered")

    # ── Training setup ────────────────────────────────────────────────────────
    EPOCHS = 15
    T_temp = 4.0   # distillation temperature
    alpha  = 0.1   # weight on hard label loss (0.9 on soft)

    optimizer = torch.optim.SGD(
        list(student.parameters()) + list(adapter.parameters()),
        lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\n── Training ({EPOCHS} epochs, T={T_temp}, alpha={alpha}) ──────────")
    print(f"   Distillation loss = {alpha}×hard + {1-alpha}×soft + 0.01×feature")

    best_acc = 0
    history = []

    for epoch in range(EPOCHS):
        student.train()
        adapter.train()
        total_loss = 0
        batches = 0

        for batch_idx, (imgs, local_labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            imagenet_labels = torch.tensor(
                [train_idx_to_imagenet[i] for i in local_labels.tolist()]
            ).to(device)

            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_logits = teacher(imgs)
            teacher_feat = hooks.features["teacher_feat"].detach()

            # Student forward
            student_logits = student(imgs)
            student_feat = hooks.features["student_feat"]

            # Adapt student features to teacher space
            # Global average pool both to (B, C)
            adapted = adapter(student_feat)  # (B, 2048, H, W)
            adapted_gap = adapted.mean(dim=[2, 3])
            teacher_gap = teacher_feat.mean(dim=[2, 3])

            # Losses
            L_distill = distillation_loss(
                student_logits, teacher_logits, imagenet_labels, T=T_temp, alpha=alpha
            )
            L_feat = F.mse_loss(adapted_gap, teacher_gap)

            # Anneal feature loss weight: strong early, relax later
            feat_weight = 0.05 * (0.9 ** epoch)
            loss = L_distill + feat_weight * L_feat

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1:02d}/{EPOCHS} | "
                      f"batch {batch_idx:03d}/{len(train_loader)} | "
                      f"loss={loss.item():.4f} "
                      f"(distill={L_distill.item():.4f} feat={L_feat.item():.4f})")

        scheduler.step()

        # Evaluate
        val_acc = evaluate(student, val_loader, val_idx_to_imagenet, device)
        avg_loss = total_loss / batches
        lr = scheduler.get_last_lr()[0]

        print(f"\n  ✅ Epoch {epoch+1:02d} — "
              f"val_acc={val_acc*100:.2f}% | "
              f"avg_loss={avg_loss:.4f} | "
              f"lr={lr:.6f}\n")

        history.append({
            "epoch": epoch + 1,
            "val_acc": round(val_acc * 100, 2),
            "avg_loss": round(avg_loss, 4)
        })

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "student": student.state_dict(),
                "adapter": adapter.state_dict()
            }, "mobilenetv2_distilled_best.pth")
            print(f"     💾 Best model saved ({val_acc*100:.2f}%)\n")

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\n── Final Evaluation ─────────────────────────────────────────")
    checkpoint = torch.load("mobilenetv2_distilled_best.pth")
    student.load_state_dict(checkpoint["student"])

    final_acc = evaluate(student, val_loader, val_idx_to_imagenet, device)
    final_latency = measure_latency(student, device)
    torch.save(student.state_dict(), "mobilenetv2_distilled.pth")
    final_size = os.path.getsize("mobilenetv2_distilled.pth") / 1024 / 1024

    # Load the zero-shot baselines that teacher.py already measured, rather
    # than hardcoding them, so this always reports against the latest run.
    with open("results.json") as f:
        results = json.load(f)
    teacher_acc = results["teacher"]["accuracy"]
    teacher_size = results["teacher"]["size_mb"]
    baseline_acc = results["student_pretrained"]["accuracy"]
    baseline_size = results["student_pretrained"]["size_mb"]

    print(f"\n{'='*55}")
    print("  DISTILLATION RESULTS")
    print(f"{'='*55}")
    print(f"  Teacher  (ResNet-50, zero-shot)        : {teacher_acc:.2f}% | {teacher_size:.1f} MB")
    print(f"  Student  (MobileNetV2, zero-shot)      : {baseline_acc:.2f}% | {baseline_size:.1f} MB")
    print(f"  Student  (MobileNetV2, distilled/FT)   : {final_acc*100:.2f}% | {final_size:.1f} MB")
    print(f"{'='*55}")
    print(f"  Gain over zero-shot student baseline: +{(final_acc*100 - baseline_acc):.2f}%")
    print(f"  Latency  : {final_latency:.2f}ms")
    print(f"{'='*55}")

    results["student_distilled"] = {
        "accuracy": round(final_acc * 100, 2),
        "latency_ms": round(final_latency, 2),
        "size_mb": round(final_size, 1)
    }
    results["training_history"] = history

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("✅ Results saved to results.json")
