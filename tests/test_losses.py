import torch
import torch.nn.functional as F

from losses import FeatureDistillationHook, distillation_loss


def test_alpha_one_reduces_to_plain_cross_entropy():
    torch.manual_seed(0)
    student_logits = torch.randn(8, 10)
    teacher_logits = torch.randn(8, 10)
    labels = torch.randint(0, 10, (8,))

    loss = distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=1.0)
    expected = F.cross_entropy(student_logits, labels)

    assert torch.allclose(loss, expected, atol=1e-6)


def test_alpha_zero_ignores_hard_labels():
    torch.manual_seed(0)
    student_logits = torch.randn(8, 10)
    teacher_logits = torch.randn(8, 10)
    labels_a = torch.randint(0, 10, (8,))
    labels_b = torch.randint(0, 10, (8,))

    loss_a = distillation_loss(student_logits, teacher_logits, labels_a, T=4.0, alpha=0.0)
    loss_b = distillation_loss(student_logits, teacher_logits, labels_b, T=4.0, alpha=0.0)

    assert torch.allclose(loss_a, loss_b)


def test_loss_is_zero_when_student_matches_teacher_and_labels():
    logits = torch.randn(8, 10)
    labels = logits.argmax(dim=1)
    # Make the "true" logits overwhelmingly confident so cross-entropy ~ 0 too.
    confident_logits = logits * 0 + F.one_hot(labels, 10).float() * 20

    loss = distillation_loss(confident_logits, confident_logits, labels, T=4.0, alpha=0.5)

    assert loss.item() < 1e-3


def test_loss_is_differentiable_with_respect_to_student():
    student_logits = torch.randn(4, 5, requires_grad=True)
    teacher_logits = torch.randn(4, 5)
    labels = torch.randint(0, 5, (4,))

    loss = distillation_loss(student_logits, teacher_logits, labels)
    loss.backward()

    assert student_logits.grad is not None
    assert not torch.isnan(student_logits.grad).any()


def test_feature_distillation_hook_captures_forward_output():
    hook = FeatureDistillationHook()
    layer = torch.nn.Linear(4, 4)
    layer.register_forward_hook(hook.hook("layer_out"))

    x = torch.randn(2, 4)
    out = layer(x)

    assert torch.equal(hook.features["layer_out"], out)
