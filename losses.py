"""The distillation objective: Hinton et al. soft-label loss plus a
feature-matching term between student and teacher activations."""

import torch.nn.functional as F


def distillation_loss(student_logits, teacher_logits, hard_labels, T: float = 4.0, alpha: float = 0.1):
    """Weighted sum of hard-label cross-entropy and soft teacher-student KL divergence.

    student_logits : (B, num_classes) raw logits from the student
    teacher_logits : (B, num_classes) raw logits from the frozen teacher
    hard_labels    : (B,) ground-truth class indices
    T              : distillation temperature — higher softens the teacher's distribution
    alpha          : weight on the hard-label loss (1 - alpha on the soft loss)
    """
    hard_loss = F.cross_entropy(student_logits, hard_labels)

    student_soft = F.log_softmax(student_logits / T, dim=1)
    teacher_soft = F.softmax(teacher_logits / T, dim=1)
    soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T ** 2)

    return alpha * hard_loss + (1.0 - alpha) * soft_loss


class FeatureDistillationHook:
    """Captures intermediate feature maps via forward hooks."""

    def __init__(self):
        self.features = {}

    def hook(self, name):
        def fn(module, input, output):
            self.features[name] = output
        return fn
