from torch.functional import F

def object_loss(targets, objects, targets_start_from):
    object_targets = targets[targets_start_from:targets_start_from + 352, :].transpose(1, 0).reshape(-1, 352)

    loss = F.binary_cross_entropy_with_logits(objects, object_targets)

    return loss
